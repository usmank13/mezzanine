from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.deterministic import deterministic_subsample_indices
from ..registry import ADAPTERS
from .base import WorldAdapter


@dataclass
class ODENPZAdapterConfig:
    """Load generic ODE trajectories from a .npz.

    This adapter is intentionally similar to `hamiltonian_npz` but does not
    assume Hamiltonian structure; it's for numerical ODE benchmarks.

    Expected keys (minimum):
      - train_x: [n_traj, T, D]
      - test_x:  [n_traj, T, D]

    Optional keys:
      - train_t / test_t: [T] or [n_traj, T]
      - train_system_id / test_system_id: [n_traj]
      - dt: scalar float

    The adapter samples transitions (x_t -> x_{t+1}). If include_indices=True,
    each example includes {traj_idx, t_idx} so recipes can evaluate rollouts
    against the stored ground-truth trajectories.
    """

    path: str
    n_train: int = 50000
    n_test: int = 10000
    seed: int = 0
    include_time: bool = True
    include_indices: bool = True

    def validate(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"ODENPZAdapter: file not found: {p}")
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train and n_test must be non-negative")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        out: Dict[str, np.ndarray] = {}
        for k in z.files:
            out[k] = z[k]
        return out


def _traj_time(
    t_arr: Optional[np.ndarray], traj_idx: int, t_idx: int
) -> Optional[float]:
    if t_arr is None:
        return None
    if t_arr.ndim == 1:
        return float(t_arr[t_idx])
    if t_arr.ndim == 2:
        return float(t_arr[traj_idx, t_idx])
    raise ValueError(f"Unexpected time array shape: {t_arr.shape}")


@ADAPTERS.register("ode_npz")
class ODENPZAdapter(WorldAdapter):
    NAME = "ode_npz"
    DESCRIPTION = "Generic ODE trajectory transitions (for time-origin symmetry tests)."

    def __init__(self, config: ODENPZAdapterConfig):
        config.validate()
        self.config = config
        self.path = Path(config.path)
        self._data = _load_npz(self.path)

    def fingerprint(self) -> str:
        st = self.path.stat()
        fp = {
            "type": self.NAME,
            "path": str(self.path),
            "mtime": int(st.st_mtime),
            "size": int(st.st_size),
            "config": asdict(self.config),
            "keys": sorted(list(self._data.keys())),
        }
        return json.dumps(fp, sort_keys=True)

    def _build_split(
        self, split: str, n_target: int, seed: int
    ) -> List[Dict[str, Any]]:
        x = self._data[f"{split}_x"].astype(np.float32)
        t = self._data.get(f"{split}_t", None)
        if t is not None:
            t = t.astype(np.float32)
        system_id = self._data.get(f"{split}_system_id", None)
        if system_id is not None:
            system_id = system_id.astype(np.int64)

        if x.ndim != 3:
            raise ValueError(f"{split}_x must be [n_traj,T,D], got {x.shape}")
        n_traj, T, _D = x.shape
        if T < 2:
            raise ValueError("Trajectories must have length >= 2")

        pool = n_traj * (T - 1)
        n_take = min(int(n_target), int(pool))
        idxs = (
            deterministic_subsample_indices(pool, n_take, seed)
            if n_take > 0
            else np.array([], dtype=np.int64)
        )

        out: List[Dict[str, Any]] = []
        for flat in idxs:
            traj_idx = int(flat // (T - 1))
            t_idx = int(flat % (T - 1))
            ex: Dict[str, Any] = {
                "state": x[traj_idx, t_idx],
                "next_state": x[traj_idx, t_idx + 1],
            }
            if self.config.include_time:
                ex["t"] = _traj_time(t, traj_idx, t_idx)
            if system_id is not None:
                ex["system_id"] = int(system_id[traj_idx])
            if self.config.include_indices:
                ex["traj_idx"] = traj_idx
                ex["t_idx"] = t_idx
            out.append(ex)
        return out

    def load(self) -> Dict[str, Any]:
        train = self._build_split("train", self.config.n_train, self.config.seed)
        test = self._build_split("test", self.config.n_test, self.config.seed + 1)
        dt = float(
            self._data.get("dt", np.array([np.nan], dtype=np.float32)).reshape(-1)[0]
        )
        return {
            "train": train,
            "test": test,
            "meta": {
                "adapter": self.NAME,
                "cfg": asdict(self.config),
                "fingerprint": self.fingerprint(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "dt": dt,
                "path": str(self.path),
            },
        }
