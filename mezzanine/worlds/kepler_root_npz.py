from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..core.deterministic import deterministic_subsample_indices
from ..registry import ADAPTERS
from .base import WorldAdapter


@dataclass
class KeplerRootNPZAdapterConfig:
    """Load Kepler root-finding data from a .npz.

    This is a lightweight adapter for the numerical kernel:

        Solve Kepler's equation:  M = E - e sin(E)

    Expected keys:

      - train_e: [N] eccentricity (0 <= e < 1)
      - train_M: [N] mean anomaly (radians)
      - train_y: [N,2] target = [sin(E), cos(E)]  (preferred; 2π periodicity built in)

      - test_e, test_M, test_y

    Optional keys:
      - train_E / test_E: [N] principal-value E in radians (for extra diagnostics)

    The orbit symmetry for Mezzanine is angle wrap: M -> M + 2πk.
    """

    path: str
    n_train: int = 50000
    n_test: int = 10000
    seed: int = 0

    def validate(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"KeplerRootNPZAdapter: file not found: {p}")
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train and n_test must be non-negative")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@ADAPTERS.register("kepler_root_npz")
class KeplerRootNPZAdapter(WorldAdapter):
    NAME = "kepler_root_npz"
    DESCRIPTION = "Kepler root-finding dataset (e,M)->E targets for symmetry distillation."

    def __init__(self, config: KeplerRootNPZAdapterConfig):
        config.validate()
        self.config = config
        self.path = Path(config.path)
        self._data = _load_npz(self.path)

    def fingerprint(self) -> str:
        st = self.path.stat()
        fp = {
            "type": "kepler_root_npz",
            "path": str(self.path),
            "mtime": int(st.st_mtime),
            "size": int(st.st_size),
            "config": asdict(self.config),
            "keys": sorted(list(self._data.keys())),
        }
        return json.dumps(fp, sort_keys=True)

    def _build_split(self, split: str, n_target: int, seed: int) -> List[Dict[str, Any]]:
        e = self._data[f"{split}_e"].astype(np.float32)
        M = self._data[f"{split}_M"].astype(np.float32)
        y = self._data.get(f"{split}_y", None)
        if y is None:
            raise KeyError(f"Missing {split}_y in {self.path}. Expected target [sinE, cosE].")
        y = y.astype(np.float32)
        E = self._data.get(f"{split}_E", None)
        if E is not None:
            E = E.astype(np.float32)

        if e.ndim != 1 or M.ndim != 1:
            raise ValueError(f"{split}_e and {split}_M must be 1D, got {e.shape}, {M.shape}")
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"{split}_y must be [N,2], got {y.shape}")
        if not (len(e) == len(M) == len(y)):
            raise ValueError(f"Mismatched lengths: e={len(e)} M={len(M)} y={len(y)}")

        n_total = len(e)
        n_take = min(int(n_target), int(n_total))
        idxs = deterministic_subsample_indices(n_total, n_take, seed) if n_take > 0 else np.array([], dtype=np.int64)

        out: List[Dict[str, Any]] = []
        for i in idxs:
            ex: Dict[str, Any] = {
                "e": float(e[int(i)]),
                "M": float(M[int(i)]),
                "y": y[int(i)],
            }
            if E is not None:
                ex["E"] = float(E[int(i)])
            out.append(ex)
        return out

    def load(self) -> Dict[str, Any]:
        train = self._build_split("train", self.config.n_train, self.config.seed)
        test = self._build_split("test", self.config.n_test, self.config.seed + 1)
        return {
            "train": train,
            "test": test,
            "meta": {
                "adapter": self.NAME,
                "cfg": asdict(self.config),
                "fingerprint": self.fingerprint(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            },
        }
