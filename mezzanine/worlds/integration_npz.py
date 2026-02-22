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
class IntegrationNPZAdapterConfig:
    """Load 1D periodic integration problems from a .npz.

    Expected keys:
      - train_f: [N, L] samples of a periodic function f on a uniform grid
      - train_y: [N, 1] or [N] integral value (e.g., \\int f dx or mean(f))
      - test_f, test_y

    Optional:
      - dx: scalar spacing used when computing the integral (stored for reference)

    Symmetry: circular shift of the sampled grid (periodic translation).
    """

    path: str
    n_train: int = 50000
    n_test: int = 10000
    seed: int = 0

    def validate(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"IntegrationNPZAdapter: file not found: {p}")
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train and n_test must be non-negative")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@ADAPTERS.register("integration_npz")
class IntegrationNPZAdapter(WorldAdapter):
    NAME = "integration_npz"
    DESCRIPTION = "1D periodic integration dataset (circular-shift symmetry)."

    def __init__(self, config: IntegrationNPZAdapterConfig):
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

    def _build_split(self, split: str, n_target: int, seed: int) -> List[Dict[str, Any]]:
        f = self._data[f"{split}_f"].astype(np.float32)
        y = self._data[f"{split}_y"].astype(np.float32)
        if y.ndim == 1:
            y = y[:, None]
        if f.ndim != 2:
            raise ValueError(f"{split}_f must be [N,L], got {f.shape}")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(f"{split}_y must be [N,1] or [N], got {y.shape}")
        if f.shape[0] != y.shape[0]:
            raise ValueError("Mismatched N between f and y")

        n_total = f.shape[0]
        n_take = min(int(n_target), int(n_total))
        idxs = deterministic_subsample_indices(n_total, n_take, seed) if n_take > 0 else np.array([], dtype=np.int64)

        dx = float(self._data.get("dx", np.array([np.nan], dtype=np.float32)).reshape(-1)[0])
        out: List[Dict[str, Any]] = []
        for i in idxs:
            out.append({"f": f[int(i)], "y": y[int(i)], "dx": dx})
        return out

    def load(self) -> Dict[str, Any]:
        train = self._build_split("train", self.config.n_train, self.config.seed)
        test = self._build_split("test", self.config.n_test, self.config.seed + 1)
        L = int(train[0]["f"].shape[0]) if train else (int(test[0]["f"].shape[0]) if test else 0)
        return {
            "train": train,
            "test": test,
            "meta": {
                "adapter": self.NAME,
                "cfg": asdict(self.config),
                "fingerprint": self.fingerprint(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "L": L,
            },
        }
