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
class LinearSystemNPZAdapterConfig:
    """Load small linear-system instances from a .npz.

    This adapter targets physics numerics kernels such as:
      - Poisson / diffusion / elasticity discretizations
      - sparse FEM/FD subproblems

    Expected keys:
      - train_A: [N, n, n] float32
      - train_b: [N, n] float32
      - train_x: [N, n] float32 solution (gold standard)
      - test_A, test_b, test_x

    The symmetry used in Mezzanine is node re-labeling (permutation similarity):
      A' = P A P^T, b' = P b, x' = P x.
    """

    path: str
    n_train: int = 50000
    n_test: int = 10000
    seed: int = 0

    # Optional: restrict to systems with n <= max_n (useful if you store mixed sizes).
    max_n: Optional[int] = None

    def validate(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"LinearSystemNPZAdapter: file not found: {p}")
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train and n_test must be non-negative")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@ADAPTERS.register("linear_system_npz")
class LinearSystemNPZAdapter(WorldAdapter):
    NAME = "linear_system_npz"
    DESCRIPTION = "Linear system (A,b)->x dataset for permutation-equivariance tests."

    def __init__(self, config: LinearSystemNPZAdapterConfig):
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
        A = self._data[f"{split}_A"].astype(np.float32)
        b = self._data[f"{split}_b"].astype(np.float32)
        x = self._data[f"{split}_x"].astype(np.float32)

        if A.ndim != 3:
            raise ValueError(f"{split}_A must be [N,n,n], got {A.shape}")
        if b.ndim != 2 or x.ndim != 2:
            raise ValueError(f"{split}_b and {split}_x must be [N,n], got {b.shape}, {x.shape}")
        if not (A.shape[0] == b.shape[0] == x.shape[0]):
            raise ValueError("Mismatched N across A,b,x")
        if not (A.shape[1] == A.shape[2] == b.shape[1] == x.shape[1]):
            raise ValueError("Mismatched n across A,b,x")

        if self.config.max_n is not None and int(A.shape[1]) > int(self.config.max_n):
            raise ValueError(
                f"Dataset n={A.shape[1]} exceeds max_n={self.config.max_n}. "
                "If your file contains mixed sizes, store separate files or implement an index table."
            )

        n_total = A.shape[0]
        n_take = min(int(n_target), int(n_total))
        idxs = deterministic_subsample_indices(n_total, n_take, seed) if n_take > 0 else np.array([], dtype=np.int64)

        out: List[Dict[str, Any]] = []
        for i in idxs:
            ii = int(i)
            out.append({"A": A[ii], "b": b[ii], "x": x[ii]})
        return out

    def load(self) -> Dict[str, Any]:
        train = self._build_split("train", self.config.n_train, self.config.seed)
        test = self._build_split("test", self.config.n_test, self.config.seed + 1)
        n = int(train[0]["A"].shape[0]) if train else (int(test[0]["A"].shape[0]) if test else 0)
        return {
            "train": train,
            "test": test,
            "meta": {
                "adapter": self.NAME,
                "cfg": asdict(self.config),
                "fingerprint": self.fingerprint(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n": n,
            },
        }
