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
class EigenNPZAdapterConfig:
    """Load eigenvalue benchmark instances from a .npz.

    Expected keys:
      - train_A: [N,n,n] float32 (typically symmetric / Hermitian)
      - train_eval: [N,k] float32 eigenvalues (e.g., lowest k or highest k)
      - test_A, test_eval

    Optional keys:
      - train_evec: [N,n,k] eigenvectors (for more advanced recipes)

    Symmetry: permutation similarity A' = P A P^T should leave eigenvalues
    unchanged; eigenvectors permute accordingly.
    """

    path: str
    n_train: int = 50000
    n_test: int = 10000
    seed: int = 0

    def validate(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"EigenNPZAdapter: file not found: {p}")
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train and n_test must be non-negative")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@ADAPTERS.register("eigen_npz")
class EigenNPZAdapter(WorldAdapter):
    NAME = "eigen_npz"
    DESCRIPTION = "Eigenvalue benchmark instances (A->evals) with permutation symmetry."

    def __init__(self, config: EigenNPZAdapterConfig):
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
        evals = self._data.get(f"{split}_eval", None)
        if evals is None:
            raise KeyError(f"Missing {split}_eval in {self.path}")
        evals = evals.astype(np.float32)
        evecs = self._data.get(f"{split}_evec", None)
        if evecs is not None:
            evecs = evecs.astype(np.float32)

        if A.ndim != 3:
            raise ValueError(f"{split}_A must be [N,n,n], got {A.shape}")
        if evals.ndim != 2:
            raise ValueError(f"{split}_eval must be [N,k], got {evals.shape}")
        if A.shape[0] != evals.shape[0]:
            raise ValueError("Mismatched N between A and eval")

        n_total = int(A.shape[0])
        n_take = min(int(n_target), n_total)
        idxs = deterministic_subsample_indices(n_total, n_take, seed) if n_take > 0 else np.array([], dtype=np.int64)

        out: List[Dict[str, Any]] = []
        for i in idxs:
            ii = int(i)
            ex: Dict[str, Any] = {"A": A[ii], "eval": evals[ii]}
            if evecs is not None:
                ex["evec"] = evecs[ii]
            out.append(ex)
        return out

    def load(self) -> Dict[str, Any]:
        train = self._build_split("train", self.config.n_train, self.config.seed)
        test = self._build_split("test", self.config.n_test, self.config.seed + 1)
        n = int(train[0]["A"].shape[0]) if train else (int(test[0]["A"].shape[0]) if test else 0)
        k = int(train[0]["eval"].shape[0]) if train else (int(test[0]["eval"].shape[0]) if test else 0)
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
                "k": k,
            },
        }
