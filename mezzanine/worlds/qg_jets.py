from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

from ..registry import ADAPTERS
from .base import WorldAdapter


@dataclass(frozen=True)
class QGJetsAdapterConfig:
    """Configuration for the EnergyFlow quark/gluon jet dataset.

    This uses EnergyFlow's `qg_jets` dataset (Pythia/Herwig), which is packaged as
    a set of cached files automatically downloaded on first use.

    Data schema per example:
      - particles: float32 array [P,4] with columns (pt, y, phi, pid)
      - label: int (quark=1, gluon=0)

    Notes:
      - `pad=True` is always used so `particles` has a fixed max length with
        trailing all-zero rows.
      - This adapter purposely does *not* sort or otherwise canonicalize the
        constituent order. If the source ordering is systematic, that is part of
        the realism; permutation is treated as a symmetry.
    """

    num_data: int = 50000
    generator: str = "pythia"  # or 'herwig'
    with_bc: bool = False
    cache_dir: str = "~/.energyflow"

    n_train: int = 20000
    n_test: int = 5000

    seed: int = 0


@ADAPTERS.register(
    "qg_jets_energyflow",
    description="Quark/gluon jets from EnergyFlow (Pythia/Herwig), padded particle clouds.",
)
class QGJetsAdapter(WorldAdapter):
    NAME = "qg_jets_energyflow"
    DESCRIPTION = (
        "Quark/gluon jets from EnergyFlow (Pythia/Herwig), padded particle clouds."
    )

    def __init__(self, cfg: QGJetsAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self.cfg), sort_keys=True).encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()[:16]
        return f"{self.NAME}:{h}"

    def load(self) -> Dict[str, Any]:
        try:
            from energyflow import qg_jets  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "EnergyFlow is required for QGJetsAdapter. Install with: pip install energyflow"
            ) from e

        num_data = int(self.cfg.num_data)
        if num_data <= 0:
            raise ValueError("num_data must be positive")

        X, y = qg_jets.load(
            num_data=num_data,
            pad=True,
            ncol=4,
            generator=str(self.cfg.generator),
            with_bc=bool(self.cfg.with_bc),
            cache_dir=str(self.cfg.cache_dir),
        )

        # X: [N, Pmax, 4] float
        # y: [N] int (quark=1, gluon=0)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        N = int(X.shape[0])
        if N != int(y.shape[0]):
            raise RuntimeError(
                f"EnergyFlow returned mismatched shapes: X={X.shape}, y={y.shape}"
            )

        n_train = int(self.cfg.n_train)
        n_test = int(self.cfg.n_test)
        if n_train <= 0 or n_test <= 0:
            raise ValueError("n_train and n_test must be positive")
        if n_train + n_test > N:
            raise ValueError(
                f"Requested n_train+n_test={n_train + n_test} but only have N={N}. "
                "Increase num_data or reduce n_train/n_test."
            )

        rng = np.random.default_rng(int(self.cfg.seed))
        idx = rng.permutation(N)
        idx_tr = idx[:n_train]
        idx_te = idx[n_train : n_train + n_test]

        def make_examples(idxs: np.ndarray) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for i in idxs.tolist():
                out.append(
                    {
                        "particles": X[i],
                        "label": int(y[i]),
                        "id": int(i),
                    }
                )
            return out

        train = make_examples(idx_tr)
        test = make_examples(idx_te)

        meta = {
            "adapter": self.NAME,
            "cfg": asdict(self.cfg),
            "dataset": {
                "name": "energyflow.qg_jets",
                "generator": str(self.cfg.generator),
                "with_bc": bool(self.cfg.with_bc),
                "label": {"0": "gluon", "1": "quark"},
                "particle_columns": ["pt", "y", "phi", "pid"],
            },
        }

        return {"train": train, "test": test, "meta": meta}
