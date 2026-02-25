from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


@dataclass
class EnsembleMemberSymmetryConfig:
    """Configuration for ensemble-member exchangeability."""

    num_members: int = 50
    without_replacement: bool = True


@SYMMETRIES.register("ens_member")
class EnsembleMemberSymmetry(Symmetry):
    """Symmetry over the order/choice of exchangeable ensemble members.

    A "view" is selecting a particular member index r. The symmetry group is the
    permutation group over members; sampling a view corresponds to drawing an r.
    """

    NAME = "ens_member"
    DESCRIPTION = "Exchangeable ensemble-member symmetry: sample member indices."
    CONFIG_CLS = EnsembleMemberSymmetryConfig

    def __init__(self, cfg: EnsembleMemberSymmetryConfig):
        self.cfg = cfg

    def sample(self, x: Any, *, seed: int) -> int:
        rng = np.random.default_rng(int(seed))
        n = int(self.cfg.num_members)
        if n <= 0:
            raise ValueError("num_members must be >= 1")
        return int(rng.integers(0, n))

    def batch(self, x: Any, k: int, *, seed: int) -> list[int]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        rng = np.random.default_rng(int(seed))
        n = int(self.cfg.num_members)
        if n <= 0:
            raise ValueError("num_members must be >= 1")

        if bool(self.cfg.without_replacement) and k <= n:
            idx = rng.choice(n, size=k, replace=False)
            return [int(i) for i in idx]
        return [int(rng.integers(0, n)) for _ in range(k)]
