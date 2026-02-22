from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


@dataclass
class CircularShiftConfig:
    """Circular shift (periodic translation) symmetry for 1D sampled signals."""

    key: str = "f"
    max_shift: int = 32
    axis: int = 0


@SYMMETRIES.register("circular_shift")
class CircularShiftSymmetry(Symmetry):
    CONFIG_CLS = CircularShiftConfig
    NAME = "circular_shift"
    DESCRIPTION = "Circular shift: roll x[key] by a random integer shift (stores _shift)."

    def __init__(self, config: CircularShiftConfig):
        self.config = config

    def sample(self, x: Dict[str, Any], seed: int) -> Dict[str, Any]:
        if self.config.key not in x:
            return x
        arr = np.asarray(x[self.config.key])
        rng = np.random.default_rng(int(seed))
        s = int(rng.integers(-int(self.config.max_shift), int(self.config.max_shift) + 1))
        out = dict(x)
        out[self.config.key] = np.roll(arr, shift=s, axis=int(self.config.axis))
        out["_shift"] = s
        return out
