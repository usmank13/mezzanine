from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


@dataclass
class AngleWrapConfig:
    """Angle wrap symmetry: add integer multiples of a period.

    Useful for numerical kernels like Kepler root-finding where the mean anomaly
    M is periodic with 2π.
    """

    field: str = "M"
    period: float = float(2.0 * np.pi)
    max_k: int = 2


@SYMMETRIES.register("angle_wrap")
class AngleWrapSymmetry(Symmetry):
    CONFIG_CLS = AngleWrapConfig
    NAME = "angle_wrap"
    DESCRIPTION = (
        "Angle wrap: x[field] -> x[field] + k*period with k in [-max_k,max_k]."
    )

    def __init__(self, config: AngleWrapConfig):
        self.config = config

    def sample(self, x: Dict[str, Any], seed: int) -> Dict[str, Any]:
        if self.config.field not in x:
            return x
        rng = np.random.default_rng(int(seed))
        k = int(rng.integers(-int(self.config.max_k), int(self.config.max_k) + 1))
        out = dict(x)
        out[self.config.field] = float(out[self.config.field]) + float(
            self.config.period
        ) * float(k)
        out["_wrap_k"] = k
        return out
