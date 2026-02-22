from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


@dataclass
class TimeOriginShiftConfig:
    """Shift the (arbitrary) choice of time origin.

    This is a gauge/reference-frame symmetry: for autonomous dynamics, the
    physics is invariant under t -> t + c. If a model is given absolute time,
    it can spuriously condition on it.
    """

    # Sample c ~ Uniform[-max_shift, max_shift]
    max_shift: float = 10.0

    # Optional: keep shifted time in [t_min, t_max] (avoids OOD extrapolation for time-conditioned models).
    t_min: Optional[float] = None
    t_max: Optional[float] = None


@SYMMETRIES.register("time_origin_shift")
class TimeOriginShiftSymmetry(Symmetry):
    CONFIG_CLS = TimeOriginShiftConfig

    def __init__(self, config: TimeOriginShiftConfig):
        self.config = config

    def sample(self, x: Dict[str, Any], seed: int) -> Dict[str, Any]:
        if "t" not in x:
            return x

        rng = np.random.default_rng(int(seed))

        out = dict(x)
        t = out["t"]
        if isinstance(t, (float, int, np.floating, np.integer)):
            t0 = float(t)
            # If bounds are provided, sample c so that t' remains in-range.
            if (self.config.t_min is not None) and (self.config.t_max is not None):
                lo = max(-float(self.config.max_shift), float(self.config.t_min) - t0)
                hi = min(float(self.config.max_shift), float(self.config.t_max) - t0)
                c = 0.0 if lo > hi else float(rng.uniform(lo, hi))
            else:
                c = float(rng.uniform(-self.config.max_shift, self.config.max_shift))
            out["t"] = t0 + c
        else:
            c = float(rng.uniform(-self.config.max_shift, self.config.max_shift))
            arr = np.asarray(t)
            out["t"] = (arr + c).astype(arr.dtype, copy=False)
        return out
