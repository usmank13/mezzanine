from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


@dataclass
class PeriodicTranslationConfig:
    """Periodic translation symmetry (np.roll) for field-valued examples.

    This is used for PDE/operator-learning tasks on periodic domains.

    - `keys` indicates which tensors to shift (e.g., ["x", "y"]).
    - `axes` indicates which axes are spatial (e.g., [0] for 1D, [0,1] for 2D,
      or [1,2] if tensors are [C,H,W]).
    """

    keys: List[str] = None  # type: ignore[assignment]
    axes: List[int] = None  # type: ignore[assignment]
    max_shift: int = 8

    def __post_init__(self):
        if self.keys is None:
            self.keys = ["x", "y"]
        if self.axes is None:
            self.axes = [0]


@SYMMETRIES.register("periodic_translation")
class PeriodicTranslationSymmetry(Symmetry):
    CONFIG_CLS = PeriodicTranslationConfig
    NAME = "periodic_translation"
    DESCRIPTION = (
        "Periodic translation of specified keys along spatial axes (stores _shifts)."
    )

    def __init__(self, config: PeriodicTranslationConfig):
        self.config = config

    def sample(self, x: Dict[str, Any], seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        shifts = [
            int(
                rng.integers(
                    -int(self.config.max_shift), int(self.config.max_shift) + 1
                )
            )
            for _ in self.config.axes
        ]

        out = dict(x)
        out["_shifts"] = shifts
        for k in self.config.keys:
            if k not in out:
                continue
            arr = np.asarray(out[k])
            rolled = arr
            for ax, sh in zip(self.config.axes, shifts):
                rolled = np.roll(rolled, shift=sh, axis=int(ax))
            out[k] = rolled
        return out
