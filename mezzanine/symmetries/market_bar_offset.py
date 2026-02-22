from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..utils.market_features import ReturnContextConfig, x_from_context
from .base import Symmetry


@dataclass
class MarketBarOffsetConfig:
    """Integer bar/window alignment offset.

    Offsets are sampled in:
      - [-O, 0] by default (stale-only, avoids lookahead)
      - [-O, +O] if allow_positive=True
    """

    max_offset: int = 1
    allow_positive: bool = False


class MarketBarOffsetSymmetry(Symmetry):
    NAME = "market_bar_offset"
    DESCRIPTION = "Shift return-window alignment by an integer offset (simulates bar alignment / feed staleness)."

    def __init__(self, cfg: MarketBarOffsetConfig):
        self.cfg = cfg
        self._ctx_cfg = ReturnContextConfig(max_offset=int(cfg.max_offset))

    def sample(self, x: Dict[str, Any], *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        O = int(self.cfg.max_offset)
        if O <= 0:
            off = 0
        elif bool(self.cfg.allow_positive):
            off = int(rng.integers(-O, O + 1))
        else:
            off = int(rng.integers(-O, 1))  # [-O, 0]

        r_context = np.asarray(x["r_context"], dtype=np.float32)
        trend = float(x.get("trend", 0.0))

        out = dict(x)
        out["bar_offset"] = off
        out["x"] = x_from_context(r_context, trend=trend, offset=off, cfg=self._ctx_cfg)
        return out


# Register
from ..registry import SYMMETRIES

SYMMETRIES.register("market_bar_offset")(MarketBarOffsetSymmetry)

