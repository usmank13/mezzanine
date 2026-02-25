from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


@dataclass
class FieldCodecConfig:
    """Noisy representation codec applied in standardized space.

    Operations (applied in order):
      - optional clipping
      - optional fp16 roundtrip
      - uniform quantization to a random bit depth
      - optional Gaussian noise

    If x is a dict, we read/write x["field"]. Otherwise we treat x as an array.
    """

    clip: float = 6.0
    fp16_prob: float = 0.5
    quant_bits_min: int = 8
    quant_bits_max: int = 10
    noise_std_min: float = 0.00
    noise_std_max: float = 0.05


@SYMMETRIES.register("field_codec")
class FieldCodecSymmetry(Symmetry):
    NAME = "field_codec"
    DESCRIPTION = "fp16 roundtrip + uniform quantization + Gaussian noise (apply in standardized space)."
    CONFIG_CLS = FieldCodecConfig

    def __init__(self, cfg: FieldCodecConfig | None = None):
        self.cfg = cfg or FieldCodecConfig()

    def sample(self, x: Any, *, seed: int) -> Any:
        rng = np.random.default_rng(int(seed))

        if isinstance(x, dict):
            if "field" not in x:
                return x
            y = np.asarray(x["field"], dtype=np.float32)
        else:
            y = np.asarray(x, dtype=np.float32)

        clip = float(self.cfg.clip)
        if clip > 0:
            y = np.clip(y, -clip, clip)

        if float(self.cfg.fp16_prob) > 0 and float(rng.random()) < float(
            self.cfg.fp16_prob
        ):
            y = y.astype(np.float16).astype(np.float32)

        b0 = int(min(self.cfg.quant_bits_min, self.cfg.quant_bits_max))
        b1 = int(max(self.cfg.quant_bits_min, self.cfg.quant_bits_max))
        bits = int(rng.integers(b0, b1 + 1))
        if bits > 1:
            levels = (1 << bits) - 1
            if clip <= 0:
                clip = float(np.max(np.abs(y)) + 1e-6)
            t = (np.clip(y, -clip, clip) / clip) * 0.5 + 0.5
            t = np.round(t * levels) / float(levels)
            y = (t - 0.5) * 2.0 * clip

        s0 = float(min(self.cfg.noise_std_min, self.cfg.noise_std_max))
        s1 = float(max(self.cfg.noise_std_min, self.cfg.noise_std_max))
        if s1 > 0:
            std = float(rng.uniform(s0, s1))
            if std > 0:
                y = y + rng.normal(0.0, std, size=y.shape).astype(np.float32)

        if isinstance(x, dict):
            out: Dict[str, Any] = dict(x)
            out["field"] = y
            return out
        return y
