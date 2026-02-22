from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReturnContextConfig:
    """Conventions for working with a return context buffer.

    A return context is expected to be of length L + 2*O, where:
      - L = lookback window length
      - O = max_offset (max absolute integer shift allowed)

    The canonical (offset=0) window is the slice [O : O+L].
    """

    max_offset: int = 1


def window_from_context(r_context: np.ndarray, *, offset: int, cfg: ReturnContextConfig) -> np.ndarray:
    r_context = np.asarray(r_context, dtype=np.float32).reshape(-1)
    O = int(cfg.max_offset)
    if O < 0:
        raise ValueError("max_offset must be >= 0")
    L = int(r_context.shape[0] - 2 * O)
    if L <= 0:
        raise ValueError(f"Invalid context length {r_context.shape[0]} for max_offset={O}.")
    if offset < -O or offset > O:
        raise ValueError(f"offset must be in [-{O},{O}], got {offset}.")
    start = O + int(offset)
    end = start + L
    if start < 0 or end > int(r_context.shape[0]):
        raise ValueError("offset slice out of bounds (bad context construction).")
    return r_context[start:end]


def x_from_trend_and_window(trend: float, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    return np.concatenate([np.array([float(trend)], dtype=np.float32), w], axis=0).astype(np.float32, copy=False)


def x_from_context(r_context: np.ndarray, *, trend: float, offset: int, cfg: ReturnContextConfig) -> np.ndarray:
    w = window_from_context(r_context, offset=offset, cfg=cfg)
    return x_from_trend_and_window(trend, w)

