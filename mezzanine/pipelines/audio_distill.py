"""Audio distillation helpers used by audio recipes."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ..symmetries.base import Symmetry
from ..utils.audio_features import AudioFeatureConfig, extract_features


def view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def build_views(
    examples: List[Any],
    *,
    symmetry: Symmetry,
    seed: int,
    k: int,
) -> List[List[Any]]:
    """Return list of K view-lists, each view-list is examples for all items."""
    if k <= 0:
        raise ValueError("k must be >= 1")

    views: List[List[Any]] = [list(examples)]
    if k == 1:
        return views

    for j in range(1, k):
        vj: List[Any] = []
        for i, ex in enumerate(examples):
            vj.append(symmetry.sample(ex, seed=view_seed(seed, i, j)))
        views.append(vj)
    return views


def encode_examples(examples: List[Any], cfg: AudioFeatureConfig) -> np.ndarray:
    feats = []
    for ex in examples:
        if isinstance(ex, dict):
            audio = ex.get("audio")
            if audio is None:
                raise ValueError("Example missing 'audio' field")
            sr = int(ex.get("sr", cfg.sr))
        else:
            audio = ex
            sr = int(cfg.sr)
        fcfg = AudioFeatureConfig(
            sr=sr,
            n_bands=cfg.n_bands,
            rolloff_frac=cfg.rolloff_frac,
            include_zcr=cfg.include_zcr,
            include_rms=cfg.include_rms,
            include_centroid=cfg.include_centroid,
            include_rolloff=cfg.include_rolloff,
            include_band_energies=cfg.include_band_energies,
        )
        feats.append(extract_features(audio, fcfg))
    return np.stack(feats, axis=0).astype(np.float32)
