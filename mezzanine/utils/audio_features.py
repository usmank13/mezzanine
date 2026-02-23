"""Lightweight audio feature extraction for small demos/tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .audio import ensure_2d, to_mono


@dataclass
class AudioFeatureConfig:
    sr: int
    n_bands: int = 4
    rolloff_frac: float = 0.85
    include_zcr: bool = True
    include_rms: bool = True
    include_centroid: bool = True
    include_rolloff: bool = True
    include_band_energies: bool = True


def _spectrum(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    x_m = to_mono(x)
    n = x_m.shape[0]
    mag = np.abs(np.fft.rfft(x_m.astype(np.float64)))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    return freqs, mag


def extract_features(x: np.ndarray, cfg: AudioFeatureConfig) -> np.ndarray:
    x2 = ensure_2d(x)
    feats = []

    if cfg.include_rms:
        feats.append(float(np.sqrt(np.mean(x2 * x2) + 1e-12)))

    if cfg.include_zcr:
        mono = to_mono(x2)
        s = np.sign(mono)
        feats.append(float(np.mean(s[1:] != s[:-1])))

    freqs, mag = _spectrum(x2, cfg.sr)
    mag_sum = float(np.sum(mag) + 1e-12)

    if cfg.include_centroid:
        centroid = float(np.sum(freqs * mag) / mag_sum)
        feats.append(centroid)

    if cfg.include_rolloff:
        cumulative = np.cumsum(mag)
        idx = int(np.searchsorted(cumulative, cfg.rolloff_frac * cumulative[-1]))
        rolloff = float(freqs[min(idx, len(freqs) - 1)])
        feats.append(rolloff)

    if cfg.include_band_energies:
        n_bands = max(1, int(cfg.n_bands))
        band_edges = np.linspace(0, len(freqs), n_bands + 1, dtype=int)
        for i in range(n_bands):
            lo, hi = band_edges[i], band_edges[i + 1]
            feats.append(float(np.mean(mag[lo:hi]) if hi > lo else 0.0))

    return np.array(feats, dtype=np.float32)
