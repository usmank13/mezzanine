from __future__ import annotations

"""Audio playback symmetry (nuisance mapped to SYMMETRIES).

This symmetry simulates mild playback variation: gain, noise, mono fold,
low-pass filtering, and small time-stretch.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Symmetry
from ..utils.audio import ensure_2d, add_noise_db, apply_lowpass, _resample_to_length, _pad_or_trim


@dataclass
class AudioPlaybackConfig:
    sr: int = 48000
    gain_db_min: float = -3.0
    gain_db_max: float = 3.0
    noise_db_min: float = -45.0
    noise_db_max: float = -30.0
    mono_prob: float = 0.2
    lowpass_hz_min: float = 6000.0
    lowpass_hz_max: float = 12000.0
    time_stretch_min: float = 0.95
    time_stretch_max: float = 1.05


class AudioPlaybackSymmetry(Symmetry):
    """Playback-chain nuisance mapped into the Symmetry registry."""
    NAME = "audio_playback"
    DESCRIPTION = "Playback nuisance: gain/noise/mono/lowpass/time-stretch."

    def __init__(self, cfg: AudioPlaybackConfig = AudioPlaybackConfig()):
        self.cfg = cfg

    def sample(self, x: Any, *, seed: int) -> Any:
        rng = np.random.default_rng(seed)

        if isinstance(x, dict):
            audio = x.get("audio")
            if audio is None:
                return x
            sr = int(x.get("sr", self.cfg.sr))
        else:
            audio = x
            sr = int(self.cfg.sr)

        y = ensure_2d(np.asarray(audio, dtype=np.float32))
        n, ch = y.shape

        gain_db = float(rng.uniform(self.cfg.gain_db_min, self.cfg.gain_db_max))
        gain = float(10.0 ** (gain_db / 20.0))
        y = np.clip(y * gain, -1.0, 1.0)

        if rng.random() < float(self.cfg.mono_prob):
            mono = np.mean(y, axis=1, keepdims=True)
            y = np.repeat(mono, ch, axis=1) if ch > 1 else mono

        noise_db = float(rng.uniform(self.cfg.noise_db_min, self.cfg.noise_db_max))
        y = add_noise_db(y, noise_db, rng=rng)

        cutoff = float(rng.uniform(self.cfg.lowpass_hz_min, self.cfg.lowpass_hz_max))
        y = apply_lowpass(y, sr=sr, cutoff_hz=cutoff)

        rate = float(rng.uniform(self.cfg.time_stretch_min, self.cfg.time_stretch_max))
        if abs(rate - 1.0) > 1e-3:
            y_rs = _resample_to_length(y, int(round(n / rate)))
            y = _pad_or_trim(y_rs, n)

        if isinstance(x, dict):
            out = dict(x)
            out["audio"] = y.astype(np.float32)
            out["sr"] = sr
            return out

        return y.astype(np.float32)


from ..registry import SYMMETRIES
SYMMETRIES.register("audio_playback")(AudioPlaybackSymmetry)
