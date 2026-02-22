from __future__ import annotations

"""Minimal audio utilities for WAV IO and deterministic transforms."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import hashlib
import math

import numpy as np


# ----------------------------
# IO
# ----------------------------


def _try_soundfile_read(path: Path) -> Optional[Tuple[np.ndarray, int]]:
    try:  # pragma: no cover
        import soundfile as sf  # type: ignore

        x, sr = sf.read(str(path), always_2d=True)
        return x.astype(np.float32), int(sr)
    except Exception:
        return None


def _read_wav_wave(path: Path) -> Tuple[np.ndarray, int]:
    """Read WAV using stdlib `wave` (PCM 8/16/24/32-bit best-effort)."""
    import wave

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        x = x / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        if b.size % 3 != 0:
            raise ValueError(f"Invalid 24-bit WAV payload length: {b.size}")
        b = b.reshape(-1, 3)
        x = (
            (b[:, 0].astype(np.int32))
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        mask = 1 << 23
        x = (x ^ mask) - mask
        x = x.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        x_i = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
        x_f = np.frombuffer(raw, dtype=np.float32)
        if np.isfinite(x_f).all() and (np.max(np.abs(x_f)) <= 2.0):
            x = x_f.astype(np.float32)
        else:
            x = x_i
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if n_channels > 1:
        x = x.reshape(-1, n_channels)
    else:
        x = x.reshape(-1, 1)
    return x.astype(np.float32), int(sr)


def load_audio(
    path: str | Path,
    *,
    target_sr: Optional[int] = None,
    clip_seconds: Optional[float] = None,
    mono: bool = False,
) -> Tuple[np.ndarray, int]:
    """Load audio from file.

    Returns float32 in [-1, 1] with shape [T, C].
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    sf = _try_soundfile_read(p)
    if sf is not None:
        x, sr = sf
    else:
        x, sr = _read_wav_wave(p)

    if clip_seconds is not None:
        n = int(round(float(clip_seconds) * sr))
        x = x[: max(0, n), :]

    if mono:
        x = np.mean(x, axis=1, keepdims=True)

    if target_sr is not None and int(target_sr) != int(sr):
        x = resample_linear(x, orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)

    return x.astype(np.float32), int(sr)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    """Write 16-bit PCM WAV with minimal dependencies."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    x2 = ensure_2d(x)
    x2 = np.clip(x2, -1.0, 1.0)
    pcm = (x2 * 32767.0).astype(np.int16)

    import wave

    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


# ----------------------------
# Array helpers
# ----------------------------


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1D or 2D audio array, got shape {x.shape}")


def to_mono(x: np.ndarray) -> np.ndarray:
    x2 = ensure_2d(x)
    return np.mean(x2, axis=1).astype(np.float32)


def resample_linear(x: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    x2 = ensure_2d(np.asarray(x, dtype=np.float32))
    if orig_sr == target_sr:
        return x2

    n_in = x2.shape[0]
    n_out = int(round(n_in * float(target_sr) / float(orig_sr)))
    if n_out <= 1:
        return x2[:1, :]

    t_in = np.linspace(0.0, 1.0, n_in, endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float64)

    y = np.zeros((n_out, x2.shape[1]), dtype=np.float32)
    for c in range(x2.shape[1]):
        y[:, c] = np.interp(t_out, t_in, x2[:, c].astype(np.float64)).astype(np.float32)
    return y


def _resample_to_length(x: np.ndarray, new_len: int) -> np.ndarray:
    x2 = ensure_2d(x)
    n = x2.shape[0]
    if new_len <= 1:
        return x2[:1]
    t_in = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, new_len, endpoint=False, dtype=np.float64)
    y = np.zeros((new_len, x2.shape[1]), dtype=np.float32)
    for c in range(x2.shape[1]):
        y[:, c] = np.interp(t_out, t_in, x2[:, c].astype(np.float64)).astype(np.float32)
    return y


def _pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    x2 = ensure_2d(x)
    if x2.shape[0] == n:
        return x2
    if x2.shape[0] > n:
        return x2[:n]
    pad = np.zeros((n - x2.shape[0], x2.shape[1]), dtype=np.float32)
    return np.concatenate([x2, pad], axis=0)


# ----------------------------
# Metrics / transforms
# ----------------------------


def rms(x: np.ndarray) -> float:
    x2 = ensure_2d(x)
    return float(np.sqrt(np.mean(x2 * x2) + 1e-12))


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(max(1e-12, rms(x)))


def peak(x: np.ndarray) -> float:
    x2 = ensure_2d(x)
    return float(np.max(np.abs(x2)))


def peak_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(max(1e-12, peak(x)))


def normalize_rms_db(x: np.ndarray, target_db: float) -> np.ndarray:
    cur = rms_db(x)
    gain_db = float(target_db) - float(cur)
    gain = float(10.0 ** (gain_db / 20.0))
    return (ensure_2d(x) * gain).astype(np.float32)


def add_noise_db(x: np.ndarray, noise_db: float, *, rng: np.random.Generator) -> np.ndarray:
    x2 = ensure_2d(x)
    sigma = float(10.0 ** (float(noise_db) / 20.0))
    n = rng.normal(0.0, sigma, size=x2.shape).astype(np.float32)
    return np.clip(x2 + n, -1.0, 1.0).astype(np.float32)


def apply_lowpass(x: np.ndarray, *, sr: int, cutoff_hz: float) -> np.ndarray:
    x2 = ensure_2d(x)
    n = x2.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    mag = 1.0 / (1.0 + (freqs / float(cutoff_hz)) ** 4)
    y = np.zeros_like(x2)
    for c in range(x2.shape[1]):
        X = np.fft.rfft(x2[:, c].astype(np.float64))
        Y = X * mag
        y[:, c] = np.fft.irfft(Y, n=n).astype(np.float32)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ----------------------------
# Phase vocoder (torch)
# ----------------------------


@dataclass
class PhaseVocoderConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024


def time_stretch(x: np.ndarray, *, rate: float, cfg: Optional[PhaseVocoderConfig] = None) -> np.ndarray:
    """Time-stretch with a basic phase vocoder.

    rate > 1 speeds up (shorter), rate < 1 slows down (longer).
    """
    cfg = cfg or PhaseVocoderConfig()
    if rate <= 0:
        raise ValueError("rate must be > 0")

    import torch

    x2 = ensure_2d(x)
    if x2.shape[1] > 1:
        outs = []
        for c in range(x2.shape[1]):
            outs.append(_time_stretch_mono(x2[:, c], rate, cfg))
        y = np.stack(outs, axis=1)
    else:
        y = _time_stretch_mono(x2[:, 0], rate, cfg)[:, None]
    return y.astype(np.float32)


def _time_stretch_mono(x: np.ndarray, rate: float, cfg: PhaseVocoderConfig) -> np.ndarray:
    import torch

    xt = torch.from_numpy(x.astype(np.float32))
    window = torch.hann_window(cfg.win_length)
    D = torch.stft(
        xt,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        return_complex=True,
    )

    n_bins, n_frames = D.shape
    time_steps = torch.arange(0, n_frames, rate, dtype=torch.float32)
    n_out = int(time_steps.numel())

    omega = 2.0 * math.pi * torch.arange(0, n_bins, dtype=torch.float32) / float(cfg.n_fft)
    phi = torch.angle(D[:, 0])

    out = torch.zeros((n_bins, n_out), dtype=torch.complex64)
    for i, t in enumerate(time_steps):
        t0 = int(torch.floor(t).item())
        t1 = min(t0 + 1, n_frames - 1)
        frac = float((t - t0).item())

        D0 = D[:, t0]
        D1 = D[:, t1]
        mag = (1.0 - frac) * torch.abs(D0) + frac * torch.abs(D1)

        p0 = torch.angle(D0)
        p1 = torch.angle(D1)
        dp = p1 - p0 - omega * float(cfg.hop_length)
        dp = (dp + math.pi) % (2.0 * math.pi) - math.pi
        phi = phi + omega * float(cfg.hop_length) + dp

        out[:, i] = mag * torch.exp(1j * phi)

    out_len = int(round((x.shape[0] / float(rate))))
    y_out = torch.istft(
        out,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        length=out_len,
    )
    return y_out.detach().cpu().numpy().astype(np.float32)


def pitch_shift(x: np.ndarray, *, n_steps: float, cfg: Optional[PhaseVocoderConfig] = None) -> np.ndarray:
    """Pitch shift by n_steps semitones while preserving duration (approx)."""
    cfg = cfg or PhaseVocoderConfig()
    x2 = ensure_2d(x)
    rate = float(2.0 ** (float(n_steps) / 12.0))

    n = x2.shape[0]
    n_rs = max(2, int(round(n / rate)))
    x_rs = _resample_to_length(x2, n_rs)
    x_ts = time_stretch(x_rs, rate=1.0 / rate, cfg=cfg)
    return _pad_or_trim(x_ts, n).astype(np.float32)


def audio_hash(x: np.ndarray) -> str:
    x2 = ensure_2d(x).astype(np.float32)
    return hashlib.sha256(x2.tobytes()).hexdigest()
