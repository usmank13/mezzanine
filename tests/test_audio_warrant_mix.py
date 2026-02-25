from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mezzanine.utils.audio import write_wav


def _sine(sr: int, seconds: float, freq: float, phase: float) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)


def _square(sr: int, seconds: float, freq: float, phase: float) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return np.sign(np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)


def _write_dataset(root: Path) -> Path:
    class_sine = root / "sine"
    class_square = root / "square"
    class_sine.mkdir(parents=True, exist_ok=True)
    class_square.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    sr = 16000
    freq = 440.0
    for i in range(12):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(0.7, 0.95))
        y = amp * _sine(sr, 1.0, freq=freq, phase=phase)
        write_wav(class_sine / f"sine_{i}.wav", y, sr)
    for i in range(12):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(0.7, 0.95))
        y = amp * _square(sr, 1.0, freq=freq, phase=phase)
        write_wav(class_square / f"square_{i}.wav", y, sr)
    return root


def test_audio_warrant_mix_e2e(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path / "audio")
    out_dir = tmp_path / "out"

    import mezzanine.cli as cli

    argv = [
        "run",
        "audio_warrant_mix",
        "--out",
        str(out_dir),
        "--audio_dir",
        str(data_dir),
        "--label_from_subdir",
        "--sr",
        "16000",
        "--clip_seconds",
        "1.0",
        "--mono",
        "--lowpass_hz_min",
        "400",
        "--lowpass_hz_max",
        "800",
        "--noise_db_min",
        "-35",
        "--noise_db_max",
        "-25",
        "--k_train",
        "3",
        "--k_test",
        "4",
        "--base_steps",
        "120",
        "--student_steps",
        "120",
        "--batch_size",
        "16",
        "--device",
        "cpu",
    ]

    cli.main_args(argv)

    results = out_dir / "results.json"
    assert results.exists(), "results.json not written"
    data = json.loads(results.read_text(encoding="utf-8"))
    base_acc = float(data["metrics"]["base"]["acc"])
    base_view_acc = float(data["metrics"]["base"]["view_acc"])
    assert base_acc > 0.75, "base accuracy too low on clean audio"
    assert base_view_acc < base_acc - 0.05, (
        "playback nuisance should reduce base accuracy"
    )
