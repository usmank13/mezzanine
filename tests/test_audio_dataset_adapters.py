from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mezzanine.utils.audio import write_wav
from mezzanine.worlds.esc50 import Esc50Adapter, Esc50AdapterConfig
from mezzanine.worlds.urbansound8k import UrbanSound8KAdapter, UrbanSound8KAdapterConfig


def _sine(sr: int, seconds: float, freq: float) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _read_fixture(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_esc50_adapter_with_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "esc50_subset.csv"
    rows = _read_fixture(fixture)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        y = _sine(16000, 1.0, 440.0)
        write_wav(audio_dir / row["filename"], y, 16000)

    csv_path = tmp_path / "esc50.csv"
    csv_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = Esc50AdapterConfig(
        csv_path=str(csv_path),
        audio_dir=str(audio_dir),
        sr=16000,
        clip_seconds=1.0,
        test_folds=[2],
        include_audio=True,
    )
    world = Esc50Adapter(cfg).load()

    assert len(world["train"]) == 2
    assert len(world["test"]) == 2
    assert len(world["labels"]) == 2


def test_urbansound_adapter_with_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "urbansound8k_subset.csv"
    rows = _read_fixture(fixture)

    audio_root = tmp_path / "audio"
    (audio_root / "fold1").mkdir(parents=True, exist_ok=True)
    (audio_root / "fold2").mkdir(parents=True, exist_ok=True)
    for row in rows:
        y = _sine(16000, 1.0, 880.0)
        fold_dir = audio_root / f"fold{row['fold']}"
        write_wav(fold_dir / row["slice_file_name"], y, 16000)

    csv_path = tmp_path / "urbansound.csv"
    csv_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = UrbanSound8KAdapterConfig(
        csv_path=str(csv_path),
        audio_root=str(audio_root),
        sr=16000,
        clip_seconds=1.0,
        test_folds=[2],
        include_audio=True,
    )
    world = UrbanSound8KAdapter(cfg).load()

    assert len(world["train"]) == 2
    assert len(world["test"]) == 2
    assert len(world["labels"]) == 2
