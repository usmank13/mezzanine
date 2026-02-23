from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.cache import hash_dict
from ..registry import ADAPTERS
from .base import WorldAdapter
from ..utils.audio import load_audio


@dataclass
class Esc50AdapterConfig:
    csv_path: str
    audio_dir: str
    folds: Optional[List[int]] = None
    test_folds: Optional[List[int]] = None
    classes: Optional[List[str]] = None
    sr: int = 44100
    clip_seconds: float = 5.0
    mono: bool = False
    seed: int = 0
    max_files: Optional[int] = None
    train_fraction: float = 0.8
    include_audio: bool = True

    def validate(self) -> None:
        csv_path = Path(self.csv_path)
        audio_dir = Path(self.audio_dir)
        if not csv_path.exists():
            raise ValueError(f"csv_path does not exist: {csv_path}")
        if not audio_dir.exists() or not audio_dir.is_dir():
            raise ValueError(
                f"audio_dir does not exist or is not a directory: {audio_dir}"
            )
        if not (0.0 <= float(self.train_fraction) <= 1.0):
            raise ValueError("train_fraction must be within [0,1]")
        if self.sr <= 0:
            raise ValueError("sr must be > 0")
        if self.clip_seconds <= 0:
            raise ValueError("clip_seconds must be > 0")


def _parse_metadata(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "fold", "category"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"ESC-50 metadata must include {sorted(required)}")
        rows = []
        for row in reader:
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            rows.append(
                {
                    "filename": filename,
                    "fold": int(row["fold"]),
                    "category": str(row["category"]).strip(),
                }
            )
    return rows


@ADAPTERS.register(
    "esc50", description="Load ESC-50 metadata + audio as a labeled world."
)
class Esc50Adapter(WorldAdapter):
    NAME = "esc50"
    DESCRIPTION = "ESC-50 adapter (CSV + audio dir)."

    def __init__(self, cfg: Esc50AdapterConfig):
        cfg.validate()
        self.cfg = cfg

    def fingerprint(self) -> str:
        rows = _parse_metadata(Path(self.cfg.csv_path))
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        d["rows"] = rows
        return hash_dict(d)

    def _select_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cfg = self.cfg
        if cfg.folds:
            folds = {int(f) for f in cfg.folds}
            rows = [r for r in rows if int(r["fold"]) in folds]
        if cfg.classes:
            classes = {c.strip().lower() for c in cfg.classes if c.strip()}
            rows = [r for r in rows if str(r["category"]).strip().lower() in classes]
        if cfg.max_files is not None and int(cfg.max_files) < len(rows):
            rng = np.random.default_rng(int(cfg.seed))
            idx = rng.choice(len(rows), size=int(cfg.max_files), replace=False)
            rows = [rows[int(i)] for i in sorted(idx)]
        rows.sort(key=lambda r: (int(r["fold"]), r["filename"]))
        return rows

    def _label_mapping(self, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        cats = sorted({str(r["category"]).strip() for r in rows})
        return {c: i for i, c in enumerate(cats)}

    def _load_item(
        self, row: Dict[str, Any], label_map: Dict[str, int]
    ) -> Dict[str, Any]:
        cfg = self.cfg
        path = Path(cfg.audio_dir) / row["filename"]
        item: Dict[str, Any] = {
            "name": Path(row["filename"]).stem,
            "path": str(path.resolve()),
            "sr": int(cfg.sr),
            "label": int(label_map[str(row["category"]).strip()]),
            "meta": {
                "fold": int(row["fold"]),
                "category": str(row["category"]).strip(),
            },
        }
        if cfg.include_audio:
            audio, orig_sr = load_audio(
                path,
                target_sr=int(cfg.sr),
                clip_seconds=float(cfg.clip_seconds),
                mono=bool(cfg.mono),
            )
            item["audio"] = audio.astype(np.float32)
            item["meta"]["orig_sr"] = int(orig_sr)
            item["meta"]["duration_seconds"] = float(audio.shape[0]) / float(cfg.sr)
        return item

    def load(self) -> Dict[str, Any]:
        cfg = self.cfg
        rows = self._select_rows(_parse_metadata(Path(cfg.csv_path)))
        if not rows:
            raise ValueError("No rows matched ESC-50 filters.")
        label_map = self._label_mapping(rows)
        items = [self._load_item(r, label_map) for r in rows]

        for i, it in enumerate(items):
            it["id"] = int(i)

        if cfg.test_folds:
            test_folds = {int(f) for f in cfg.test_folds}
            train = [it for it in items if int(it["meta"]["fold"]) not in test_folds]
            test = [it for it in items if int(it["meta"]["fold"]) in test_folds]
        else:
            n = len(items)
            n_train = int(round(float(cfg.train_fraction) * float(n)))
            train = items[:n_train]
            test = items[n_train:]

        return {"train": train, "test": test, "labels": label_map}
