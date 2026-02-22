from __future__ import annotations

"""Audio folder world adapter.

This adapter treats a folder of WAV files as a dataset. If label_from_subdir
is True, the parent folder name is used as the class label.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.cache import hash_dict
from ..registry import ADAPTERS
from .base import WorldAdapter
from ..utils.audio import load_audio


@dataclass
class AudioFolderAdapterConfig:
    audio_dir: str
    pattern: str = "*.wav"
    recursive: bool = False

    sr: int = 48000
    clip_seconds: float = 5.0
    mono: bool = True

    seed: int = 0
    max_files: Optional[int] = None
    train_fraction: float = 0.8

    include_audio: bool = True
    label_from_subdir: bool = False

    def validate(self) -> None:
        p = Path(self.audio_dir)
        if not p.exists() or not p.is_dir():
            raise ValueError(f"audio_dir does not exist or is not a directory: {p}")
        if not (0.0 <= float(self.train_fraction) <= 1.0):
            raise ValueError("train_fraction must be within [0,1]")
        if self.sr <= 0:
            raise ValueError("sr must be > 0")
        if self.clip_seconds <= 0:
            raise ValueError("clip_seconds must be > 0")


def _list_files(cfg: AudioFolderAdapterConfig) -> List[Path]:
    root = Path(cfg.audio_dir)
    paths = list(root.rglob(cfg.pattern)) if cfg.recursive else list(root.glob(cfg.pattern))
    files = [p for p in paths if p.is_file()]
    files.sort(key=lambda p: str(p).lower())
    return files


def _file_sig(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {
        "name": p.name,
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


@ADAPTERS.register("audio_folder", description="Load a folder of WAV tracks as a world.")
class AudioFolderAdapter(WorldAdapter):
    NAME = "audio_folder"
    DESCRIPTION = "Audio folder adapter (WAV-first)."

    def __init__(self, cfg: AudioFolderAdapterConfig):
        if cfg.label_from_subdir and not cfg.recursive:
            cfg.recursive = True
        cfg.validate()
        self.cfg = cfg

    def fingerprint(self) -> str:
        files = _list_files(self.cfg)
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        d["files"] = [_file_sig(p) for p in files]
        return hash_dict(d)

    def _label_mapping(self, files: List[Path]) -> Dict[str, int]:
        names = sorted({p.parent.name for p in files})
        return {name: i for i, name in enumerate(names)}

    def _load_item(self, p: Path, label_map: Optional[Dict[str, int]]) -> Dict[str, Any]:
        cfg = self.cfg
        item: Dict[str, Any] = {
            "name": p.stem,
            "path": str(p.resolve()),
            "sr": int(cfg.sr),
            "meta": {},
        }

        if cfg.label_from_subdir and label_map is not None:
            item["label"] = int(label_map[p.parent.name])
            item["label_name"] = p.parent.name

        if cfg.include_audio:
            audio, orig_sr = load_audio(
                p,
                target_sr=int(cfg.sr),
                clip_seconds=float(cfg.clip_seconds),
                mono=bool(cfg.mono),
            )
            item["audio"] = audio.astype(np.float32)
            item["meta"] = {
                "orig_sr": int(orig_sr),
                "duration_seconds": float(audio.shape[0]) / float(cfg.sr),
            }
        return item

    def load(self) -> Dict[str, Any]:
        cfg = self.cfg
        files = _list_files(cfg)

        if cfg.max_files is not None and int(cfg.max_files) < len(files):
            rng = np.random.default_rng(int(cfg.seed))
            idx = np.arange(len(files))
            rng.shuffle(idx)
            idx = idx[: int(cfg.max_files)]
            files = [files[int(i)] for i in sorted(idx)]

        label_map = self._label_mapping(files) if cfg.label_from_subdir else None
        items = [self._load_item(p, label_map) for p in files]

        for i, it in enumerate(items):
            it["id"] = int(i)

        n = len(items)
        n_train = int(round(float(cfg.train_fraction) * float(n)))
        train = items[:n_train]
        test = items[n_train:]

        meta = {
            "cfg": asdict(cfg),
            "adapter": self.NAME,
            "n_total": int(n),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }
        if label_map is not None:
            meta["label_map"] = dict(label_map)

        return {"train": train, "test": test, "meta": meta}
