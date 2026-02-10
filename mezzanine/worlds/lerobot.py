from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from ..core.cache import hash_dict
from ..core.deterministic import deterministic_subsample_indices
from .base import WorldAdapter


@dataclass
class LeRobotAdapterConfig:
    repo_id: str = "lerobot/pusht_image"
    train_split: str = "train"
    test_split: str = "test"
    n_train: int = 4000
    n_test: int = 2000
    seed: int = 0
    data_dir: str = "data"

    # These are *metadata* here; decoding is recipe-specific
    camera_key: str = "observation.image"
    action_key: str = "action"
    timestamp_key: str = "timestamp"


class LeRobotAdapter(WorldAdapter):
    """Adapter for LeRobot datasets hosted on HF.

    v1.0 note:
      - This adapter focuses on loading + stable subsampling + metadata.
      - Video decoding / frame selection is intentionally handled by recipes, because
        it depends heavily on the experiment (camera, delta, action windowing).
    """
    NAME = "lerobot"
    DESCRIPTION = "LeRobot adapter (HF-hosted robotics trajectories)."

    def __init__(self, cfg: LeRobotAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def load(self) -> Dict[str, Any]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError("LeRobot adapter requires `datasets`. Install: pip install mezzanine[robotics]") from e
        ds = load_dataset(self.cfg.repo_id, data_dir=self.cfg.data_dir)

        if self.cfg.train_split not in ds:
            # Most LeRobot datasets expose a `train` split, but keep the error message explicit.
            raise KeyError(f"LeRobot dataset {self.cfg.repo_id} has splits={list(ds.keys())}, missing train_split={self.cfg.train_split!r}")

        train = ds[self.cfg.train_split]

        # Some datasets are single-split (train only). In that case, create a deterministic
        # disjoint train/test partition from the same split.
        if self.cfg.test_split in ds:
            test = ds[self.cfg.test_split]
            tr_idx = deterministic_subsample_indices(len(train), min(self.cfg.n_train, len(train)), self.cfg.seed)
            te_idx = deterministic_subsample_indices(len(test), min(self.cfg.n_test, len(test)), self.cfg.seed + 1)
            split_fallback = False
        else:
            test = train
            n_total = min(int(self.cfg.n_train) + int(self.cfg.n_test), len(train))
            all_idx = deterministic_subsample_indices(len(train), n_total, self.cfg.seed)
            ntr = min(int(self.cfg.n_train), len(all_idx))
            tr_idx = all_idx[:ntr]
            te_idx = all_idx[ntr:ntr + min(int(self.cfg.n_test), max(0, len(all_idx) - ntr))]
            split_fallback = True

        # Return an indexable view rather than decoding videos here.
        meta = {
            "repo_id": self.cfg.repo_id,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "seed": self.cfg.seed,
            "split_fallback": split_fallback,
            "camera_key": self.cfg.camera_key,
            "action_key": self.cfg.action_key,
            "timestamp_key": self.cfg.timestamp_key,
            "data_dir": self.cfg.data_dir,
            "train_fingerprint": getattr(train, "_fingerprint", None),
            "test_fingerprint": getattr(test, "_fingerprint", None),
        }
        return {"train_ds": train, "test_ds": test, "train_idx": tr_idx, "test_idx": te_idx, "meta": meta}


# Register
from ..registry import ADAPTERS
ADAPTERS.register("lerobot")(LeRobotAdapter)
