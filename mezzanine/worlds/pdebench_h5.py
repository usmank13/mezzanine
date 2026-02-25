from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.deterministic import deterministic_subsample_indices
from ..registry import ADAPTERS
from .base import WorldAdapter


@dataclass
class PDEBenchH5AdapterConfig:
    """Load PDEBench-style operator learning data from an HDF5 file.

    PDEBench files are HDF5 containers. There isn't a single universal schema
    across all PDE families, so this adapter is configurable:

    Common schema (many operator-learning datasets):
      - a dataset `u` with shape [N, T, ...] (trajectory)
      - task is to map u(t0) -> u(t1)

    Config options support both this and a direct (X,Y) pair format.

    Parameters
    ----------
    path:
      HDF5 file path.
    train_group, test_group:
      HDF5 group names (defaults: 'train', 'test').
    u_key:
      Dataset name/path containing solution trajectories (default: 'u').
    x_key, y_key:
      If provided, use these datasets directly as input/output pairs.
    t0, t1:
      Indices into the time axis of `u` if using trajectory format.
    n_train, n_test:
      Number of examples to sample.

    Notes
    -----
    Symmetry tests typically use periodic translations (np.roll) along spatial
    axes. Use `periodic_translation` symmetry with axes configured to match the
    loaded tensor layout.
    """

    path: str
    n_train: int = 50000
    n_test: int = 10000
    seed: int = 0

    train_group: str = "train"
    test_group: str = "test"

    # Either provide (x_key, y_key) or provide u_key + (t0,t1)
    u_key: str = "u"
    x_key: Optional[str] = None
    y_key: Optional[str] = None
    t0: int = 0
    t1: int = -1

    # Optional: if the stored tensors are huge, you can downsample by slicing.
    # Example: spatial_slice='::2,::2' for 2D fields.
    spatial_slice: Optional[str] = None

    def validate(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"PDEBenchH5Adapter: file not found: {p}")
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train and n_test must be non-negative")


def _apply_spatial_slice(arr: np.ndarray, spatial_slice: Optional[str]) -> np.ndarray:
    if spatial_slice is None or spatial_slice.strip() == "":
        return arr
    # We interpret spatial_slice as a comma-separated list of python slices.
    # Applied to the *last* dimensions of the array.
    parts = [p.strip() for p in spatial_slice.split(",")]
    slicers: List[slice] = []
    for p in parts:
        if p == ":":
            slicers.append(slice(None))
        elif p.startswith("::"):
            step = int(p[2:])
            slicers.append(slice(None, None, step))
        else:
            # allow 'a:b:c'
            bits = p.split(":")
            bits += [""] * (3 - len(bits))
            a = int(bits[0]) if bits[0] else None
            b = int(bits[1]) if bits[1] else None
            c = int(bits[2]) if bits[2] else None
            slicers.append(slice(a, b, c))
    # Apply to trailing dims.
    idx = (Ellipsis, *slicers)
    return arr[idx]


def _h5_get(obj: Any, key: str):
    # key may be a path like 'train/u' or '/train/u'
    k = key[1:] if key.startswith("/") else key
    if k in obj:
        return obj[k]
    # try direct
    if key in obj:
        return obj[key]
    raise KeyError(f"Key not found in HDF5: {key}")


@ADAPTERS.register("pdebench_h5")
class PDEBenchH5Adapter(WorldAdapter):
    NAME = "pdebench_h5"
    DESCRIPTION = "PDEBench HDF5 operator-learning dataset adapter."

    def __init__(self, config: PDEBenchH5AdapterConfig):
        config.validate()
        self.config = config
        self.path = Path(config.path)

        try:
            import h5py  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("pdebench_h5 adapter requires h5py") from e

        # We keep the file closed by default; load() reads eagerly into memory.
        self._h5py = h5py

    def fingerprint(self) -> str:
        st = self.path.stat()
        fp = {
            "type": self.NAME,
            "path": str(self.path),
            "mtime": int(st.st_mtime),
            "size": int(st.st_size),
            "config": asdict(self.config),
        }
        return json.dumps(fp, sort_keys=True)

    def _load_pair_arrays(self, f: Any, group: str) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        g = _h5_get(f, group)
        if cfg.x_key is not None and cfg.y_key is not None:
            X = np.asarray(_h5_get(g, cfg.x_key))
            Y = np.asarray(_h5_get(g, cfg.y_key))
            return X, Y

        # Trajectory format via u_key
        u = np.asarray(_h5_get(g, cfg.u_key))
        if u.ndim < 2:
            raise ValueError(f"Expected u to have time axis: got shape {u.shape}")
        t0 = int(cfg.t0)
        t1 = int(cfg.t1)
        X = u[:, t0]
        Y = u[:, t1]
        return X, Y

    def _build_split(
        self, split: str, n_target: int, seed: int
    ) -> List[Dict[str, Any]]:
        cfg = self.config
        group = cfg.train_group if split == "train" else cfg.test_group

        with self._h5py.File(self.path, "r") as f:
            X, Y = self._load_pair_arrays(f, group)

        X = _apply_spatial_slice(np.asarray(X), cfg.spatial_slice).astype(np.float32)
        Y = _apply_spatial_slice(np.asarray(Y), cfg.spatial_slice).astype(np.float32)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Mismatched N between X and Y: {X.shape} vs {Y.shape}")

        n_total = int(X.shape[0])
        n_take = min(int(n_target), n_total)
        idxs = (
            deterministic_subsample_indices(n_total, n_take, seed)
            if n_take > 0
            else np.array([], dtype=np.int64)
        )

        out: List[Dict[str, Any]] = []
        for i in idxs:
            ii = int(i)
            out.append({"x": X[ii], "y": Y[ii]})
        return out

    def load(self) -> Dict[str, Any]:
        train = self._build_split("train", self.config.n_train, self.config.seed)
        test = self._build_split("test", self.config.n_test, self.config.seed + 1)
        x_shape = (
            tuple(train[0]["x"].shape)
            if train
            else (tuple(test[0]["x"].shape) if test else ())
        )
        return {
            "train": train,
            "test": test,
            "meta": {
                "adapter": self.NAME,
                "cfg": asdict(self.config),
                "fingerprint": self.fingerprint(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "x_shape": x_shape,
            },
        }
