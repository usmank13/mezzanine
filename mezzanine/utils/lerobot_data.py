from __future__ import annotations

"""Utilities for working with LeRobot-format datasets.

LeRobot datasets on HuggingFace can appear in (at least) two common shapes:

1. *Step format*: each dataset row corresponds to a single timestep.
2. *Episode format*: each dataset row corresponds to an episode and contains
   sequences (lists/arrays) of observations/actions.

These helpers try to be robust across both layouts while staying lightweight.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from collections.abc import Mapping

import io

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from .nested import get_by_dotted_key


def _walk_leaf_paths(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield (dotted_path, leaf_value) pairs for nested mappings.

    This is used only for lightweight key inference / debugging on a single example.
    """
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            kk = str(k)
            # If the dataset already uses flattened dotted keys, treat those as leaves.
            if "." in kk and not isinstance(v, Mapping):
                yield prefix + kk, v
            elif isinstance(v, Mapping):
                yield from _walk_leaf_paths(v, prefix + kk + ".")
            else:
                yield prefix + kk, v
    else:
        if prefix:
            yield prefix.rstrip("."), obj


def infer_image_key(example: Any) -> Optional[str]:
    """Best-effort inference of an image column/key.

    Returns a dotted key usable with `get_by_dotted_key`.
    """
    candidates: List[str] = []
    for path, v in _walk_leaf_paths(example):
        p = path.lower()
        if ("image" in p or "images" in p or "cam" in p) and v is not None:
            try:
                _ = _to_uint8_rgb_array(v)
                candidates.append(path)
            except Exception:
                continue
    if not candidates:
        return None

    def score(k: str) -> Tuple[int, int, str]:
        kl = k.lower()
        # Prefer "cam_high" and exterior views for montages.
        pri = 5
        if "cam_high" in kl:
            pri = 0
        elif "exterior" in kl:
            pri = 1
        elif "wrist" in kl:
            pri = 2
        elif kl.endswith("image"):
            pri = 3
        return (pri, len(k), k)

    candidates.sort(key=score)
    return candidates[0]


def infer_action_key(example: Any) -> Optional[str]:
    """Best-effort inference of an action vector key."""
    candidates: List[str] = []
    for path, v in _walk_leaf_paths(example):
        p = path.lower()
        if "action" in p and v is not None:
            try:
                a = _to_float_action(v)
                if a.size > 0:
                    candidates.append(path)
            except Exception:
                continue
    if not candidates:
        return None
    # Prefer exactly "action" if it exists.
    candidates.sort(key=lambda k: (0 if k == "action" else 1, len(k), k))
    return candidates[0]


def resolve_camera_action_keys(
    example: Any,
    *,
    camera_key: str,
    action_key: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve keys against an example, falling back to inferred keys if needed."""
    meta: Dict[str, Any] = {
        "requested": {"camera_key": camera_key, "action_key": action_key},
        "resolved": {},
    }

    cam = get_by_dotted_key(example, camera_key)
    act = get_by_dotted_key(example, action_key)

    cam_key_used = camera_key
    act_key_used = action_key

    if cam is None:
        inf = infer_image_key(example)
        if inf is not None:
            cam_key_used = inf
        meta["resolved"]["camera_key"] = cam_key_used

    if act is None:
        inf = infer_action_key(example)
        if inf is not None:
            act_key_used = inf
        meta["resolved"]["action_key"] = act_key_used

    return cam_key_used, act_key_used, meta


def _to_uint8_rgb_array(frame: Any) -> np.ndarray:
    """Convert a dataset frame to a uint8 HxWx3 array."""
    if frame is None:
        raise ValueError("Frame is None")

    # HuggingFace `Image` feature typically returns PIL.Image.Image.
    if Image is not None and isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"), dtype=np.uint8)

    if isinstance(frame, np.ndarray):
        arr = frame
        # Accept HxWxC; if CxHxW, transpose.
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        # Grayscale -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        # Normalize dtype/range
        if arr.dtype != np.uint8:
            # Common case: float images in [0, 1]
            if np.issubdtype(arr.dtype, np.floating):
                mx = float(np.nanmax(arr)) if arr.size else 0.0
                if mx <= 1.0 + 1e-6:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    # Some datasets may return dict-like payloads for images.
    if isinstance(frame, dict):
        # Common keys: "array" or "image".
        for k in ("array", "image", "pixels"):
            if k in frame:
                return _to_uint8_rgb_array(frame[k])

        # HuggingFace Image feature can appear as {"bytes": ..., "path": ...}
        # when not automatically decoded.
        if Image is not None and "bytes" in frame and frame["bytes"] is not None:
            try:
                b = frame["bytes"]
                if isinstance(b, (bytes, bytearray)):
                    with Image.open(io.BytesIO(b)) as im:  # type: ignore[attr-defined]
                        return np.array(im.convert("RGB"), dtype=np.uint8)
            except Exception:
                pass
        if Image is not None and "path" in frame and frame["path"] is not None:
            try:
                with Image.open(frame["path"]) as im:  # type: ignore[attr-defined]
                    return np.array(im.convert("RGB"), dtype=np.uint8)
            except Exception:
                pass

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _to_float_action(action: Any) -> np.ndarray:
    if action is None:
        raise ValueError("Action is None")
    a = np.asarray(action, dtype=np.float32)
    return a.reshape(-1)


def infer_episode_key(column_names: Sequence[str]) -> Optional[str]:
    """Best-effort inference of an episode id column in step-format datasets."""
    candidates = [
        "episode_index",
        "episode_id",
        "episode",
        "traj_id",
        "trajectory_id",
        "sequence_id",
        "seq_id",
    ]
    for k in candidates:
        if k in column_names:
            return k
    return None


def _is_sequence_tensor(x: Any) -> bool:
    # Episode format often stores a list/tuple of frames or an array with a time dimension.
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        first = x[0]
        # A list of scalars is likely a single action vector, not a time sequence.
        if isinstance(first, (list, tuple, np.ndarray)):
            return True
        if Image is not None and isinstance(first, Image.Image):
            return True
        if isinstance(first, dict):
            return True
        return False
    if isinstance(x, np.ndarray) and x.ndim >= 4:
        return True
    return False


@dataclass
class PairSample:
    img_t: np.ndarray
    img_tp: np.ndarray
    action_feat: np.ndarray


def build_pairs(
    ds: Any,
    indices: Sequence[int],
    *,
    camera_key: str,
    action_key: str,
    delta_steps: int = 1,
    per_episode_samples: int = 1,
    seed: int = 0,
    episode_key: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Create (img_t, img_{t+Δ}, action_t) training pairs.

    Returns a list of dicts with keys: img_t, img_tp, action_feat.
    """
    if delta_steps < 1:
        raise ValueError("delta_steps must be >= 1")
    rng = np.random.default_rng(seed)

    if len(indices) == 0:
        return []

    # Peek at one example to determine layout.
    ex0 = ds[int(indices[0])]
    cam0 = get_by_dotted_key(ex0, camera_key)
    act0 = get_by_dotted_key(ex0, action_key)
    episode_format = _is_sequence_tensor(cam0) or _is_sequence_tensor(act0)

    out: List[Dict[str, Any]] = []

    if episode_format:
        # Each row is an episode with sequences.
        for ep_i in indices:
            ex = ds[int(ep_i)]
            frames = get_by_dotted_key(ex, camera_key)
            actions = get_by_dotted_key(ex, action_key)
            if frames is None or actions is None:
                continue

            # Normalize to python sequences.
            if isinstance(frames, np.ndarray):
                # frames: [T,H,W,C]
                T = int(frames.shape[0])
                frame_get = lambda t: frames[t]
            else:
                T = len(frames)
                frame_get = lambda t: frames[t]

            if isinstance(actions, np.ndarray):
                Ta = int(actions.shape[0])
                action_get = lambda t: actions[t]
            else:
                Ta = len(actions)
                action_get = lambda t: actions[t]

            T_eff = min(T, Ta)
            if T_eff <= delta_steps:
                continue

            # Sample one or more (t, t+Δ) pairs from this episode.
            n_samp = int(per_episode_samples)
            for _ in range(n_samp):
                t = int(rng.integers(0, T_eff - delta_steps))
                try:
                    img_t = _to_uint8_rgb_array(frame_get(t))
                    img_tp = _to_uint8_rgb_array(frame_get(t + delta_steps))
                    a = _to_float_action(action_get(t))
                except Exception:
                    continue
                out.append({"img_t": img_t, "img_tp": img_tp, "action_feat": a})
                if max_pairs is not None and len(out) >= max_pairs:
                    return out

        return out

    # Step format: each row is a timestep. We pair idx -> idx+Δ.
    if episode_key is None:
        try:
            episode_key = infer_episode_key(getattr(ds, "column_names", []))
        except Exception:
            episode_key = None

    n_ds = len(ds)
    for i in indices:
        ii = int(i)
        jj = ii + int(delta_steps)
        if jj >= n_ds:
            continue
        ex_i = ds[ii]
        ex_j = ds[jj]
        if episode_key is not None:
            try:
                # episode_key can be flat or nested depending on dataset formatting.
                if get_by_dotted_key(ex_i, episode_key) != get_by_dotted_key(ex_j, episode_key):
                    continue
            except Exception:
                # If episode_key access fails, fall back to trusting the dataset.
                pass
        try:
            img_t = _to_uint8_rgb_array(get_by_dotted_key(ex_i, camera_key))
            img_tp = _to_uint8_rgb_array(get_by_dotted_key(ex_j, camera_key))
            a = _to_float_action(get_by_dotted_key(ex_i, action_key))
        except Exception:
            continue
        out.append({"img_t": img_t, "img_tp": img_tp, "action_feat": a})
        if max_pairs is not None and len(out) >= max_pairs:
            break

    return out
