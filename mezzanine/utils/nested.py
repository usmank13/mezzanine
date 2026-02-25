"""Small helpers for nested dict-like structures.

HuggingFace Datasets and RL/robotics stacks often return nested dicts
(e.g. observation.images.cam_high). Recipes should be able to access these
without hard-coding the nesting.
"""

from __future__ import annotations

from typing import Any, Mapping


def get_by_dotted_key(obj: Any, dotted_key: str, default: Any = None) -> Any:
    """Safely read `obj[a][b][c]` given a dotted key "a.b.c".

    Works on nested mappings. If any key is missing, returns `default`.

    NOTE: Many HuggingFace datasets "flatten" nested keys into a single
    column name that itself contains dots (e.g. "observation.images.cam_high").
    For those, we first try a direct lookup by the full dotted key.
    """
    if isinstance(obj, Mapping) and dotted_key in obj:
        return obj[dotted_key]

    cur: Any = obj
    for part in str(dotted_key).split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur
