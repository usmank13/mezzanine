from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


_SRC_ALIAS = "_mezzanine_src_time_origin_shift_bounds"


def _ensure_src_alias() -> str:
    """Import mezzanine modules without executing mezzanine/__init__.py."""
    if _SRC_ALIAS in sys.modules:
        return _SRC_ALIAS
    root = Path(__file__).resolve().parents[1]
    pkg_dir = root / "mezzanine"
    pkg = types.ModuleType(_SRC_ALIAS)
    pkg.__path__ = [str(pkg_dir)]
    sys.modules[_SRC_ALIAS] = pkg
    return _SRC_ALIAS


def _src_import(rel_module: str):
    alias = _ensure_src_alias()
    return importlib.import_module(f"{alias}.{rel_module}")


def test_time_origin_shift_respects_t_bounds() -> None:
    m = _src_import("symmetries.time_origin_shift")
    TimeOriginShiftConfig = m.TimeOriginShiftConfig
    TimeOriginShiftSymmetry = m.TimeOriginShiftSymmetry

    sym = TimeOriginShiftSymmetry(TimeOriginShiftConfig(max_shift=50.0, t_min=0.0, t_max=10.0))

    for t0 in [0.0, 1.0, 9.0, 10.0]:
        x = {"t": t0}
        for seed in range(50):
            y = sym.sample(x, seed=seed)
            assert 0.0 <= float(y["t"]) <= 10.0


def test_time_origin_shift_unbounded_when_no_bounds() -> None:
    m = _src_import("symmetries.time_origin_shift")
    TimeOriginShiftConfig = m.TimeOriginShiftConfig
    TimeOriginShiftSymmetry = m.TimeOriginShiftSymmetry

    sym = TimeOriginShiftSymmetry(TimeOriginShiftConfig(max_shift=1.0, t_min=None, t_max=None))
    x = {"t": 0.0}
    y = sym.sample(x, seed=0)
    assert -1.0 <= float(y["t"]) <= 1.0

