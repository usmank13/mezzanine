from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


def _require_pymatgen():
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SpaceGroupSymmetry requires optional dependency `pymatgen`."
        ) from e
    return SpacegroupAnalyzer


@dataclass
class SpaceGroupConfig:
    """Apply a random crystallographic space-group operation.

    Expects examples where x["X"] is a pymatgen Structure.
    """

    symprec: float = 1e-2
    return_operation: bool = False


@SYMMETRIES.register("space_group")
class SpaceGroupSymmetry(Symmetry):
    CONFIG_CLS = SpaceGroupConfig

    def __init__(self, config: SpaceGroupConfig):
        self.config = config

    def sample(self, x: Dict[str, Any], *, seed: int) -> Dict[str, Any]:
        if "X" not in x:
            return x

        struct = x["X"]
        # Avoid importing pymatgen unless needed.
        SpacegroupAnalyzer = _require_pymatgen()

        # Heuristic: Structure objects have a copy() method and lattice attribute.
        if not hasattr(struct, "copy"):
            return x

        sga = SpacegroupAnalyzer(struct, symprec=float(self.config.symprec))
        try:
            ops = sga.get_symmetry_operations(cartesian=False)
        except TypeError:
            # Older pymatgen versions
            ops = sga.get_symmetry_operations()
        if not ops:
            return x

        rng = np.random.default_rng(int(seed))
        op = ops[int(rng.integers(0, len(ops)))]

        struct2 = struct.copy()
        # Pymatgen's apply_operation signature differs slightly across versions.
        try:
            struct2.apply_operation(op, fractional=True)
        except TypeError:
            struct2.apply_operation(op)

        out = dict(x)
        out["X"] = struct2
        if bool(self.config.return_operation):
            out["symm_op"] = op
        return out

