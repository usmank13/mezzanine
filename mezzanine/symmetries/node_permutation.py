from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


@dataclass
class NodePermutationConfig:
    """Node re-labeling / permutation similarity for graph-structured numerics.

    Applies a random permutation π to keys:
      - A: [n,n]  -> A[π][:,π]
      - b: [n]    -> b[π]
      - x: [n]    -> x[π]   (if present)

    It also stores the permutation in the returned example under `_perm` so
    recipes can canonicalize predictions (un-permute) before computing warrant
    gaps.
    """

    key_A: str = "A"
    key_b: str = "b"
    key_x: str = "x"  # optional


@SYMMETRIES.register("node_permutation")
class NodePermutationSymmetry(Symmetry):
    CONFIG_CLS = NodePermutationConfig
    NAME = "node_permutation"
    DESCRIPTION = "Random node permutation π: A->PAP^T, b->Pb, x->Px (stores _perm)."

    def __init__(self, config: NodePermutationConfig):
        self.config = config

    def sample(self, x: Dict[str, Any], seed: int) -> Dict[str, Any]:
        if self.config.key_A not in x:
            return x
        A = np.asarray(x[self.config.key_A])
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected A to be square [n,n], got {A.shape}")
        n = int(A.shape[0])
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n).astype(np.int64)

        out = dict(x)
        out["_perm"] = perm
        out[self.config.key_A] = A[np.ix_(perm, perm)]
        if self.config.key_b in out:
            b = np.asarray(out[self.config.key_b])
            out[self.config.key_b] = b[perm]
        if self.config.key_x in out:
            xx = np.asarray(out[self.config.key_x])
            out[self.config.key_x] = xx[perm]
        return out
