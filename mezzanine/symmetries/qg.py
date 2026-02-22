from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .base import Symmetry


def _wrap_phi(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2 * np.pi) - np.pi


@dataclass
class QGPermutationConfig:
    """Permutation symmetry for jet constituents."""

    keep_padding: bool = True


class QGPermutationSymmetry(Symmetry):
    NAME = "qg_permutation"
    DESCRIPTION = "Randomly permute the order of jet constituents (S_N symmetry)."

    def __init__(self, cfg: QGPermutationConfig | None = None):
        self.cfg = cfg or QGPermutationConfig()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        P = np.asarray(x["particles"], dtype=np.float32)
        if P.ndim != 2 or P.shape[1] < 4:
            raise ValueError(f"Expected particles [P,4], got {P.shape}")

        pt = P[:, 0]
        mask = pt > 0
        active = P[mask]
        n = int(active.shape[0])
        if n <= 1:
            return dict(x)

        perm = rng.permutation(n)
        active = active[perm]

        outP = np.zeros_like(P)
        outP[:n] = active
        # padding remains zeros by construction
        out = dict(x)
        out["particles"] = outP
        return out


@dataclass
class QGSO2RotateConfig:
    """Internal SO(2) rotation in the (y,phi) plane around the jet axis.

    This models the fact that, for unpolarized jets, there is no preferred
    direction in the local (Δy, Δφ) plane.
    """

    theta_max: float = float(2 * np.pi)


class QGSO2RotateSymmetry(Symmetry):
    NAME = "qg_so2_rotate"
    DESCRIPTION = "Rotate the (y,phi) coordinates by a random angle θ (SO(2))."

    def __init__(self, cfg: QGSO2RotateConfig | None = None):
        self.cfg = cfg or QGSO2RotateConfig()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        P = np.asarray(x["particles"], dtype=np.float32)
        pt = P[:, 0]
        mask = pt > 0
        if not np.any(mask):
            return dict(x)

        theta_max = float(self.cfg.theta_max)
        if theta_max < 0:
            raise ValueError("theta_max must be >= 0")

        theta = rng.uniform(-theta_max, theta_max)
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        outP = P.copy()
        y = outP[mask, 1]
        phi = outP[mask, 2]

        y_new = c * y - s * phi
        phi_new = s * y + c * phi
        outP[mask, 1] = y_new
        outP[mask, 2] = _wrap_phi(phi_new)

        out = dict(x)
        out["particles"] = outP
        return out


@dataclass
class QGReflectionConfig:
    flip_phi: bool = True
    flip_y: bool = False


class QGReflectionSymmetry(Symmetry):
    NAME = "qg_reflection"
    DESCRIPTION = "Discrete reflection (O(2) parity) in the (y,phi) plane."

    def __init__(self, cfg: QGReflectionConfig | None = None):
        self.cfg = cfg or QGReflectionConfig()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        # Deterministic given seed: decide which reflection to apply.
        rng = np.random.default_rng(seed)
        do = bool(rng.integers(0, 2))
        if not do:
            return dict(x)

        P = np.asarray(x["particles"], dtype=np.float32)
        pt = P[:, 0]
        mask = pt > 0
        if not np.any(mask):
            return dict(x)

        outP = P.copy()
        if self.cfg.flip_y:
            outP[mask, 1] = -outP[mask, 1]
        if self.cfg.flip_phi:
            outP[mask, 2] = _wrap_phi(-outP[mask, 2])

        out = dict(x)
        out["particles"] = outP
        return out


@dataclass
class QGCoordNoiseConfig:
    sigma_y: float = 0.0
    sigma_phi: float = 0.0
    sigma_pt_frac: float = 0.0


class QGCoordNoiseSymmetry(Symmetry):
    NAME = "qg_coord_noise"
    DESCRIPTION = "Add small Gaussian noise to (pt,y,phi) (measurement symmetry)."

    def __init__(self, cfg: QGCoordNoiseConfig | None = None):
        self.cfg = cfg or QGCoordNoiseConfig()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        P = np.asarray(x["particles"], dtype=np.float32)
        pt = P[:, 0]
        mask = pt > 0
        if not np.any(mask):
            return dict(x)

        outP = P.copy()

        # pt noise is applied multiplicatively to preserve scale
        if float(self.cfg.sigma_pt_frac) > 0:
            eps = rng.normal(0.0, float(self.cfg.sigma_pt_frac), size=outP[mask, 0].shape).astype(np.float32)
            outP[mask, 0] = np.maximum(0.0, outP[mask, 0] * (1.0 + eps))

        if float(self.cfg.sigma_y) > 0:
            outP[mask, 1] = outP[mask, 1] + rng.normal(0.0, float(self.cfg.sigma_y), size=outP[mask, 1].shape).astype(np.float32)

        if float(self.cfg.sigma_phi) > 0:
            outP[mask, 2] = _wrap_phi(outP[mask, 2] + rng.normal(0.0, float(self.cfg.sigma_phi), size=outP[mask, 2].shape).astype(np.float32))

        out = dict(x)
        out["particles"] = outP
        return out


# Register
from ..registry import SYMMETRIES

SYMMETRIES.register("qg_permutation")(QGPermutationSymmetry)
SYMMETRIES.register("qg_so2_rotate")(QGSO2RotateSymmetry)
SYMMETRIES.register("qg_reflection")(QGReflectionSymmetry)
SYMMETRIES.register("qg_coord_noise")(QGCoordNoiseSymmetry)
