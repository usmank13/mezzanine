from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

from .base import Encoder
from ..registry import ENCODERS


def _wrap_phi(phi: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal periodic difference a-b in [-pi, pi)."""
    return _wrap_phi(a - b)


@dataclass(frozen=True)
class QGFlattenEncoderConfig:
    max_particles: int = 64
    include_pid: bool = True
    pid_scale: float = 0.01
    pt_normalize: bool = True  # use pt/sum(pt) per jet


class QGFlattenEncoder(Encoder):
    NAME = "qg_flatten"
    DESCRIPTION = "Flatten first K jet constituents into a fixed vector (order-sensitive)."

    def __init__(self, cfg: QGFlattenEncoderConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self.cfg), sort_keys=True).encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()[:16]
        return f"{self.NAME}:{h}"

    def encode(self, batch: Any) -> np.ndarray:
        # batch: list of examples, each with 'particles': [Pmax,4]
        xs: List[Dict[str, Any]] = list(batch)
        K = int(self.cfg.max_particles)
        if K <= 0:
            raise ValueError("max_particles must be positive")

        out_dim = K * (4 if self.cfg.include_pid else 3)
        Z = np.zeros((len(xs), out_dim), dtype=np.float32)

        for i, ex in enumerate(xs):
            P = np.asarray(ex["particles"], dtype=np.float32)
            if P.ndim != 2 or P.shape[1] < 4:
                raise ValueError(f"Expected particles [P,4], got {P.shape}")

            # active particles (padding has pt=0)
            pt = P[:, 0]
            mask = pt > 0
            P_act = P[mask]

            # keep first K in the *given* order (this is intentionally order-sensitive)
            Pk = np.zeros((K, 4), dtype=np.float32)
            n = min(K, int(P_act.shape[0]))
            if n > 0:
                Pk[:n] = P_act[:n]

            if self.cfg.pt_normalize:
                s = float(Pk[:, 0].sum())
                if s > 0:
                    Pk[:, 0] /= s

            if self.cfg.include_pid:
                Pk[:, 3] = Pk[:, 3] * float(self.cfg.pid_scale)
                feat = Pk[:, :4]
            else:
                feat = Pk[:, :3]

            Z[i] = feat.reshape(-1)

        return Z


@dataclass(frozen=True)
class QGEEC2EncoderConfig:
    """2-point energy-energy correlator histogram in (y,phi) space.

    This is analogous to a radial distribution function / pair correlation, but
    with energy (pT) weights.
    """

    n_bins: int = 64
    r_max: float = 1.0
    max_particles: int = 64
    pt_normalize: bool = True
    include_log_mult: bool = True


class QGEEC2Encoder(Encoder):
    NAME = "qg_eec2"
    DESCRIPTION = "Energy-weighted pairwise ΔR histogram (permutation + O(2) invariant)."

    def __init__(self, cfg: QGEEC2EncoderConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self.cfg), sort_keys=True).encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()[:16]
        return f"{self.NAME}:{h}"

    def encode(self, batch: Any) -> np.ndarray:
        xs: List[Dict[str, Any]] = list(batch)
        nb = int(self.cfg.n_bins)
        rmax = float(self.cfg.r_max)
        K = int(self.cfg.max_particles)
        if nb <= 0:
            raise ValueError("n_bins must be positive")
        if rmax <= 0:
            raise ValueError("r_max must be positive")
        if K <= 1:
            raise ValueError("max_particles must be >=2")

        extra = 1 if self.cfg.include_log_mult else 0
        Z = np.zeros((len(xs), nb + extra), dtype=np.float32)

        edges = np.linspace(0.0, rmax, nb + 1, dtype=np.float32)
        for i, ex in enumerate(xs):
            P = np.asarray(ex["particles"], dtype=np.float32)
            pt = P[:, 0]
            mask = pt > 0
            P_act = P[mask]

            # Keep top-K by pt for efficiency/robustness
            if P_act.shape[0] > K:
                order = np.argsort(P_act[:, 0])[::-1]
                P_act = P_act[order[:K]]

            n = int(P_act.shape[0])
            if n < 2:
                if self.cfg.include_log_mult:
                    Z[i, -1] = 0.0
                continue

            pt = P_act[:, 0].astype(np.float32)
            y = P_act[:, 1].astype(np.float32)
            phi = P_act[:, 2].astype(np.float32)

            if self.cfg.pt_normalize:
                s = float(pt.sum())
                if s > 0:
                    pt = pt / s

            # pairwise ΔR
            # ydiff: [n,n]
            ydiff = y[:, None] - y[None, :]
            pdiff = _delta_phi(phi[:, None], phi[None, :])
            dr = np.sqrt(ydiff * ydiff + pdiff * pdiff, dtype=np.float32)

            # weights w_ij = pt_i pt_j for i<j
            w = (pt[:, None] * pt[None, :]).astype(np.float32)

            iu = np.triu_indices(n, k=1)
            dr_u = dr[iu]
            w_u = w[iu]

            # histogram
            hist, _ = np.histogram(dr_u, bins=edges, weights=w_u)
            hist = hist.astype(np.float32)

            # Normalize to make scale-comparable
            s = float(hist.sum())
            if s > 0:
                hist /= s

            Z[i, :nb] = hist
            if self.cfg.include_log_mult:
                Z[i, -1] = np.log(float(n) + 1e-6).astype(np.float32)

        return Z


# Register
ENCODERS.register("qg_flatten")(QGFlattenEncoder)
ENCODERS.register("qg_eec2")(QGEEC2Encoder)
