from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Psi4VTKConfig:
    grid: int = 96
    frames: int = 60
    extent: float = 0.0  # <=0 means auto from time window
    c: float = 1.0
    prefix: str = "psi4"
    normalize: bool = True


def _psi4_from_strain(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if t.ndim != 1 or s.ndim != 1 or t.shape[0] != s.shape[0]:
        raise ValueError("t and s must be 1D arrays of equal length")
    if t.shape[0] < 4:
        raise ValueError("Need at least 4 time samples to compute psi4")
    # psi4 proxy: second time derivative of strain
    d1 = np.gradient(s, t)
    d2 = np.gradient(d1, t)
    return d2.astype(np.float32)


def _auto_extent(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2:
        return 1.0
    return float(max(1e-6, 0.25 * (t[-1] - t[0])))


def _write_structured_points(path: Path, field: np.ndarray, *, origin: float, spacing: float) -> None:
    nz, ny, nx = field.shape
    flat = field.ravel(order="C")
    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("psi4 synthetic field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin:.6e} {origin:.6e} {origin:.6e}\n")
        f.write(f"SPACING {spacing:.6e} {spacing:.6e} {spacing:.6e}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write("SCALARS psi4 float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(0, flat.size, 6):
            chunk = flat[i:i + 6]
            f.write(" ".join(f"{v:.6e}" for v in chunk) + "\n")


def write_psi4_vtk_series(
    *,
    t: np.ndarray,
    strain: np.ndarray,
    out_dir: Path,
    cfg: Psi4VTKConfig,
) -> Dict[str, float]:
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(strain, dtype=np.float64)
    if t.ndim != 1 or s.ndim != 1 or t.shape[0] != s.shape[0]:
        raise ValueError("t and strain must be 1D arrays of equal length")
    if cfg.grid < 2:
        raise ValueError("grid must be >= 2")
    if cfg.frames < 1:
        raise ValueError("frames must be >= 1")

    psi4 = _psi4_from_strain(t, s)
    if cfg.normalize:
        scale = float(np.max(np.abs(psi4))) if psi4.size else 0.0
        if scale > 0:
            psi4 = psi4 / scale

    extent = float(cfg.extent) if cfg.extent > 0 else _auto_extent(t)
    n = int(cfg.grid)
    coords = np.linspace(-extent, extent, n, dtype=np.float32)
    z = coords[:, None, None]
    y = coords[None, :, None]
    x = coords[None, None, :]
    r = np.sqrt(x * x + y * y + z * z, dtype=np.float32)

    spacing = float(2.0 * extent / max(1, n - 1))
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_times = np.linspace(t[0], t[-1], int(cfg.frames))
    for i, tf in enumerate(frame_times):
        t_ret = tf - (r / float(cfg.c))
        field = np.interp(t_ret, t, psi4, left=0.0, right=0.0).astype(np.float32)
        path = out_dir / f"{cfg.prefix}_{i:04d}.vtk"
        _write_structured_points(path, field, origin=-extent, spacing=spacing)

    return {
        "grid": float(cfg.grid),
        "frames": float(cfg.frames),
        "extent": float(extent),
        "spacing": float(spacing),
    }
