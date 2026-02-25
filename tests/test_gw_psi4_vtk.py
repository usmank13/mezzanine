from __future__ import annotations

from pathlib import Path

import numpy as np

from mezzanine.viz.gw_psi4_vtk import Psi4VTKConfig, write_psi4_vtk_series


def test_write_psi4_vtk_series(tmp_path: Path) -> None:
    t = np.linspace(0.0, 1.0, 200, dtype=np.float64)
    s = np.sin(2.0 * np.pi * 5.0 * t).astype(np.float64)

    cfg = Psi4VTKConfig(grid=8, frames=3, extent=1.0, prefix="psi4_test")
    meta = write_psi4_vtk_series(t=t, strain=s, out_dir=tmp_path, cfg=cfg)

    assert meta["frames"] == 3.0
    first = tmp_path / "psi4_test_0000.vtk"
    assert first.exists()
    text = first.read_text()
    assert "DATASET STRUCTURED_POINTS" in text
