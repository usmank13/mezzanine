from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_gw_smbh_viz_exports(tmp_path: Path) -> None:
    pytest.importorskip("lalsimulation")
    pytest.importorskip("lal")
    pytest.importorskip("matplotlib")

    from mezzanine.recipes.gw_smbh_viz import GWSMBHVizRecipe

    recipe = GWSMBHVizRecipe(out_dir=tmp_path)
    argv = [
        "--out_dir",
        str(tmp_path),
        "--n_events",
        "1",
        "--event_index",
        "0",
        "--k_views",
        "2",
        "--plot_last_seconds",
        "10",
        "--export_psi4_vtk",
        "--psi4_vtk_grid",
        "8",
        "--psi4_vtk_frames",
        "2",
    ]
    recipe.run(argv)

    res = json.loads((tmp_path / "results.json").read_text())
    psi4_dir = res["artifacts"]["psi4_vtk_dir"]
    assert psi4_dir is not None
    out_dir = Path(psi4_dir)
    assert out_dir.exists()
    assert (out_dir / "psi4_0000.vtk").exists()
