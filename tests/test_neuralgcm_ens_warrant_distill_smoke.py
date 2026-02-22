from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_tiny_neuralgcm_zarr(tmp_path: Path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")

    R = 4
    T = 6
    L = 2
    Lev = 1
    Lon = 2
    Lat = 2

    deltas = np.array([np.timedelta64(6, "h"), np.timedelta64(24, "h")])
    rng = np.random.default_rng(0)

    members_temperature = rng.standard_normal((R, T, L, Lev, Lon, Lat)).astype(np.float32)
    members_geopotential = rng.standard_normal((R, T, L, Lev, Lon, Lat)).astype(np.float32)

    mean_temperature = members_temperature.mean(axis=0).astype(np.float32)
    mean_geopotential = members_geopotential.mean(axis=0).astype(np.float32)

    members = xr.Dataset(
        data_vars={
            "temperature": (("realization", "time", "prediction_timedelta", "level", "longitude", "latitude"), members_temperature),
            "geopotential": (("realization", "time", "prediction_timedelta", "level", "longitude", "latitude"), members_geopotential),
        },
        coords={
            "realization": np.arange(R, dtype=np.int64),
            "time": np.arange(T, dtype=np.int64),
            "prediction_timedelta": deltas,
            "level": np.array([500.0], dtype=np.float32),
            "longitude": np.arange(Lon, dtype=np.float32),
            "latitude": np.arange(Lat, dtype=np.float32),
        },
    )
    mean = xr.Dataset(
        data_vars={
            "temperature": (("time", "prediction_timedelta", "level", "longitude", "latitude"), mean_temperature),
            "geopotential": (("time", "prediction_timedelta", "level", "longitude", "latitude"), mean_geopotential),
        },
        coords={
            "time": np.arange(T, dtype=np.int64),
            "prediction_timedelta": deltas,
            "level": np.array([500.0], dtype=np.float32),
            "longitude": np.arange(Lon, dtype=np.float32),
            "latitude": np.arange(Lat, dtype=np.float32),
        },
    )

    members_path = tmp_path / "members.zarr"
    mean_path = tmp_path / "mean.zarr"
    members.to_zarr(members_path, mode="w")
    mean.to_zarr(mean_path, mode="w")
    return members_path, mean_path


def test_neuralgcm_ens_warrant_distill_smoke(tmp_path: Path) -> None:
    members_path, mean_path = _write_tiny_neuralgcm_zarr(tmp_path)

    from mezzanine.recipes.neuralgcm_ens_warrant_distill import NeuralGCMEnsWarrantDistillRecipe

    out_dir = tmp_path / "out"
    recipe = NeuralGCMEnsWarrantDistillRecipe(out_dir=out_dir, config={})
    res = recipe.run(
        [
            "--members_zarr",
            str(members_path),
            "--mean_zarr",
            str(mean_path),
            "--variables",
            "temperature,geopotential",
            "--lead_hours",
            "24",
            "--num_members",
            "4",
            "--k_train",
            "2",
            "--k_test",
            "2",
            "--n_train_times",
            "2",
            "--n_val_times",
            "1",
            "--n_test_times",
            "2",
            "--max_points_per_time",
            "0",
            "--steps",
            "2",
            "--batch",
            "4",
            "--hidden",
            "8",
            "--depth",
            "2",
            "--eval_every",
            "1",
            "--seed",
            "0",
        ]
    )

    assert (out_dir / "results.json").exists()
    assert (out_dir / "diagnostics.png").exists()

    loaded = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert loaded["exp"] == "neuralgcm_ens_warrant_distill"
    assert "make_break" in loaded
    assert res["exp"] == "neuralgcm_ens_warrant_distill"

