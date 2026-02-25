from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def _run_script(script: Path, args: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_linear_system_generate_dataset_accepts_legacy_flags(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "linear_system_generate_dataset.py"
    out = tmp_path / "lin_sys.npz"

    _run_script(
        script,
        [
            "--out",
            str(out),
            "--n_train",
            "4",
            "--n_test",
            "2",
            "--n",
            "8",
            "--seed",
            "0",
            "--density",
            "0.05",
            "--spd",
        ],
    )

    data = np.load(out)
    assert data["train_A"].shape == (4, 8, 8)
    assert data["test_A"].shape == (2, 8, 8)

    A0 = np.asarray(data["train_A"][0])
    assert np.allclose(A0, A0.T, atol=1e-5)


def test_ode_generate_dataset_accepts_legacy_flags(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "ode_generate_dataset.py"
    out = tmp_path / "ode.npz"

    _run_script(
        script,
        [
            "--out",
            str(out),
            "--system",
            "lorenz",
            "--n_traj",
            "5",
            "--t_max",
            "0.2",
            "--dt",
            "0.05",
            "--seed",
            "0",
        ],
    )

    data = np.load(out)
    assert data["train_x"].shape == (4, 5, 3)
    assert data["test_x"].shape == (1, 5, 3)

    t = np.asarray(data["train_t"])
    assert t.shape == (5,)
    assert np.isclose(float(t[-1]), 0.2, atol=1e-6)


def test_integration_generate_dataset_accepts_n_grid(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "integration_generate_dataset.py"
    out = tmp_path / "integration.npz"

    _run_script(
        script,
        [
            "--out",
            str(out),
            "--n_train",
            "4",
            "--n_test",
            "2",
            "--n_grid",
            "32",
            "--K",
            "3",
            "--seed",
            "0",
        ],
    )

    data = np.load(out)
    assert data["train_f"].shape == (4, 32)
    assert data["test_f"].shape == (2, 32)
    assert data["train_y"].shape == (4, 1)
    assert data["test_y"].shape == (2, 1)


def test_eigen_generate_dataset_accepts_density(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "eigen_generate_dataset.py"
    out = tmp_path / "eigen.npz"

    _run_script(
        script,
        [
            "--out",
            str(out),
            "--n_train",
            "3",
            "--n_test",
            "1",
            "--n",
            "12",
            "--k",
            "4",
            "--density",
            "0.2",
            "--seed",
            "0",
        ],
    )

    data = np.load(out)
    assert data["train_A"].shape == (3, 12, 12)
    assert data["test_A"].shape == (1, 12, 12)
    assert data["train_eval"].shape == (3, 4)
    assert data["test_eval"].shape == (1, 4)
