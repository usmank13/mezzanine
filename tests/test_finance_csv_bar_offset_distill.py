from __future__ import annotations

from pathlib import Path

import numpy as np


def _public_ohlcv_csv_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    p = root / "examples" / "data" / "spy_daily_ohlcv.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing expected repo dataset: {p}")
    return p


def _write_close_csv(path: Path, close: np.ndarray) -> None:
    path.write_text(
        "timestamp,close\n"
        + "\n".join([f"{i},{float(c)}" for i, c in enumerate(close.tolist())])
        + "\n"
    )


def _synthetic_close_series(*, T: int = 600, seed: int = 0) -> np.ndarray:
    """Generate a close series with an engineered lag dependency (offset fragility)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        s[t] = 0.98 * s[t - 1] + 0.5 * float(rng.standard_normal())

    r = np.zeros(T - 1, dtype=np.float32)
    r[0] = 0.01 * float(rng.standard_normal())
    for t in range(1, T - 1):
        r[t] = 0.002 * float(s[t - 1]) + 0.35 * float(r[t - 1]) + 0.01 * float(rng.standard_normal())

    close = np.exp(np.cumsum(np.concatenate([[0.0], r.astype(np.float64)]))).astype(np.float64)
    return close


def test_registry_wiring_includes_finance_csv() -> None:
    from mezzanine.plugins import load_builtin_plugins
    from mezzanine.registry import ADAPTERS, SYMMETRIES

    load_builtin_plugins()
    assert "finance_csv" in ADAPTERS.list()
    assert "market_bar_offset" in SYMMETRIES.list()


def test_finance_csv_adapter_builds_expected_shapes(tmp_path: Path) -> None:
    from mezzanine.worlds.finance_csv import FinanceCSVTapeAdapter, FinanceCSVTapeAdapterConfig

    close = _synthetic_close_series(T=300, seed=0)
    csv_path = tmp_path / "tape.csv"
    _write_close_csv(csv_path, close)

    cfg = FinanceCSVTapeAdapterConfig(
        path=str(csv_path),
        n_train=50,
        n_test=20,
        lookback=16,
        max_offset=2,
        trend_lookback=64,
        label_horizon=1,
        stride=1,
    )
    data = FinanceCSVTapeAdapter(cfg).load()
    assert len(data["train"]) == 50
    assert len(data["test"]) == 20

    ex = data["train"][0]
    assert ex["x"].shape == (1 + 16,)
    assert ex["r_context"].shape == (16 + 2 * 2,)
    assert isinstance(ex["label"], int)


def test_finance_csv_adapter_parses_public_ohlcv_csv() -> None:
    from mezzanine.worlds.finance_csv import FinanceCSVTapeAdapter, FinanceCSVTapeAdapterConfig

    cfg = FinanceCSVTapeAdapterConfig(
        path=str(_public_ohlcv_csv_path()),
        close_col="Close",
        timestamp_col="Date",
        n_train=50,
        n_test=20,
        lookback=16,
        max_offset=2,
        trend_lookback=64,
        label_horizon=1,
        stride=1,
    )
    data = FinanceCSVTapeAdapter(cfg).load()
    ex = data["train"][0]
    assert "timestamp" in ex
    assert ex["x"].shape == (1 + 16,)
    assert ex["r_context"].shape == (16 + 2 * 2,)


def test_finance_csv_bar_offset_distill_runs_end_to_end(tmp_path: Path) -> None:
    from mezzanine.recipes.finance_csv_bar_offset_distill import FinanceCSVBarOffsetDistillRecipe

    out_dir = tmp_path / "out"
    recipe = FinanceCSVBarOffsetDistillRecipe(out_dir=out_dir, config={})
    res = recipe.run(
        [
            "--path",
            str(_public_ohlcv_csv_path()),
            "--close_col",
            "Close",
            "--timestamp_col",
            "Date",
            "--n_train",
            "160",
            "--n_test",
            "80",
            "--lookback",
            "16",
            "--max_offset",
            "2",
            "--trend_lookback",
            "64",
            "--k_train",
            "3",
            "--k_test",
            "3",
            "--base_steps",
            "10",
            "--student_steps",
            "10",
            "--hidden",
            "64",
            "--depth",
            "2",
            "--batch_size",
            "64",
            "--seed",
            "0",
        ]
    )

    assert (out_dir / "results.json").exists()
    assert (out_dir / "diagnostics.png").exists()
    assert "baseline" in res and "student" in res
    assert res["baseline"]["gap_mean_tv_to_mean"] >= 0.0
    assert res["student"]["gap_mean_tv_to_mean"] >= 0.0
    assert np.isfinite(res["baseline"]["gap_mean_tv_to_mean"])
    assert np.isfinite(res["student"]["gap_mean_tv_to_mean"])
