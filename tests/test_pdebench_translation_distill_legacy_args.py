from __future__ import annotations

from pathlib import Path

from mezzanine.recipes.pdebench_translation_distill import PDEBenchTranslationDistillRecipe, _apply_legacy_u0_u1_keys


def test_pdebench_translation_distill_accepts_legacy_u0_u1_keys(tmp_path: Path) -> None:
    recipe = PDEBenchTranslationDistillRecipe(out_dir=tmp_path / "out", config={})
    p = recipe._build_arg_parser()
    args = p.parse_args(
        [
            "--dataset",
            "dummy.h5",
            "--train_u0_key",
            "train/u0",
            "--train_u1_key",
            "train/u1",
            "--test_u0_key",
            "test/u0",
            "--test_u1_key",
            "test/u1",
        ]
    )

    _apply_legacy_u0_u1_keys(args, p)
    assert args.train_group == "train"
    assert args.test_group == "test"
    assert args.x_key == "u0"
    assert args.y_key == "u1"
