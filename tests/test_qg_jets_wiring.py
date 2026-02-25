from __future__ import annotations


def test_registry_wiring_includes_qg() -> None:
    from mezzanine.plugins import load_builtin_plugins
    from mezzanine.registry import ADAPTERS, SYMMETRIES, ENCODERS

    load_builtin_plugins()
    assert "qg_jets_energyflow" in ADAPTERS.list()
    assert "qg_permutation" in SYMMETRIES.list()
    assert "qg_so2_rotate" in SYMMETRIES.list()
    assert "qg_reflection" in SYMMETRIES.list()
    assert "qg_coord_noise" in SYMMETRIES.list()
    assert "qg_flatten" in ENCODERS.list()
    assert "qg_eec2" in ENCODERS.list()


def test_recipe_registry_includes_qg() -> None:
    from mezzanine.recipes.registry import list_recipes

    assert "qg_jets_distill" in list_recipes()


def test_cli_list_adapters_includes_qg(capsys) -> None:
    import mezzanine.cli as cli

    cli.main_args(["list-adapters"])
    out = capsys.readouterr().out
    assert "qg_jets_energyflow" in out
