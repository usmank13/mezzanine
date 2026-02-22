from __future__ import annotations


def test_registry_wiring_includes_neuralgcm_symmetries() -> None:
    from mezzanine.plugins import load_builtin_plugins
    from mezzanine.registry import SYMMETRIES

    load_builtin_plugins()
    assert "ens_member" in SYMMETRIES.list()
    assert "field_codec" in SYMMETRIES.list()


def test_recipe_registry_includes_neuralgcm() -> None:
    from mezzanine.recipes.registry import list_recipes

    assert "neuralgcm_ens_warrant_distill" in list_recipes()

