from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .recipe_base import Recipe

# Recipes may depend on optional backends (vision/physics/robotics libs). We keep
# the registry import-tolerant so `import mezzanine` works in minimal setups.
_REGISTRY: Dict[str, Type[Recipe]] = {}

def _try_register(cls: Type[Recipe]) -> None:
    _REGISTRY[cls.NAME] = cls

try:  # pragma: no cover
    from .iphyre_latent_dynamics import IPhyreLatentDynamicsRecipe
    _try_register(IPhyreLatentDynamicsRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .lerobot_latent_dynamics import LeRobotLatentDynamicsRecipe
    _try_register(LeRobotLatentDynamicsRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .hf_text_order_distill import HFTextOrderDistillRecipe
    _try_register(HFTextOrderDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .hf_llm_hiddenstate_order_distill import HFLLMHiddenStateOrderDistill
    _try_register(HFLLMHiddenStateOrderDistill)
except Exception:
    pass

try:  # pragma: no cover
    from .gw_smbh_viz import GWSMBHVizRecipe
    _try_register(GWSMBHVizRecipe)
except Exception:
    pass


try:  # pragma: no cover
    from .lj_fluid_distill import LJFluidDistillRecipe
    _try_register(LJFluidDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .qg_jets_distill import QGJetsDistillRecipe
    _try_register(QGJetsDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .finance_csv_bar_offset_distill import FinanceCSVBarOffsetDistillRecipe
    _try_register(FinanceCSVBarOffsetDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .audio_warrant_mix import AudioWarrantMixRecipe
    _try_register(AudioWarrantMixRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .crystal_spacegroup_distill import CrystalSpaceGroupDistillRecipe
    _try_register(CrystalSpaceGroupDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .neuralgcm_ens_warrant_distill import NeuralGCMEnsWarrantDistillRecipe
    _try_register(NeuralGCMEnsWarrantDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .kepler_root_distill import KeplerRootDistillRecipe
    _try_register(KeplerRootDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .linear_system_permutation_distill import LinearSystemPermutationDistillRecipe
    _try_register(LinearSystemPermutationDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .ode_time_origin_distill import ODETimeOriginDistillRecipe
    _try_register(ODETimeOriginDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .integration_circular_shift_distill import IntegrationCircularShiftDistillRecipe
    _try_register(IntegrationCircularShiftDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .pdebench_translation_distill import PDEBenchTranslationDistillRecipe
    _try_register(PDEBenchTranslationDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .eigen_permutation_distill import EigenPermutationDistillRecipe
    _try_register(EigenPermutationDistillRecipe)
except Exception:
    pass



def list_recipes() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, cls in _REGISTRY.items():
        out[k] = {"description": getattr(cls, "DESCRIPTION", ""), "class": cls.__name__}
    return out


def get_recipe(name: str) -> Type[Recipe]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown recipe: {name}. Available: {', '.join(sorted(_REGISTRY.keys()))}")
    return _REGISTRY[name]


def run_recipe(
    name: str,
    *,
    out_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    argv: Optional[list[str]] = None,
) -> Dict[str, Any]:
    cls = get_recipe(name)
    recipe = cls(out_dir=out_dir, config=overrides or {})
    return recipe.run(argv=argv or [])
