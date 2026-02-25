from .base import WorldAdapter
from .iphyre import IPhyreCollectConfig, collect_iphyre, IPhyreAdapter
from .hf_dataset import HFDatasetAdapter, HFDatasetAdapterConfig
from .hf_qa import HFQADatasetAdapter, HFQADatasetAdapterConfig
from .lerobot import LeRobotAdapter, LeRobotAdapterConfig
from .gymnasium import GymnasiumAdapter, GymnasiumAdapterConfig
from .gw_merger_lal import GWMergerLALAdapter, GWMergerLALAdapterConfig
from .finance_csv import FinanceCSVTapeAdapter, FinanceCSVTapeAdapterConfig
from .audio_folder import AudioFolderAdapter, AudioFolderAdapterConfig
from .esc50 import Esc50Adapter, Esc50AdapterConfig
from .urbansound8k import UrbanSound8KAdapter, UrbanSound8KAdapterConfig
from .kepler_root_npz import KeplerRootNPZAdapter, KeplerRootNPZAdapterConfig
from .linear_system_npz import LinearSystemNPZAdapter, LinearSystemNPZAdapterConfig
from .ode_npz import ODENPZAdapter, ODENPZAdapterConfig
from .integration_npz import IntegrationNPZAdapter, IntegrationNPZAdapterConfig
from .eigen_npz import EigenNPZAdapter, EigenNPZAdapterConfig

__all__ = [
    "WorldAdapter",
    "IPhyreCollectConfig",
    "collect_iphyre",
    "IPhyreAdapter",
    "HFDatasetAdapter",
    "HFDatasetAdapterConfig",
    "HFQADatasetAdapter",
    "HFQADatasetAdapterConfig",
    "LeRobotAdapter",
    "LeRobotAdapterConfig",
    "GymnasiumAdapter",
    "GymnasiumAdapterConfig",
    "GWMergerLALAdapter",
    "GWMergerLALAdapterConfig",
    "FinanceCSVTapeAdapter",
    "FinanceCSVTapeAdapterConfig",
    "AudioFolderAdapter",
    "AudioFolderAdapterConfig",
    "Esc50Adapter",
    "Esc50AdapterConfig",
    "UrbanSound8KAdapter",
    "UrbanSound8KAdapterConfig",
    "KeplerRootNPZAdapter",
    "KeplerRootNPZAdapterConfig",
    "LinearSystemNPZAdapter",
    "LinearSystemNPZAdapterConfig",
    "ODENPZAdapter",
    "ODENPZAdapterConfig",
    "IntegrationNPZAdapter",
    "IntegrationNPZAdapterConfig",
    "EigenNPZAdapter",
    "EigenNPZAdapterConfig",
]

# LJ adapter relies on h5py. Keep optional.
try:  # pragma: no cover
    from .lj_fluid import LJFluidH5Adapter, LJFluidH5AdapterConfig

    __all__ += ["LJFluidH5Adapter", "LJFluidH5AdapterConfig"]
except Exception:
    pass

# Jet-physics adapter relies on EnergyFlow. Keep optional.
try:  # pragma: no cover
    from .qg_jets import QGJetsAdapter, QGJetsAdapterConfig

    __all__ += ["QGJetsAdapter", "QGJetsAdapterConfig"]
except Exception:
    pass

# PDEBench adapter relies on h5py. Keep optional.
try:  # pragma: no cover
    from .pdebench_h5 import PDEBenchH5Adapter, PDEBenchH5AdapterConfig

    __all__ += ["PDEBenchH5Adapter", "PDEBenchH5AdapterConfig"]
except Exception:
    pass
