from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.deterministic import deterministic_subsample_indices
from ..registry import ADAPTERS
from .base import WorldAdapter


@dataclass
class MatbenchTaskAdapterConfig:
    """Adapter around the `matbench` Python package.

    Matbench provides curated train/test splits and is widely used in the
    materials ML community.

    Requirements (extras): `matbench` (and its dependencies, incl. matminer).

    Config notes:
      * Matbench tasks are defined with nested CV folds. We select one fold
        (default: 0) and treat its train+val split as "train" and its test
        split as "test".
      * Many tasks have structure inputs (pymatgen Structure); some are
        composition-only.
    """

    dataset_name: str  # e.g. "matbench_mp_e_form", "matbench_phonons"
    fold: int = 0

    n_train: int = 100000
    n_test: int = 20000
    seed: int = 0

    def validate(self) -> None:
        if int(self.fold) < 0:
            raise ValueError("fold must be >= 0")
        if int(self.n_train) < 0 or int(self.n_test) < 0:
            raise ValueError("n_train and n_test must be non-negative")


def _require_matbench():
    try:
        from matbench.task import MatbenchTask  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MatbenchTaskAdapter requires optional dependency `matbench`. "
            "Install via `pip install matbench` (and typically `matminer`)."
        ) from e
    return MatbenchTask


@ADAPTERS.register("matbench_task")
class MatbenchTaskAdapter(WorldAdapter):
    """Matbench world adapter."""

    def __init__(self, config: MatbenchTaskAdapterConfig):
        config.validate()
        self.config = config

        MatbenchTask = _require_matbench()
        self.task = MatbenchTask(str(config.dataset_name), autoload=True)
        if int(config.fold) not in self.task.folds:
            raise ValueError(
                f"Fold {config.fold} not available for {config.dataset_name}. "
                f"Available folds: {list(self.task.folds)}"
            )

    def fingerprint(self) -> str:
        fp = {
            "type": "matbench_task",
            "dataset_name": self.config.dataset_name,
            "fold": int(self.config.fold),
            "config": asdict(self.config),
        }
        return json.dumps(fp, sort_keys=True)

    def _subsample_pairs(
        self,
        X: List[Any] | np.ndarray,
        y: List[Any] | np.ndarray,
        n_target: int,
        seed: int,
    ) -> Tuple[List[Any], List[Any]]:
        def _take(seq: Any, i: int) -> Any:
            # Matbench sometimes returns pandas Series; `ser[i]` is being deprecated for
            # positional access. Prefer .iloc when available.
            iloc = getattr(seq, "iloc", None)
            if iloc is not None:
                return iloc[int(i)]
            return seq[int(i)]

        n = int(len(X))
        if n == 0:
            if int(n_target) == 0:
                return [], []
            raise ValueError("Matbench task split has no examples to sample from.")

        n_take = min(int(n_target), n)
        idxs = deterministic_subsample_indices(n, n_take, int(seed))
        X_s = [_take(X, int(i)) for i in idxs]
        y_s = [_take(y, int(i)) for i in idxs]
        return X_s, y_s

    def get_train_examples(self) -> List[Dict[str, Any]]:
        X, y = self.task.get_train_and_val_data(int(self.config.fold), as_type="tuple")
        X_s, y_s = self._subsample_pairs(
            X, y, int(self.config.n_train), int(self.config.seed)
        )
        return [{"X": X_s[i], "y": y_s[i]} for i in range(len(X_s))]

    def get_test_examples(self) -> List[Dict[str, Any]]:
        X, y = self.task.get_test_data(
            int(self.config.fold), as_type="tuple", include_target=True
        )
        X_s, y_s = self._subsample_pairs(
            X, y, int(self.config.n_test), int(self.config.seed) + 1
        )
        return [{"X": X_s[i], "y": y_s[i]} for i in range(len(X_s))]

    def load(self) -> Dict[str, Any]:
        train = self.get_train_examples()
        test = self.get_test_examples()
        meta: Dict[str, Any] = {
            "adapter": "matbench_task",
            "cfg": asdict(self.config),
            "fingerprint": self.fingerprint(),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }
        return {"train": train, "test": test, "meta": meta}
