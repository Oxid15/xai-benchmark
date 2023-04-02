from typing import Any

from ...base import Case, Dataset, Model
from ...metrics.feature_importance import Sparsity


class CompactnessCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "compactness"
        self._metric_objs["sparsity"] = Sparsity(ds, model)
