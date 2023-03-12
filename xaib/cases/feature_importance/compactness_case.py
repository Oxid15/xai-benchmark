from typing import Any
from ...base import Dataset, Model, Case
from ...metrics.feature_importance import Sparsity


class CompactnessCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__('compactness', ds, model, *args, **kwargs)
        self._metric_objs['sparsity'] = Sparsity(ds, model)
