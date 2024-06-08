from typing import Any

from ...base import Case, Dataset, Explainer, Model
from ...metrics.feature_importance import Sparsity


class CompactnessCase(Case):
    def __init__(self, ds: Dataset, model: Model, expl: Explainer, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "compactness"
        self.metrics.append(Sparsity(ds, model, expl))
