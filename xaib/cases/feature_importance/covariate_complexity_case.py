from typing import Any

from ...base import Case, Dataset, Model
from ...metrics.feature_importance import CovariateRegularity


class CovariateComplexityCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "covariate_complexity"
        self._metric_objs["covariate_regularity"] = CovariateRegularity(ds, model)
