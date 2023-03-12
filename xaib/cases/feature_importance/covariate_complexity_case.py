from typing import Any
from ...base import Dataset, Model, Case
from ...metrics.feature_importance import CovariateRegularity


class CovariateComplexityCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__('covariate_complexity', ds, model, *args, **kwargs)
        self._metric_objs['covariate_regularity'] = CovariateRegularity(ds, model)
