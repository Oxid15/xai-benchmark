from typing import Any

from ...base import Case, Dataset, Explainer, Model
from ...metrics.example_selection import CovariateRegularity


class CovariateComplexityCase(Case):
    def __init__(
        self, ds: Dataset, model: Model, explainer: Explainer, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, explainer, *args, **kwargs)
        self.name = "covariate_complexity"
        self.metrics.append(CovariateRegularity(ds, model))
