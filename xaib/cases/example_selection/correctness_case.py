from typing import Any

from ...base import Case, Dataset, Explainer, Model
from ...metrics.example_selection import ModelRandomizationCheck


class CorrectnessCase(Case):
    def __init__(
        self,
        ds: Dataset,
        model: Model,
        explainer: Explainer,
        noisy_model: Model,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(ds, model, explainer, *args, **kwargs)
        self.name = "correctness"
        self.metrics.append(ModelRandomizationCheck(ds, model, explainer, noisy_model))
