from typing import Any

from ...base import Case, Dataset, Explainer, Model
from ...metrics.example_selection import TargetDiscriminativeness


class ContrastivityCase(Case):
    def __init__(
        self, ds: Dataset, model: Model, explainer: Explainer, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "contrastivity"
        self.metrics.append(TargetDiscriminativeness(ds, model, explainer))
