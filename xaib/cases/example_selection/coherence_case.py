from typing import Any

from ...base import Case, Dataset, Explainer, Model
from ...metrics.example_selection import SameClassCheck


class CoherenceCase(Case):
    def __init__(
        self, ds: Dataset, model: Model, explainer: Explainer, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, ds=ds, model=model, explainer=explainer, **kwargs)
        self.name = "coherence"
        self.metrics.append(SameClassCheck(ds, model, explainer))
