from typing import Any

from ...base import Case, Dataset, Explainer, Model
from ...metrics.example_selection import SmallNoiseCheck


class ContinuityCase(Case):
    def __init__(
        self,
        ds: Dataset,
        noisy_ds: Dataset,
        model: Model,
        explainer: Explainer,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, ds=ds, model=model, explainer=explainer, **kwargs)
        self.name = "continuity"
        self.metrics.append(SmallNoiseCheck(ds, model, explainer, noisy_ds=noisy_ds))
