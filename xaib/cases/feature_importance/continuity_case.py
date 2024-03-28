from typing import Any

from ...base import Case, Dataset, Model
from ...metrics.feature_importance import SmallNoiseCheck


class ContinuityCase(Case):
    def __init__(
        self, ds: Dataset, noisy_ds: Dataset, model: Model, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "continuity"
        self.metrics.append(SmallNoiseCheck(ds, noisy_ds, model))
