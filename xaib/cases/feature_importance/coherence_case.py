from typing import Any

from ...base import Case, Dataset, Model
from ...metrics.feature_importance import OtherDisagreement


class CoherenceCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "coherence"
        self._metric_objs["other_disagreement"] = OtherDisagreement(ds, model)
