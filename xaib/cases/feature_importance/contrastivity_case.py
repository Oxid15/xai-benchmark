from typing import Any

from ...base import Case, Dataset, Model
from ...metrics.feature_importance import LabelDifference


class ContrastivityCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "contrastivity"
        self._metric_objs["label_difference"] = LabelDifference(ds, model)
