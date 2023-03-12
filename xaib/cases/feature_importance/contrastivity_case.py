from typing import Any
from ...base import Dataset, Model, Case
from ...metrics.feature_importance import LabelDifference


class ContrastivityCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__('contrastivity', ds, model, *args, **kwargs)
        self._metric_objs['label_difference'] = LabelDifference(ds, model)
