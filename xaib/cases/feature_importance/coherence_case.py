from typing import Any
from ...base import Dataset, Model, Case
from ...metrics.feature_importance import OtherDisagreement


class CoherenceCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__('coherence', ds, model, *args, **kwargs)
        self._metric_objs['other_disagreement'] = OtherDisagreement(ds, model)
