from typing import Any
from ...base import Dataset, Model, Case
from ...metrics.feature_importance import ParameterRandomizationCheck


class CorrectnessCase(Case):
    def __init__(self, ds: Dataset, model: Model, noisy_model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__('correctness', ds, model, *args, **kwargs)
        self._metric_objs['parameter_randomization_check'] = ParameterRandomizationCheck(ds, model, noisy_model)
