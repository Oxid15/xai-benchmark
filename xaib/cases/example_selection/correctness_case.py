from typing import Any

from ...base import Case, Dataset, Model
from ...metrics.example_selection import ParameterRandomizationCheck


class CorrectnessCase(Case):
    def __init__(
        self, ds: Dataset, model: Model, noisy_model: Model, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "correctness"
        self._metric_objs[
            "parameter_randomization_check"
        ] = ParameterRandomizationCheck(ds, model, noisy_model)
