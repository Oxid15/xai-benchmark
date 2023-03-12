from typing import Any
from ...base import Dataset, Model, Case
from ...metrics.example_selection import SmallNoiseCheck


class ContinuityCase(Case):
    def __init__(self, ds: Dataset, noisy_ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__('continuity', ds, model, *args, **kwargs)
        self._metric_objs['small_noise_check'] = SmallNoiseCheck(ds, noisy_ds, model)
