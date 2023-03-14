from ...base import Dataset, Model, Factory
from ...cases.example_selection import (
    ContinuityCase
)

from ..utils import NoiseApplier


def continuity(test_ds, model):
    test_ds_noisy = NoiseApplier(test_ds, multiplier=0.01)
    return ContinuityCase(
        test_ds,
        test_ds_noisy,
        model,
        multiplier=0.01
    )


class CaseFactory(Factory):
    def __init__(self, test_ds: Dataset, model: Model) -> None:
        super().__init__()
        self._constructors['continuity'] = lambda: continuity(test_ds, model)
