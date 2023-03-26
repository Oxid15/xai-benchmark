from ...base import Dataset, Model, Factory
from ...cases.example_selection import (
    ContinuityCase, ContrastivityCase, CovariateComplexityCase,
    CorrectnessCase
)

from ..utils import NoiseApplier, RandomNeighborsBaseline


def continuity(test_ds, model):
    test_ds_noisy = NoiseApplier(test_ds, multiplier=0.01)
    return ContinuityCase(
        test_ds,
        test_ds_noisy,
        model,
        multiplier=0.01
    )

def correctness(test_ds, model):
    noisy_model = RandomNeighborsBaseline(len(test_ds))

    return CorrectnessCase(
        test_ds,
        model,
        noisy_model
    )


class CaseFactory(Factory):
    def __init__(self, test_ds: Dataset, model: Model) -> None:
        super().__init__()
        self._constructors['continuity'] = lambda: continuity(test_ds, model)
        self._constructors['contrastivity'] = lambda: ContrastivityCase(test_ds, model)
        self._constructors['covariate_complexity'] = lambda: CovariateComplexityCase(test_ds, model)
        self._constructors['correctness'] = lambda: correctness(test_ds, model)
