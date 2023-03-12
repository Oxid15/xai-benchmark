from ...base import Dataset, Model, Factory
from ...cases.feature_importance import (
    CorrectnessCase, ContinuityCase,
    ContrastivityCase, CoherenceCase, 
    CompactnessCase, CovariateComplexityCase
)

from utils import NoiseApplier, RandomBinBaseline


def correctness(test_ds, model):
    noisy_model = RandomBinBaseline()
    return CorrectnessCase(test_ds, model, noisy_model)


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
        self._constructors['correctness'] = lambda: correctness(test_ds, model)
        self._constructors['continuity'] = lambda: continuity(test_ds, model)
        self._constructors['contrastivity'] = lambda: ContrastivityCase(test_ds, model)
        self._constructors['coherence'] = lambda: CoherenceCase(test_ds, model)
        self._constructors['compactness'] = lambda: CompactnessCase(test_ds, model)
        self._constructors['covariate_complexity'] = lambda: CovariateComplexityCase(test_ds, model)
