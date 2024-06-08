from ...base import Dataset, Explainer, Factory, Model
from ...cases.example_selection import (
    CoherenceCase,
    ContinuityCase,
    ContrastivityCase,
    CorrectnessCase,
    CovariateComplexityCase,
)
from ...utils import NoiseApplier, RandomNeighborsBaseline


def continuity(test_ds, model, explainer):
    test_ds_noisy = NoiseApplier(test_ds, multiplier=0.01)
    return ContinuityCase(test_ds, test_ds_noisy, model, explainer, multiplier=0.01)


def correctness(test_ds, model, explainer):
    noisy_model = RandomNeighborsBaseline(len(test_ds))
    return CorrectnessCase(test_ds, model, explainer, noisy_model)


class CaseFactory(Factory):
    def __init__(
        self, dataset: Dataset = None, model: Model = None, explainer: Explainer = None
    ) -> None:
        super().__init__()
        self._constructors["continuity"] = lambda: continuity(dataset, model, explainer)
        self._constructors["contrastivity"] = lambda: ContrastivityCase(dataset, model, explainer)
        self._constructors["covariate_complexity"] = lambda: CovariateComplexityCase(dataset, model, explainer)
        self._constructors["correctness"] = lambda: correctness(dataset, model, explainer)
        self._constructors["coherence"] = lambda: CoherenceCase(dataset, model, explainer)
