from xaib import Factory
from xaib.evaluation.example_selection import CaseFactory

from ..utils import experiment


class ExperimentFactory(Factory):
    def __init__(
        self, repo_path=None, explainers=None, test_ds=None, model=None, batch_size=None
    ) -> None:
        super().__init__()

        case_factory = CaseFactory(test_ds, model)

        @experiment(repo_path, explainers=explainers, batch_size=batch_size)
        def continuity():
            return case_factory.get("continuity")

        @experiment(repo_path, explainers=explainers, batch_size=batch_size)
        def contrastivity():
            return case_factory.get("contrastivity")

        @experiment(repo_path, explainers=explainers, batch_size=batch_size)
        def covariate_complexity():
            return case_factory.get("covariate_complexity")

        @experiment(repo_path, explainers=explainers, batch_size=batch_size)
        def correctness():
            return case_factory.get("correctness")

        self._constructors["continuity"] = lambda: continuity
        self._constructors["contrastivity"] = lambda: contrastivity
        self._constructors["covariate_complexity"] = lambda: covariate_complexity
        self._constructors["correctness"] = lambda: correctness
