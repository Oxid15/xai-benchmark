from xaib import Factory
from xaib.evaluation.example_selection import CaseFactory
from ..utils import experiment


class ExperimentFactory(Factory):
    def __init__(self, repo_path, explainers, test_ds, model, batch_size) -> None:
        super().__init__()

        case_factory = CaseFactory(test_ds, model)

        @experiment(repo_path, explainers=explainers, batch_size=batch_size)
        def continuity():
            return case_factory.get('continuity')


        self._constructors['continuity'] = lambda: continuity
