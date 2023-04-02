import os
import sys

from cascade import data as cdd
from cascade.models import ModelRepo
from cascade.utils.sk_model import SkModel
from xaib.evaluation import DatasetFactory, ModelFactory
from xaib.evaluation.example_selection import ExperimentFactory, ExplainerFactory

SCRIPT_DIR = os.path.dirname(__file__)
# xaib/results/...
REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(SCRIPT_DIR)), "results", "example_selection"
)

sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import WrapperModel, visualize_results


class SkWrapper(SkModel):
    def __init__(self, *args, blocks=None, name=None, **kwargs) -> None:
        super().__init__(*args, blocks=blocks, **kwargs)
        self.name = name


BS = 5

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

for dataset in ["synthetic_noisy", "synthetic"]:
    for model in ["knn"]:
        train_ds, test_ds = DatasetFactory().get(dataset)
        print(train_ds.get_meta())

        model = ModelFactory(train_ds, test_ds).get(model)
        print(model.get_meta())

        explainers = ExplainerFactory(train_ds, model).get("all")
        experiment_factory = ExperimentFactory(
            REPO_PATH, explainers, test_ds, model, BS
        )

        experiments = experiment_factory.get("all")
        for name in experiments:
            experiments[name]()

visualize_results(REPO_PATH, REPO_PATH)
