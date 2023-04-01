import os
import sys

from cascade import data as cdd
from cascade.utils.sk_model import SkModel
from cascade.models import ModelRepo

from xaib.evaluation.feature_importance import ExplainerFactory, ExperimentFactory
from xaib.evaluation import ModelFactory, DatasetFactory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(PROJECT_DIR)

from xaib.evaluation.utils import visualize_results
from xaib.evaluation.model_factory import SkWrapper


REPO_PATH = os.path.join(os.path.dirname(BASE_DIR), "results", "feature_importance")
BS = 5


# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

for dataset in ["synthetic_noisy", "synthetic"]:
    for model in ["svm"]:
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
