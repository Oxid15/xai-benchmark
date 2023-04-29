import matplotlib

matplotlib.use("TkAgg")
import os
import sys

from cascade import data as cdd
from cascade.models import ModelRepo
from xaib.evaluation import DatasetFactory, ModelFactory
from xaib.evaluation.feature_importance import ExperimentFactory, ExplainerFactory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(PROJECT_DIR)

from xaib.evaluation.utils import visualize_results

REPO_PATH = os.path.join(os.path.dirname(BASE_DIR), "results", "feature_importance")
BS = 100
MAX_EVAL_SAMPLES = 1000

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

for dataset in [
    "lfw_people",
    "kddcup99",
    "covtype",
    "breast_cancer",
    "digits",
    "iris",
    "synthetic_noisy",
    "synthetic",
]:
    for model in ["svm", "nn"]:
        print(dataset, model)

        train_ds, test_ds = DatasetFactory().get(dataset)
        print(train_ds.get_meta())

        labels = train_ds.labels

        model = ModelFactory(train_ds, test_ds).get(model)
        print(model.get_meta())

        if len(test_ds) > MAX_EVAL_SAMPLES:
            test_ds = cdd.RandomSampler(test_ds, MAX_EVAL_SAMPLES)

        explainers = ExplainerFactory(train_ds, model, labels=labels).get("all")
        experiment_factory = ExperimentFactory(
            REPO_PATH, explainers, test_ds, model, labels, BS
        )

        experiments = experiment_factory.get("all")
        for name in experiments:
            experiments[name]()

visualize_results(REPO_PATH, REPO_PATH)
