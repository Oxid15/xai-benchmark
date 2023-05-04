import matplotlib

matplotlib.use("TkAgg")

import os
import sys

from cascade.models import ModelRepo
from xaib.evaluation import DatasetFactory, ModelFactory
from xaib.evaluation.feature_importance import (
    ExperimentFactory,
    ExplainerFactory,
    CaseFactory,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(PROJECT_DIR)

from xaib.evaluation.utils import visualize_results, Setup


REPO_PATH = os.path.join(os.path.dirname(BASE_DIR), "results", "feature_importance")
BS = 100

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

factories = (DatasetFactory(), ModelFactory(), ExplainerFactory(), CaseFactory())
setups = [Setup(*factories, models_except=["knn"])]

for setup in setups:
    for dataset in setup.datasets:
        for model in setup.models:
            print(dataset, model)

            train_ds, test_ds = DatasetFactory().get(dataset)
            print(train_ds.get_meta())

            labels = train_ds.labels

            model = ModelFactory(train_ds, test_ds).get(model)
            print(model.get_meta())

            explainers = {
                explainer: ExplainerFactory(train_ds, model, labels=labels).get(
                    explainer
                )
                for explainer in setup.explainers
            }

            experiment_factory = ExperimentFactory(
                REPO_PATH, explainers, test_ds, model, labels, BS
            )

            for case in setup.cases:
                experiment = experiment_factory.get(case)
                experiment()

visualize_results(REPO_PATH, REPO_PATH)
