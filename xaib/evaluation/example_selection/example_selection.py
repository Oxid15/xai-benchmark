import os
import sys

from cascade.models import ModelRepo
from cascade.utils.sk_model import SkModel
from xaib.evaluation import DatasetFactory, ModelFactory
from xaib.evaluation.example_selection import (
    ExperimentFactory,
    ExplainerFactory,
    CaseFactory,
)
from xaib.utils import ModelCache

SCRIPT_DIR = os.path.dirname(__file__)
# xaib/results/...
REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(SCRIPT_DIR)),
    "results",
    "example_selection_coherence_model_f1",
)

sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import Setup, visualize_results


BS = 100

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

factories = (DatasetFactory(), ModelFactory(), ExplainerFactory(), CaseFactory())
setups = [Setup(*factories, models=["knn"], cases=["coherence"], explainers=["knn"])]

for setup in setups:
    for dataset in setup.datasets:
        train_ds, test_ds = DatasetFactory().get(dataset)
        model_factory = ModelCache(ModelFactory(train_ds, test_ds))

        for model in setup.models:
            print(train_ds.get_meta())

            model = model_factory.get(model, key=dataset)
            print(model.get_meta())

            explainers = {
                explainer: ExplainerFactory(train_ds, model).get(explainer)
                for explainer in setup.explainers
            }

            experiment_factory = ExperimentFactory(
                REPO_PATH, explainers, test_ds, model, BS
            )

            for case in setup.cases:
                experiment = experiment_factory.get(case)
                experiment()

visualize_results(REPO_PATH, REPO_PATH)
