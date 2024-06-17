import os
import sys

from cascade.models import ModelRepo

from xaib.evaluation import DatasetFactory, ModelFactory
from xaib.evaluation.example_selection import CaseFactory, ExplainerFactory
from xaib.utils import ModelCache

SCRIPT_DIR = os.path.dirname(__file__)
# xaib/results/...
REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(SCRIPT_DIR)), "results", "example_selection"
)

sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import Setup, run_experiment

BS = 100

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

factories = (DatasetFactory(), ModelFactory(), ExplainerFactory(), CaseFactory())
setups = [Setup(*factories, models_except=["knn"])]

for setup in setups:
    for dataset in setup.datasets:
        train_ds, test_ds = DatasetFactory().get(dataset)
        model_factory = ModelCache(ModelFactory(train_ds, test_ds))

        for model in setup.models:
            model = model_factory.get(model, key=dataset)
            print(f"Model: {model.name} trained on: {train_ds.name}")

            for case in setup.cases:
                print(f"Evaluating: {case}")
                for explainer in setup.explainers:
                    print(f"Explainer: {explainer}")
                    explainer = ExplainerFactory(train_ds, model).get(explainer)
                    case_obj = CaseFactory(test_ds, model, explainer).get(case)
                    success = run_experiment(case_obj, REPO_PATH)
                    print("Success\n" if success else "Failed\n")

# visualize_results(REPO_PATH, REPO_PATH)
