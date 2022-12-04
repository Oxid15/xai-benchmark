import os
import sys
from typing import Dict

from xaib.explainers.feature_importance.constant_explainer import ConstantExplainer
from xaib.explainers.feature_importance.random_explainer import RandomExplainer
from xaib.explainers.feature_importance.shap_explainer import ShapExplainer
from xaib.explainers.feature_importance.lime_explainer import LimeExplainer
from xaib.cases.feature_importance import CorrectnessCase
from xaib.base import Explainer

from cascade import utils as cdu
from cascade import models as cdm
from cascade import data as cdd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset, RandomBinBaseline


def correctness(explainers: Dict[str, Explainer], batch_size: int) -> None:
    test_ds = cdd.Pickler('test_ds')

    model = cdu.SkModel()
    model.load('svm')

    noisy_model = RandomBinBaseline()

    c = CorrectnessCase(test_ds, model, noisy_model)
    repo = cdm.ModelRepo('repo')
    line = repo.add_line('correctness')

    for name in explainers:
        c.evaluate(name, explainers[name], batch_size=batch_size)
        line.save(c, only_meta=True)
