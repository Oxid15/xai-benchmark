import os
import sys
from typing import Dict

from xaib.cases.feature_importance import CovariateComplexityCase
from xaib.base import Explainer

from cascade import utils as cdu
from cascade import data as cdd
from cascade import models as cdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset, NoiseApplier


# SEED = 0

def covariate_complexity(explainers: Dict[str, Explainer], batch_size: int) -> None:
    test_ds = cdd.Pickler('test_ds')
    test_ds = cdd.CyclicSampler(test_ds, 10)

    model = cdu.SkModel()
    model.load('svm')

    c = CovariateComplexityCase(test_ds, model)
    repo = cdm.ModelRepo('repo')
    line = repo.add_line('covariate_complexity')

    for name in explainers:
        c.evaluate(name, explainers[name], batch_size=batch_size)
        line.save(c, only_meta=True)
