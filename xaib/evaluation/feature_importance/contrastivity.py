import os
import sys
from typing import Dict

from xaib.cases.feature_importance import ContrastivityCase
from xaib.base import Explainer

from cascade import utils as cdu
from cascade import data as cdd
from cascade import models as cdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset


# SEED = 0

def contrastivity(explainers: Dict[str, Explainer], batch_size: int) -> None:
    test_ds = cdd.Pickler('test_ds')

    model = cdu.SkModel()
    model.load('svm')

    c = ContrastivityCase(test_ds, model)
    repo = cdm.ModelRepo('repo')
    line = repo.add_line('contrastivity')

    for name in explainers:
        c.evaluate(name, explainers[name], batch_size=batch_size)
        line.save(c, only_meta=True)
