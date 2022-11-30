import os
import sys
from typing import Dict

from xaib.cases.feature_importance import ContinuityCase
from xaib.base import Explainer

from cascade import utils as cdu
from cascade import data as cdd
from cascade import models as cdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset, NoiseApplier


# SEED = 0
MULTIPLIER = 0.01

def continuity(explainers: Dict[str, Explainer], batch_size: int) -> None:
    test_ds = cdd.Pickler('test_ds')
    test_ds_noisy = NoiseApplier(test_ds, multiplier=MULTIPLIER)

    model = cdu.SkModel()
    model.load('svm')

    c = ContinuityCase(test_ds, test_ds_noisy, model, multiplier=MULTIPLIER)

    for name in explainers:
        c.evaluate(name, explainers[name], batch_size=batch_size)

    repo = cdm.ModelRepo('repo')
    line = repo.add_line('continuity')
    line.save(c, only_meta=True)
