import os
import sys
from typing import Dict

from xaib.base import Explainer
from xaib.cases.feature_importance import CoherenceCase

from cascade import utils as cdu
from cascade import data as cdd
from cascade import models as cdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset


# SEED = 0

def coherence(explainers: Dict[str, Explainer], batch_size: int) -> None:
    test_ds = cdd.Pickler('test_ds')

    model = cdu.SkModel()
    model.load('svm')

    c = CoherenceCase(test_ds, model)
    repo = cdm.ModelRepo('repo')
    line = repo.add_line('coherence')

    for name in explainers:
        c.evaluate(
            name, explainers[name],
            [explainers[another]
                for another in explainers if another != name],
            batch_size=batch_size)

        line.save(c, only_meta=True)
