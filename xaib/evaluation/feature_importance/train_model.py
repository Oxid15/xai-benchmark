import os
import sys

import numpy as np
from cascade import data as cdd
from cascade import utils as cdu
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from xaib.evaluation import ModelFactory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(PROJECT_DIR)
from datasets import SyntheticDataset


train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'))
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'))

model = ModelFactory(train_ds, test_ds).get('svm')

print(model.get_meta())
model.save(os.path.join(SCRIPT_DIR, 'svm'))
