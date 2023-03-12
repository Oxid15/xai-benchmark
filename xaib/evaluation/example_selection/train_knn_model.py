import os
import sys

from xaib.evaluation import ModelFactory
from cascade import data as cdd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset


N_NEIGHBORS = 3


train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'))
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'))

model = ModelFactory(train_ds, test_ds).get('knn')
print(model.get_meta())
model.save(os.path.join(SCRIPT_DIR, 'model'))
