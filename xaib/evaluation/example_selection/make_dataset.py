import os
import sys
from pprint import pprint
from cascade import data as cdd
from xaib.evaluation import DatasetFactory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


train_ds, test_ds = DatasetFactory().get('synthetic')

cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'), train_ds)
cdd.Pickler(os.path.join(SCRIPT_DIR, 'test_ds'), test_ds)

pprint(train_ds.get_meta())
pprint(test_ds.get_meta())
