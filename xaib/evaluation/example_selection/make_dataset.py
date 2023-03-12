import os
import sys
from pprint import pprint
from cascade import data as cdd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(BASE_DIR)
from utils import MakeClassificationDataset


params = {
    'n_samples': 1000,
    'n_features': 10,
    'random_state': 0,
    'n_informative': 5,
    'n_redundant': 0,
    'n_repeated': 0,
    'n_clusters_per_class': 1
}

ds = MakeClassificationDataset(**params)
train_ds, test_ds = cdd.split(ds, frac=0.90)

train_ds.update_meta({'n_features': params['n_features']})
train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'), train_ds)
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'test_ds'), test_ds)

pprint(train_ds.get_meta())
pprint(test_ds.get_meta())
