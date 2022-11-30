import os
import sys
from pprint import pprint
from cascade import data as cdd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset


params = {
    'n_samples': 1000,
    'n_features': 2,
    'random_state': 0,
    'n_informative': 1,
    'n_redundant': 0,
    'n_repeated': 0,
    'n_clusters_per_class': 1
}

ds = MakeClassificationDataset(**params)
train_ds, test_ds = cdd.split(ds, frac=0.99)

train_ds.update_meta({'n_features': params['n_features']})
train_ds = cdd.Pickler('train_ds', train_ds)
test_ds = cdd.Pickler('test_ds', test_ds)

pprint(train_ds.get_meta())
pprint(test_ds.get_meta())

import numpy as np
from matplotlib import pyplot as plt
X_train, Y_train = np.array([x['item'] for x in train_ds]), np.array([x['label'] for x in train_ds])

_, ax = plt.subplots(1, params['n_features'])

for f in range(params['n_features']):
    ax[f].scatter(X_train[:, f], Y_train)
plt.show()
