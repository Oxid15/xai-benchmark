import os
import sys

import numpy as np
from cascade import data as cdd
from cascade import utils as cdu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)
from utils import MakeClassificationDataset


N_NEIGHBORS = 3


train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'))
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds'))

X_train, Y_train = [x['item'] for x in train_ds], [x['label'] for x in train_ds]
X_test, Y_test = [x['item'] for x in test_ds], [x['label'] for x in test_ds]

X_train = np.array(X_train)
Y_train = np.array(Y_train, dtype=int)
X_test = np.array(X_test)
Y_test = np.array(Y_test, dtype=int)


model = cdu.SkModel(blocks=[KNeighborsClassifier(n_neighbors=N_NEIGHBORS)])
model.fit(X_train, Y_train)
model.evaluate(X_test, Y_test, {'f1': f1_score})
print(model.get_meta())
model.save(os.path.join(SCRIPT_DIR, 'model'))
