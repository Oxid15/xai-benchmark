# !python make_dataset.py
# !python train_knn_model.py

from cascade.models import ModelRepo
from cascade.meta import MetricViewer
from cascade import data as cdd
from cascade import utils as cdu

from xaib.explainers.example_selection.constant_explainer import ConstantExplainer
from xaib.explainers.example_selection.random_explainer import RandomExplainer
from xaib.explainers.example_selection.knn_explainer import KNNExplainer

import os
import sys

sys.path.append(os.path.abspath('..'))
from utils import case

BS = 5

# Overwrite previous run
ModelRepo('repo', overwrite=True)

train_ds = cdd.Pickler('train_ds').ds()
test_ds = cdd.Pickler('test_ds')
n_features = train_ds.get_meta()[0]['n_features']
model = cdu.SkModel()
model.load('model')

explainers = {
    'const': ConstantExplainer(train_ds, train_ds[0]['item']),
    'random': RandomExplainer(train_ds),
    'knn': KNNExplainer(train_ds)
}

from xaib.cases.example_selection import ContinuityCase
from utils import NoiseApplier


MULTIPLIER = 0.01

@case(explainers=explainers, batch_size=BS)
def continuity() -> None:
    test_ds_noisy = NoiseApplier(test_ds, multiplier=MULTIPLIER)

    return ContinuityCase(test_ds, test_ds_noisy, model, multiplier=MULTIPLIER)

continuity()

repo = ModelRepo('repo')
t = MetricViewer(repo).table
print(t)
