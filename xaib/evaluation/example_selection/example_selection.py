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

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import case, visualize_results


BS = 5

# Overwrite previous run
ModelRepo('repo', overwrite=True)

train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds')).ds()
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'test_ds')).ds()
n_features = train_ds.get_meta()[0]['n_features']
model = cdu.SkModel()
model.load(os.path.join(SCRIPT_DIR, 'model'))

explainers = {
    'const': ConstantExplainer(train_ds, train_ds[0]['item']),
    'random': RandomExplainer(train_ds),
    'knn': KNNExplainer(train_ds)
}

from xaib.cases.example_selection import ContinuityCase
from utils import NoiseApplier


MULTIPLIER = 0.01

@case(SCRIPT_DIR, explainers=explainers, batch_size=BS)
def continuity() -> None:
    test_ds_noisy = NoiseApplier(test_ds, multiplier=MULTIPLIER)

    return ContinuityCase(test_ds, test_ds_noisy, model, multiplier=MULTIPLIER)

continuity()

visualize_results(os.path.join(SCRIPT_DIR, 'repo'), os.path.join(SCRIPT_DIR, 'results.png'))
