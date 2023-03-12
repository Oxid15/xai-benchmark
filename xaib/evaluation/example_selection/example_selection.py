import os
import sys

from cascade.models import ModelRepo
from cascade import data as cdd
from cascade import utils as cdu

from xaib.explainers.example_selection.constant_explainer import ConstantExplainer
from xaib.explainers.example_selection.random_explainer import RandomExplainer
from xaib.explainers.example_selection.knn_explainer import KNNExplainer

from xaib.cases.example_selection import ContinuityCase


SCRIPT_DIR = os.path.dirname(__file__)
# xaib/results/...
REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'results', 'example_selection')

sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import case, visualize_results


BS = 5

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

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


from utils import NoiseApplier

MULTIPLIER = 0.01


@case(REPO_PATH, explainers=explainers, batch_size=BS)
def continuity():
    test_ds_noisy = NoiseApplier(test_ds, multiplier=MULTIPLIER)
    return ContinuityCase(
        test_ds,
        test_ds_noisy,
        model,
        multiplier=MULTIPLIER
    )


continuity()

visualize_results(REPO_PATH, os.path.join(REPO_PATH, 'results.png'))
