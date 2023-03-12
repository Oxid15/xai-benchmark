import os
import sys

from cascade.models import ModelRepo
from cascade import data as cdd
from cascade import utils as cdu

from xaib.evaluation.example_selection import ExplainerFactory

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

model = cdu.SkModel()
model.load(os.path.join(SCRIPT_DIR, 'model'))

explainers = ExplainerFactory(train_ds, model).get('all')


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
