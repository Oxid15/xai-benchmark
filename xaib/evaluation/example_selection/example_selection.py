import os
import sys

from cascade.models import ModelRepo
from cascade import data as cdd
from cascade.utils.sk_model import SkModel

from xaib.evaluation.example_selection import ExplainerFactory, CaseFactory

from xaib.cases.example_selection import ContinuityCase


SCRIPT_DIR = os.path.dirname(__file__)
# xaib/results/...
REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'results', 'example_selection')

sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import experiment, visualize_results


BS = 5

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)

train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds')).ds()
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'test_ds')).ds()

model = SkModel()
model.load(os.path.join(SCRIPT_DIR, 'model'))

explainers = ExplainerFactory(train_ds, model).get('all')
case_factory = CaseFactory(test_ds, model)


@experiment(REPO_PATH, explainers=explainers, batch_size=BS)
def continuity():
    return case_factory.get('continuity')


continuity()

visualize_results(REPO_PATH, os.path.join(REPO_PATH, 'results.png'))
