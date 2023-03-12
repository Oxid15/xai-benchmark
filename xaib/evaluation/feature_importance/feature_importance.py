import os
import sys

from cascade import data as cdd
from cascade import utils as cdu
from cascade.models import ModelRepo

from xaib.evaluation.feature_importance import CaseFactory
from xaib.evaluation.feature_importance import ExplainerFactory


SCRIPT_DIR = os.path.dirname(__file__)

# xaib/results/...
REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'results', 'feature_importance')

sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import case, visualize_results


BS = 5

# Overwrite previous run
ModelRepo(REPO_PATH, overwrite=True)


train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds')).ds()
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'test_ds')).ds()

model = cdu.SkModel()
model.load(os.path.join(SCRIPT_DIR, 'svm'))

explainers = ExplainerFactory(train_ds, model).get('all')
case_factory = CaseFactory(test_ds, model)


@case(REPO_PATH, explainers=explainers, batch_size=BS)
def correctness():
    return case_factory.get('correctness')


@case(REPO_PATH, explainers=explainers, batch_size=BS)
def continuity() -> None:
    return case_factory.get('continuity')


@case(REPO_PATH, explainers=explainers, batch_size=BS)
def contrastivity():
    return case_factory.get('contrastivity')


@case(REPO_PATH, explainers=explainers, expls=list(explainers.values()), batch_size=BS)
def coherence():
    return case_factory.get('coherence')


@case(REPO_PATH, explainers=explainers, batch_size=BS)
def compactness():
    return case_factory.get('compactness')


@case(REPO_PATH, explainers=explainers, batch_size=BS)
def covariate_complexity():
    return case_factory.get('covariate_complexity')


correctness()
continuity()
contrastivity()
coherence()
compactness()
covariate_complexity()


visualize_results(REPO_PATH, os.path.join(REPO_PATH, 'results.png'))
