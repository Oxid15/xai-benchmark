import os
import sys

from cascade import data as cdd
from cascade import utils as cdu
from cascade.models import ModelRepo

from xaib.explainers.feature_importance.constant_explainer import ConstantExplainer
from xaib.explainers.feature_importance.random_explainer import RandomExplainer
from xaib.explainers.feature_importance.shap_explainer import ShapExplainer
from xaib.explainers.feature_importance.lime_explainer import LimeExplainer

from xaib.cases.feature_importance import (
    CorrectnessCase, ContinuityCase,
    ContrastivityCase, CoherenceCase, CompactnessCase, CovariateComplexityCase
)

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.dirname(SCRIPT_DIR)))
from utils import case, visualize_results


BS = 5

# Overwrite previous run
ModelRepo(os.path.join(SCRIPT_DIR, 'repo'), overwrite=True)

train_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'train_ds')).ds()
test_ds = cdd.Pickler(os.path.join(SCRIPT_DIR, 'test_ds')).ds()
n_features = train_ds.get_meta()[0]['n_features']

model = cdu.SkModel()
model.load(os.path.join(SCRIPT_DIR, 'svm'))

explainers = {
    'const': ConstantExplainer(n_features=n_features, constant=1),
    'random': RandomExplainer(n_features=n_features, shift=-15, magnitude=10),
    # 'shap': ShapExplainer(train_ds),
    # 'lime': LimeExplainer(train_ds, labels=(0, 1))
}


from utils import RandomBinBaseline


@case(SCRIPT_DIR, explainers=explainers, batch_size=BS)
def correctness():
    noisy_model = RandomBinBaseline()

    c = CorrectnessCase(test_ds, model, noisy_model)
    return c


correctness()


from utils import NoiseApplier


MULTIPLIER = 0.01


@case(SCRIPT_DIR, explainers=explainers, batch_size=BS)
def continuity() -> None:
    test_ds_noisy = NoiseApplier(test_ds, multiplier=MULTIPLIER)

    c = ContinuityCase(test_ds, test_ds_noisy, model, multiplier=MULTIPLIER)
    return c


continuity()


@case(SCRIPT_DIR, explainers=explainers, batch_size=BS)
def contrastivity():
    return ContrastivityCase(test_ds, model)


contrastivity()


@case(SCRIPT_DIR, explainers=explainers, expls=list(explainers.values()), batch_size=BS)
def coherence():
    return CoherenceCase(test_ds, model)


coherence()


@case(SCRIPT_DIR, explainers=explainers, batch_size=BS)
def compactness():
    return CompactnessCase(test_ds, model)


compactness()


@case(SCRIPT_DIR, explainers=explainers, batch_size=BS)
def covariate_complexity():
    return CovariateComplexityCase(test_ds, model)


covariate_complexity()


visualize_results(os.path.join(SCRIPT_DIR, 'repo'), os.path.join(SCRIPT_DIR, 'repo', 'results.png'))
