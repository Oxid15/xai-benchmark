from cascade.models import ModelRepo
from cascade.meta import MetricViewer
from cascade import data as cdd
from cascade import utils as cdu

from xaib.explainers.feature_importance.constant_explainer import ConstantExplainer
from xaib.explainers.feature_importance.random_explainer import RandomExplainer
from xaib.explainers.feature_importance.shap_explainer import ShapExplainer
from xaib.explainers.feature_importance.lime_explainer import LimeExplainer

import os
import sys

sys.path.append(os.path.abspath('..'))
from utils import case

BS = 5

# Overwrite previous run
ModelRepo('repo', overwrite=True)

train_ds = cdd.Pickler('train_ds').ds()
test_ds = cdd.Pickler('test_ds').ds()
n_features = train_ds.get_meta()[0]['n_features']

model = cdu.SkModel()
model.load('svm')

explainers = {
    'const': ConstantExplainer(n_features=n_features, constant=1),
    'random': RandomExplainer(n_features=n_features, shift=-15, magnitude=10),
    # 'shap': ShapExplainer(train_ds),
    # 'lime': LimeExplainer(train_ds, labels=(0, 1))
}


from xaib.cases.feature_importance import CorrectnessCase

from utils import RandomBinBaseline

@case(explainers=explainers, batch_size=BS)
def correctness():
    noisy_model = RandomBinBaseline()

    c = CorrectnessCase(test_ds, model, noisy_model)
    return c


correctness()


from xaib.cases.feature_importance import ContinuityCase

from utils import NoiseApplier

MULTIPLIER = 0.01

@case(explainers=explainers, batch_size=BS)
def continuity() -> None:
    test_ds_noisy = NoiseApplier(test_ds, multiplier=MULTIPLIER)

    c = ContinuityCase(test_ds, test_ds_noisy, model, multiplier=MULTIPLIER)
    return c

continuity()

from xaib.cases.feature_importance import ContrastivityCase

@case(explainers=explainers, batch_size=BS)
def contrastivity():
    return ContrastivityCase(test_ds, model)

contrastivity()

from xaib.cases.feature_importance import CoherenceCase

@case(explainers=explainers, expls=list(explainers.values()), batch_size=BS)
def coherence():
    return CoherenceCase(test_ds, model)

coherence()

from xaib.cases.feature_importance import CompactnessCase

@case(explainers=explainers, batch_size=BS)
def compactness():
    return CompactnessCase(test_ds, model)

compactness()

from xaib.cases.feature_importance import CovariateComplexityCase

@case(explainers=explainers, batch_size=BS)
def covariate_complexity():
    return CovariateComplexityCase(test_ds, model)

covariate_complexity()

repo = ModelRepo('repo')

t = MetricViewer(repo).table
print(t)
