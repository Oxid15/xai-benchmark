import os
import sys

from .conftest import test_feature_importance_explainer_interface

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from xaib.explainers.shap_explainer import ShapExplainer


def test_explainer(train_test_pair, dummy_model):
    train_x, test_x, train_y, test_y = train_test_pair
    explainer = ShapExplainer(train_x)

    test_feature_importance_explainer_interface(explainer, dummy_model, test_x)
