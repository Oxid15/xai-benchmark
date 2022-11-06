import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from base import Explainer, Model

N_FEATURES = 20


class DummyModel(Model):
    def predict(self, x):
        return np.array([np.random.choice((0, 1)) for _ in range(len(x))])


@pytest.fixture
def train_test_pair():
    x, y = make_classification(n_features=N_FEATURES)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return x_train, x_test, y_train, y_test


@pytest.fixture
def dummy_model():
    model = DummyModel()
    return model


def test_feature_importance_explainer_interface(
    explainer: Explainer,
    model: Model,
    input_data: np.ndarray):

    # should be in form [b, n], where b - batch size and n - number of features
    input_shape = input_data.shape
    assert len(input_shape) == 2 

    attributions = explainer.predict(input_data, model)

    assert input_shape == attributions.shape
