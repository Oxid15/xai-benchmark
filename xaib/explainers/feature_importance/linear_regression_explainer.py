import numpy as np

from ...base import Explainer


class LinearRegressionExplainer(Explainer):
    def __init__(self, get_coef) -> None:
        super().__init__()
        self._get_coef = get_coef

    def predict(self, x, model):
        explanations = np.asarray(x) * self._get_coef(model)
        return np.asarray(explanations)
