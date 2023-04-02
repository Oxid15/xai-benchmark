import numpy as np

from ...base import Explainer


class ConstantExplainer(Explainer):
    def __init__(self, train_ds, example, **kwargs) -> None:
        super().__init__(**kwargs)
        self._example = example

    def predict(self, x, model):
        return np.asarray([self._example for _ in range(len(x))])
