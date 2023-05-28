import numpy as np

from ...base import Explainer


class KNNExplainer(Explainer):
    def __init__(self, train_ds, **kwargs) -> None:
        super().__init__(**kwargs)
        self._train_ds = train_ds

    def predict(self, x, model):
        _, indices = model._pipeline[0].kneighbors(x, 1)
        return [self._train_ds[i[0]] for i in indices]
