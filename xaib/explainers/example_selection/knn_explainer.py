import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ...base import Explainer


class KNNExplainer(Explainer):
    def __init__(self, train_ds, **kwargs) -> None:
        super().__init__(**kwargs)
        self._train_ds = train_ds

        self._explainer = KNeighborsClassifier(n_neighbors=1)

    def predict(self, x, model):
        _, indices = self._explainer.kneighbors(x)
        return [self._train_ds[i[0]] for i in indices]
