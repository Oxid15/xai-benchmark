import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ...base import Explainer


class KNNExplainer(Explainer):
    def __init__(self, train_ds, **kwargs) -> None:
        super().__init__(**kwargs)
        self._train_ds = train_ds
        self.name = "knn"

        X, Y = [x["item"] for x in train_ds], [x["label"] for x in train_ds]

        X = np.array(X)
        Y = np.array(Y, dtype=int)

        self._explainer = KNeighborsClassifier(n_neighbors=1)
        self._explainer.fit(X, Y)

    def predict(self, x, model):
        _, indices = self._explainer.kneighbors(x)
        return [self._train_ds[i[0]] for i in indices]


class KNNCosineExplainer(KNNExplainer):
    def __init__(self, *args, **kwargs):
        kwargs.update({"metric": "cosine"})
        super().__init__(*args, **kwargs)
        self.name = "knn_cosine"
