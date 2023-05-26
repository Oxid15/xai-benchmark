from typing import Any, Dict, Union
import numpy as np
from cascade.utils.sk_model import SkModel
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ..base import Factory


class SkWrapper(SkModel):
    def __init__(self, *args, blocks=None, name=None, **kwargs) -> None:
        super().__init__(*args, blocks=blocks, **kwargs)
        self.name = name

    def fit(self, train_ds, *args, **kwargs):
        X, Y = [x["item"] for x in train_ds], [x["label"] for x in train_ds]

        X = np.array(X)
        Y = np.array(Y, dtype=int)

        super().fit(X, Y, *args, **kwargs)

    def evaluate(self, test_ds, *args, **kwargs):
        X, Y = [x["item"] for x in test_ds], [x["label"] for x in test_ds]

        X = np.array(X)
        Y = np.array(Y, dtype=int)

        super().evaluate(X, Y, *args, **kwargs)


class ModelFactory(Factory):
    def __init__(self, train_ds=None, test_ds=None) -> None:
        super().__init__()
        self._train_ds = train_ds
        self._test_ds = test_ds

        self._constructors["svm"] = lambda: SkWrapper(blocks=[SVC()], name="nn")
        self._constructors["knn"] = lambda: SkWrapper(blocks=[KNeighborsClassifier(n_neighbors=3)], name="knn")
        self._constructors["nn"] = lambda: SkWrapper(blocks=[MLPClassifier()], name="nn")

    def get(self, name: str) -> Dict[str, Any] | Any:
        model = super().get(name)

        model.fit(self._train_ds)
        model.evaluate(self._test_ds, metrics_dict={"f1": lambda x, y: f1_score(x, y, average="macro")})
        return model
