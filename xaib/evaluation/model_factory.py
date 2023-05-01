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


# TODO: merge code repetition
def svm(train_ds, test_ds):
    X_train, Y_train = [x["item"] for x in train_ds], [x["label"] for x in train_ds]
    X_test, Y_test = [x["item"] for x in test_ds], [x["label"] for x in test_ds]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train, dtype=int)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=int)

    model = SkWrapper(blocks=[SVC(probability=True)], name="svm")
    model.fit(X_train, Y_train)
    model.evaluate(X_test, Y_test, {"f1": lambda x, y: f1_score(x, y, average="macro")})
    return model


def knn(train_ds, test_ds):
    X_train, Y_train = [x["item"] for x in train_ds], [x["label"] for x in train_ds]
    X_test, Y_test = [x["item"] for x in test_ds], [x["label"] for x in test_ds]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train, dtype=int)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=int)

    model = SkWrapper(blocks=[KNeighborsClassifier(n_neighbors=3)], name="knn")
    model.fit(X_train, Y_train)
    model.evaluate(X_test, Y_test, {"f1": lambda x, y: f1_score(x, y, average="macro")})
    return model


def nn(train_ds, test_ds):
    X_train, Y_train = [x["item"] for x in train_ds], [x["label"] for x in train_ds]
    X_test, Y_test = [x["item"] for x in test_ds], [x["label"] for x in test_ds]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train, dtype=int)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=int)

    model = SkWrapper(blocks=[MLPClassifier()], name="nn")
    model.fit(X_train, Y_train)
    model.evaluate(X_test, Y_test, {"f1": lambda x, y: f1_score(x, y, average="macro")})
    return model


class ModelFactory(Factory):
    def __init__(self, train_ds=None, test_ds=None) -> None:
        super().__init__()
        self._constructors["svm"] = lambda: svm(train_ds, test_ds)
        self._constructors["knn"] = lambda: knn(train_ds, test_ds)
        self._constructors["nn"] = lambda: nn(train_ds, test_ds)
