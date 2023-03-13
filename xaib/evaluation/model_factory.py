import numpy as np
from cascade.utils.sk_model import SkModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from ..base import Factory
from .utils import WrapperModel


def svm(train_ds, test_ds):
    X_train, Y_train = [x['item'] for x in train_ds], [x['label'] for x in train_ds]
    X_test, Y_test = [x['item'] for x in test_ds], [x['label'] for x in test_ds]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train, dtype=int)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=int)

    model = SkModel(blocks=[SVC(probability=True)])
    model.fit(X_train, Y_train)
    model.evaluate(X_test, Y_test, {'f1': f1_score})
    return WrapperModel(model, 'svm')


def knn(train_ds, test_ds):
    X_train, Y_train = [x['item'] for x in train_ds], [x['label'] for x in train_ds]
    X_test, Y_test = [x['item'] for x in test_ds], [x['label'] for x in test_ds]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train, dtype=int)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=int)


    model = SkModel(blocks=[KNeighborsClassifier(n_neighbors=3)])
    model.fit(X_train, Y_train)
    model.evaluate(X_test, Y_test, {'f1': f1_score})
    return WrapperModel(model, 'knn')


class ModelFactory(Factory):
    def __init__(self, train_ds, test_ds) -> None:
        super().__init__()
        self._constructors['svm'] = lambda: svm(train_ds, test_ds)
        self._constructors['knn'] = lambda: knn(train_ds, test_ds)