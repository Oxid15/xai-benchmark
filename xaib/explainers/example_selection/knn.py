from sklearn.neighbors import KNeighborsClassifier

from ...base import Explainer


class KNN(Explainer):
    def __init__(self, train_ds, n_neighbors=5, metric='minkowski', **kwargs) -> None:
        super().__init__(**kwargs)
        self._train_ds = train_ds

        X, Y = [], []
        for item in self._train_ds:
            X.append(item['item'])
            Y.append(item['label'])

        self._model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        self._model.fit(X, Y)

    def predict(self, x):
        _, indices = self._model.kneighbors(x, 1)
        return [self._train_ds[i[0]]['item'] for i in indices]
