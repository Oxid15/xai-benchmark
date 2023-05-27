from cascade import models as cdm
import numpy as np


class RandomBaseline(cdm.BasicModel):
    def __init__(self, labels, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)
        self._labels = labels

    def predict(self, x):
        return np.array([np.random.choice(self._labels) for _ in range(len(x))])

    def predict_proba(self, x):
        proba = np.random.random((len(x), len(self._labels)))
        return softmax(proba, axis=1)


class KNeighborsTransformer:
    def __init__(self, dataset_length) -> None:
        self._dataset_length = dataset_length

    def kneighbors(self, x, n_neighbors):
        return None, np.asarray(
            [[np.random.randint(0, self._dataset_length)] for _ in range(n_neighbors)]
        )


class RandomNeighborsBaseline(cdm.BasicModel):
    def __init__(self, dataset_length, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)
        self._pipeline = [KNeighborsTransformer(dataset_length)]
