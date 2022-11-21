import numpy as np

from ...base import Explainer


class RandomExplainer(Explainer):
    def __init__(self, train_ds, **kwargs) -> None:
        super().__init__(**kwargs)
        self._train_ds = train_ds

    def predict(self, x):
        return [self._train_ds[np.random.randint(0, len(self._train_ds) - 1)]
            for _ in range(len(x))]
