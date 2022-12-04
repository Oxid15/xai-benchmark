from typing import Any
import numpy as np
from ...base import Model, Explainer


class RandomExplainer(Explainer):
    def __init__(self, n_features, shift=0, magnitude=1, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)
        self.n_features = n_features
        self._shift = shift
        self._magnitude = magnitude

    def predict(self, x: Any, model: Model) -> Any:
        return np.random.random((len(x), self.n_features)) * self._magnitude + self._shift
