from typing import Any

import numpy as np

from ...base import Explainer, Model


class RandomExplainer(Explainer):
    def __init__(self, shift=0, magnitude=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._shift = shift
        self._magnitude = magnitude

    def predict(self, x: Any, model: Model) -> Any:
        return np.random.random((len(x), len(x[0]))) * self._magnitude + self._shift
