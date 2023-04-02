from typing import Any, Dict, Union

import numpy as np

from ...base import Explainer, Model


class ConstantExplainer(Explainer):
    def __init__(self, constant: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.constant = constant

    def predict(self, x: Any, model: Model) -> Any:
        return np.full((len(x), len(x[0])), self.constant)
