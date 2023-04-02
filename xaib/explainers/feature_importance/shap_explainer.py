import numpy as np
import shap

from ...base import Explainer


class ShapExplainer(Explainer):
    def __init__(self, train_ds, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)
        self._train_ds = train_ds
        self._train_data = np.array([item["item"] for item in self._train_ds])

    def predict(self, x, model):
        explainer = shap.Explainer(model.predict, self._train_data)
        return np.array(explainer(x).values)
