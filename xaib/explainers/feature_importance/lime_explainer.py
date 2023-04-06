import numpy as np
from lime import lime_tabular

from ...base import Explainer


class LimeExplainer(Explainer):
    def __init__(self, train_ds, labels, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)

        data = np.array([item["item"] for item in train_ds])
        self._explainer = lime_tabular.LimeTabularExplainer(
            data, feature_selection="none", training_labels=labels
        )
        self._labels = labels

    def predict(self, x, model):
        if not hasattr(model, "predict_proba"):
            raise ValueError("The model should have `predict_proba` method")

        explanations = []
        for item in x:
            ex = self._explainer.explain_instance(
                item,
                model.predict_proba,
                labels=self._labels,
                num_features=len(self._labels),
            )
            predicted_label = np.argmax(ex.predict_proba)
            ex = ex.as_map()[predicted_label]
            # Sorts by the feature order
            importance_scores = [item[1] for item in sorted(ex)]
            explanations.append(importance_scores)
        return np.asarray(explanations)
