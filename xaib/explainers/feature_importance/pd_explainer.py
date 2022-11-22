from sklearn.inspection import partial_dependence
import numpy as np
from cascade.models import ModelModifier
from ...base import Explainer


class FittedClassifier(ModelModifier):
    def __init__(self, model, labels, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self._estimator_type = 'classifier'
        self.classes_ = labels

    def __sklearn_is_fitted__(self):
        return True
    
    def predict_proba(self, *args, **kwargs):
        return self._model.predict_proba(*args, **kwargs)


class PDExplainer(Explainer):
    def __init__(self, train_ds, labels, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = np.array([item['item'] for item in train_ds])
        self._labels = labels

    def predict(self, x, model):
        if not hasattr(model, 'predict_proba'):
            raise ValueError('The model should have `predict_proba` method')

        pds = []
        for i in range(len(x[0])):
            pd = partial_dependence(FittedClassifier(model, self._labels), x,
                features=[i],
                grid_resolution=10)
            pds.append(pd['average'])
        return np.asarray(pds).T
