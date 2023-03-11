from ...base import Explainer


class KNNExplainer(Explainer):
    def __init__(self, train_ds, **kwargs) -> None:
        self._train_ds = train_ds
        super().__init__(**kwargs)

    def predict(self, x, model):
        _, indices = model._pipeline[0].kneighbors(x, 1)
        return [self._train_ds[i[0]]['item'] for i in indices]
