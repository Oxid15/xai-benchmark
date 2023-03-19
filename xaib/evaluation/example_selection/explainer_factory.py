from ...base import Dataset, Model, Factory
from ...explainers.example_selection.constant_explainer import ConstantExplainer
from ...explainers.example_selection.random_explainer import RandomExplainer
from ...explainers.example_selection.knn_explainer import KNNExplainer


class ExplainerFactory(Factory):
    def __init__(self, train_ds: Dataset, model: Model) -> None:
        super().__init__()
        self._constructors['const'] = lambda: ConstantExplainer(train_ds, train_ds[0])
        self._constructors['random'] = lambda: RandomExplainer(train_ds)
        self._constructors['knn'] = lambda: KNNExplainer(train_ds)        
