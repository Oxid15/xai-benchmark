from ...base import Dataset, Factory, Model
from ...explainers.example_selection.constant_explainer import ConstantExplainer
from ...explainers.example_selection.knn_explainer import KNNExplainer
from ...explainers.example_selection.random_explainer import RandomExplainer


class ExplainerFactory(Factory):
    def __init__(self, train_ds: Dataset = None, model: Model = None) -> None:
        super().__init__()
        self._constructors["const"] = lambda: ConstantExplainer(
            train_ds, train_ds[0]["item"]
        )
        self._constructors["random"] = lambda: RandomExplainer(train_ds)
        self._constructors["knn"] = lambda: KNNExplainer(train_ds)
