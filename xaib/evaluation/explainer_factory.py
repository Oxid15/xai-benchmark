from ..base import Dataset, Model, Factory
from ..explainers.feature_importance.constant_explainer import ConstantExplainer
from ..explainers.feature_importance.random_explainer import RandomExplainer


class ExplainerFactory(Factory):
    def __init__(self, train_ds: Dataset, model: Model) -> None:
        super().__init__()
        self._constructors['const'] = lambda: ConstantExplainer(constant=1)
        self._constructors['random'] = lambda: RandomExplainer(shift=-15, magnitude=10)
        # self._constructors['shap'] = lambda: ShapExplainer(train_ds)
        # self._constructors['lime'] = lambda: LimeExplainer(train_ds, labels=(0, 1))
