from ...base import Dataset, Factory, Model
from ...explainers.feature_importance.constant_explainer import ConstantExplainer
from ...explainers.feature_importance.lime_explainer import LimeExplainer
from ...explainers.feature_importance.random_explainer import RandomExplainer
from ...explainers.feature_importance.shap_explainer import ShapExplainer


class ExplainerFactory(Factory):
    def __init__(
        self, train_ds: Dataset = None, model: Model = None, labels=None
    ) -> None:
        super().__init__()
        self._constructors["const"] = lambda: ConstantExplainer(constant=1)
        self._constructors["random"] = lambda: RandomExplainer(shift=-15, magnitude=10)
        self._constructors["shap"] = lambda: ShapExplainer(train_ds)
        self._constructors["lime"] = lambda: LimeExplainer(train_ds, labels=labels)
