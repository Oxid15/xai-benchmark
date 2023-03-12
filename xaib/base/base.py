from typing import Any
from cascade import data as cdd
from cascade import models as cdm


class Dataset(cdd.Dataset):
    """
    Dataset is a wrapper around any collection of items to put in the ML model
    for inference
    """
    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index)


class Model(cdm.Model):
    """
    Model is a wrapper around any inference of ML or other solution in the form y = f(x)
    it implements method `predict` that given certain data x returns the response y
    """
    def predict(self, x: Any) -> Any:
        raise NotImplementedError()


class Explainer(cdm.Model):
    """
    Explainer is a special kind of Model e = g(f, x) that accepts another Model and data as input
    and also returns a response e - an explanation
    """

    def predict(self, x: Any, model: Model) -> Any:
        raise NotImplementedError()


class Metric(cdm.Model):
    """
    Metric is an entity which accepts Explainer, Model and Dataset and outputs a metric
    corresponding to the quality of Explainer v = m(g, f, x)
    """
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ds = ds
        self._model = model

    def evaluate(self, name: str, expl: Explainer, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()


class Case(cdm.Model):
    """
    Case is a collection of Metrics which represent some
    high-level property of an Explainer
    """
    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._name = name
        self._metrics = dict()

    def add_metric(self, name: str, metric: Metric) -> None:
        self._metrics[name] = metric

    def evaluate(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()
