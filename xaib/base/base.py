from typing import Union, List, Dict, Any, Callable
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
        self.params['case'] = name

        self._metric_objs = dict()

    def add_metric(self, name: str, metric: Metric) -> None:
        self._metric_objs[name] = metric

    def evaluate(self, name: str, expl: Explainer, metrics_kwargs: Union[List[Dict[Any, Any]], None] = None, **kwargs: Any) -> None:
        if metrics_kwargs is None:
            metrics_kwargs = [{} for _ in range(len(self._metric_objs))]

        for mname, mkwargs in zip(self._metric_objs, metrics_kwargs):
            self._metric_objs[mname].evaluate(name, expl, **mkwargs, **kwargs)

            self.params['name'] = name
            self.metrics.update(self._metric_objs[mname].metrics)


class Factory:
    """
    Collection of constructors to build and return objects
    from predefined hardcoded or added dynamically constructors
    """
    def __init__(self) -> None:
        self._constructors = dict()
        self._constructors_kwargs = dict()

    def _get_all(self) -> Dict[str, Any]:
        return {
                name: self._get(name)
                for name in self._constructors
            }

    def _get(self, name: str) -> Any:
        if name not in self._constructors_kwargs:
            kwargs = {}
        else:
            kwargs = self._constructors_kwargs[name]

        constructor = self._constructors[name]
        return constructor(**kwargs)

    def get(self, name: str) -> Union[Dict[str, Any], Any]:
        if name == 'all':
            return self._get_all()
        return self._get(name)

    def add(self, name: str, constructor: Callable[[Any], Any], constr_kwargs: Union[Any, None] = None) -> None:
        self._constructors[name] = constructor
        self._constructors_kwargs[name] = constr_kwargs
