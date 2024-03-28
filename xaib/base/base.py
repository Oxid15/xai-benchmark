from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from cascade.data import Dataset as CascadeDataset
from cascade.metrics import Metric as CascadeMetric
from cascade.metrics import MetricType as CascadeMetricType
from cascade.models import Model as CascadeModel

from ..version import __version__ as version


class Dataset(CascadeDataset):
    """
    Dataset is a wrapper around any collection of items to put in the ML model
    for inference
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = None

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index)


class Model(CascadeModel):
    """
    Model is a wrapper around any inference of ML or other solution in the form y = f(x)
    it implements method `predict` that given certain data x returns the response y
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = None

    def predict(self, x: Any) -> Any:
        raise NotImplementedError()


class Explainer(CascadeModel):
    """
    Explainer is a special kind of Model e = g(f, x) that accepts another Model and data as input
    and also returns a response e - an explanation
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = None

    def predict(self, x: Any, model: Model) -> Any:
        raise NotImplementedError()


class Metric(CascadeMetric):
    """
    Metric is an entity which accepts Explainer, Model and Dataset and outputs a metric
    corresponding to the quality of Explainer v = m(g, f, x)
    """

    def __init__(
        self,
        name: str,
        direction: Literal["up", "down"],
        ds: Dataset,
        model: Model,
        *args: Any,
        split: Optional[str] = None,
        interval: Optional[Tuple[CascadeMetricType, CascadeMetricType]] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            direction=direction,
            dataset=ds.name,
            split=split,
            interval=interval,
            extra=extra,
            *args,
            **kwargs,
        )

        self._ds = ds
        self._model = model

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> CascadeMetricType:
        raise NotImplementedError()


class Case(CascadeModel):
    """
    Case is a collection of Metrics which represent some
    high-level property of an Explainer
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = None
        self._meta_prefix.update({"xaib_version": version})

    def evaluate(
        self,
        expl: Explainer,
        metrics_kwargs: Union[Dict[str, Dict[Any, Any]], None] = None,
        **kwargs: Any,
    ) -> None:
        if metrics_kwargs is None:
            metrics_kwargs = {metric.name: {} for metric in self.metrics}

        for metric in self.metrics:
            m_name = metric.name

            mkwargs = {}
            if m_name in metrics_kwargs:
                mkwargs = metrics_kwargs[m_name]

            metric.compute(expl, **mkwargs, **kwargs)


class Factory:
    """
    Collection of constructors to build and return objects
    from predefined hardcoded or added dynamically constructors
    """

    def __init__(self) -> None:
        self._constructors = dict()
        self._constructors_kwargs = dict()

    def _get_all(self) -> Dict[str, Any]:
        return {name: self._get(name) for name in self._constructors}

    def _get(self, name: str) -> Any:
        if name not in self._constructors_kwargs:
            kwargs = {}
        else:
            kwargs = self._constructors_kwargs[name]

        constructor = self._constructors[name]
        return constructor(**kwargs)

    def get(self, name: str) -> Union[Dict[str, Any], Any]:
        try:
            if name == "all":
                return self._get_all()
            return self._get(name)
        except Exception as e:
            raise RuntimeError(f"Failed to create object {name} in {self}") from e

    def get_constructor(self, name: str) -> Callable:
        return self._constructors[name]

    def add(
        self,
        name: str,
        constructor: Type,
        constr_kwargs: Union[Any, None] = None,
    ) -> None:
        self._constructors[name] = constructor
        self._constructors_kwargs[name] = constr_kwargs

    def get_names(self) -> List[str]:
        return list(self._constructors.keys())
