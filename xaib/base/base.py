from typing import Any
from cascade import data as cdd
from cascade import models as cdm


class Dataset(cdd.Dataset):
    '''
    Dataset is a wrapper around any collection of items to put in the ML model
    for inference
    '''
    def __getitem__(self, index):
        super().__getitem__(index)


class Model(cdm.Model):
    '''
    Model is a wrapper around any inference of ML or other solution in the form y = f(x)
    it implements method `predict` that given certain data x returns the response y
    '''
    def predict(self, x: Any) -> Any:
        raise NotImplementedError()


class Explainer(cdm.Model):
    '''
    Explainer is a special kind of Model e = g(f, x) that accepts another Model and data as input
    and also returns a response e - an explanation
    '''

    def predict(self, x: Any, model: Model) -> Any:
        raise NotImplementedError()


class Case(cdm.Model):
    '''
    Case is an entity which accepts Explainer, Model and Dataset and outputs a metric
    corresponding to the quality of Explainer m = c(g, f, x)
    '''
    def evaluate(self, ds: Dataset, model: Model, expl: Explainer) -> None:
        raise NotImplementedError()
