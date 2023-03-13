import os
import sys
from cascade import data as cdd

from ..base import Factory
from .utils import MakeClassificationDataset, WrapperDataset


def generate_dataset(frac: float = 0.8, **kwargs):
    ds = MakeClassificationDataset(**kwargs)
    train_ds, test_ds = cdd.split(ds, frac=frac)
    train_ds.update_meta(kwargs)

    return (
        WrapperDataset(train_ds, 'synthetic'),
        WrapperDataset(test_ds, 'synthetic')
    )


class DatasetFactory(Factory):
    def __init__(self) -> None:
        super().__init__()
        self._constructors['synthetic'] = lambda: generate_dataset(
            n_samples=1000,
            n_features=10,
            random_state=0,
            n_informative=5,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            frac=0.9
        )
