import os
import sys
from cascade import data as cdd

from ..base import Factory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from utils import MakeClassificationDataset


def generate_dataset(frac: float = 0.8, **kwargs):
    ds = MakeClassificationDataset(**kwargs)
    train_ds, test_ds = cdd.split(ds, frac=frac)
    train_ds.update_meta(kwargs)

    return train_ds, test_ds


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
