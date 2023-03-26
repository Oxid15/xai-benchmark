from ..base import Factory
from ..datasets.synthetic_dataset import SyntheticDataset


def generate_dataset(frac: float = 0.9, **kwargs):
    train_ds, test_ds = (SyntheticDataset('train', frac=frac, **kwargs),
                         SyntheticDataset('test', frac=frac, **kwargs))
    return train_ds, test_ds


class DatasetFactory(Factory):
    def __init__(self) -> None:
        super().__init__()
        self._constructors['synthetic'] = lambda: generate_dataset(
            n_samples=1000,
            n_features=10,
            random_state=0,
            n_informative=5,
            n_redundant=1,
            n_repeated=0,
            n_clusters_per_class=1,
            frac=0.9
        )
