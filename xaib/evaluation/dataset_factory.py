from ..base import Factory
from ..datasets.synthetic_dataset import SyntheticDataset


def generate_dataset(name=None, frac: float = 0.9, **kwargs):
    train_ds, test_ds = (
        SyntheticDataset("train", name=name, frac=frac, **kwargs),
        SyntheticDataset("test", name=name, frac=frac, **kwargs),
    )
    return train_ds, test_ds


class DatasetFactory(Factory):
    def __init__(self) -> None:
        super().__init__()
        self._constructors["synthetic"] = lambda: generate_dataset(
            n_samples=100,
            n_features=14,
            random_state=0,
            n_informative=14,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            frac=0.8,
        )
        self._constructors["synthetic_noisy"] = lambda: generate_dataset(
            name="synthetic_noisy",
            n_samples=100,
            n_features=14,
            random_state=0,
            n_informative=7,
            n_redundant=5,
            n_repeated=2,
            n_clusters_per_class=2,
            frac=0.8,
        )
