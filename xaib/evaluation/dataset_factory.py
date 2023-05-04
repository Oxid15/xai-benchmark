from ..base import Factory
from ..datasets.synthetic_dataset import SyntheticDataset
from ..datasets.sk_dataset import SkDataset


def generate_dataset(ds_cls, *args, **kwargs):
    train_ds, test_ds = (
        ds_cls(split="train", *args, **kwargs),
        ds_cls(split="test", *args, **kwargs),
    )
    return train_ds, test_ds


class DatasetFactory(Factory):
    def __init__(self) -> None:
        super().__init__()
        self._constructors["synthetic"] = lambda: generate_dataset(
            SyntheticDataset,
            n_samples=1000,
            n_features=14,
            random_state=0,
            n_informative=14,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            frac=0.8,
        )
        self._constructors["synthetic_noisy"] = lambda: generate_dataset(
            SyntheticDataset,
            name="synthetic_noisy",
            n_samples=1000,
            n_features=14,
            random_state=0,
            n_informative=7,
            n_redundant=5,
            n_repeated=2,
            n_clusters_per_class=2,
            frac=0.8,
        )
        self._constructors["iris"] = lambda: generate_dataset(
            SkDataset, "iris", frac=0.8
        )
        self._constructors["wine"] = lambda: generate_dataset(
            SkDataset, "wine", frac=0.8
        )
        self._constructors["digits"] = lambda: generate_dataset(
            SkDataset, "digits", frac=0.8
        )
        self._constructors["breast_cancer"] = lambda: generate_dataset(
            SkDataset, "breast_cancer", frac=0.8
        )
