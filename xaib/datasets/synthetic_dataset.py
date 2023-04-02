from typing import Any, Dict

from cascade import data as cdd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class SyntheticDataset(cdd.SizedDataset):
    def __init__(self, split, name=None, frac=0.8, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        # Useful for different synthetic datasets
        if name is not None:
            self.name = name
        else:
            self.name = "synthetic"

        x, y = make_classification(*args, **kwargs)

        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=frac)

        if split == "train":
            self.x, self.y = train_x, train_y
        elif split == "test":
            self.x, self.y = test_x, test_y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"item": self.x[index], "label": self.y[index]}
