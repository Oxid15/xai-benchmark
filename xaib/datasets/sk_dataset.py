from typing import Any, Dict

import numpy as np
from cascade import data as cdd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class SkDataset(cdd.SizedDataset):
    def __init__(self, name, split, frac=0.8, *args, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = name

        constructors = {"iris": load_iris(return_X_y=True)}

        if name not in constructors:
            raise ValueError(
                f"{name} not in SkDataset use one of {list(constructors.keys())}"
            )

        x, y = constructors[name]
        self.labels = np.unique(y)

        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=frac)

        if split == "train":
            self.x, self.y = train_x, train_y
        elif split == "test":
            self.x, self.y = test_x, test_y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"item": self.x[index], "label": self.y[index]}
