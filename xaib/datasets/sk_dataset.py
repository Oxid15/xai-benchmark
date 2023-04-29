from typing import Any, Dict

import numpy as np
from cascade import data as cdd
from sklearn.datasets import (
    load_iris,
    load_digits,
    load_breast_cancer,
    fetch_covtype,
    fetch_kddcup99,
    fetch_lfw_people,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def prepare_kddcup99():
    x, y = fetch_kddcup99(return_X_y=True)  # 1, 2, 3 - categorical

    enc = OneHotEncoder(sparse_output=False)
    cats = enc.fit_transform(x[:, 1:3])
    x = np.concatenate((x[:, 0].reshape(len(x), 1), cats, x[:, 4:]), axis=1)
    y = OrdinalEncoder().fit_transform(y.reshape(len(y), 1))

    return x, y


class SkDataset(cdd.SizedDataset):
    def __init__(self, name, split, frac=0.8, *args, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = name

        constructors = {
            "iris": load_iris(return_X_y=True),
            "digits": load_digits(return_X_y=True),
            "breast_cancer": load_breast_cancer(return_X_y=True),
            "covtype": fetch_covtype(return_X_y=True),
            "kddcup99": prepare_kddcup99(),
            "lfw_people": fetch_lfw_people(return_X_y=True),
        }

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
