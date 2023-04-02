from typing import Dict, List

import numpy as np
from cascade.data import Composer, Sampler, SizedDataset


def rmse(x, y):
    err = x - y
    mse = np.sum(err * err)
    return np.sqrt(mse)


def batch_rmse(bx, by):
    return [rmse(x, y) for (x, y) in zip(bx, by)]


def entropy(x):
    x += 1e-6
    return -np.sum(x * np.log2(x))


def gini(x):
    n = len(x)
    ad = 0
    for i in range(n):
        for j in range(n):
            ad += np.abs(x[i] - x[j])
    if np.mean(x) == 0:
        return 0
    else:
        gini = ad / (2 * n * n * np.mean(x))
    return gini


def batch_gini(x):
    return [gini(i) for i in x]


def batch_count_eq(x, y) -> List[bool]:
    return [all(xi == yi) for xi, yi in zip(x, y)]


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)

    # If all values are 0 then return original
    # If all values are equal, then return ones
    if min_val == max_val:
        if min_val == 0:
            return x
        else:
            return np.ones_like(x)

    x_scaled = (x + np.abs(min_val)) / (max_val + np.abs(min_val))
    return x_scaled


class SimpleDataloader:
    def __init__(self, data, batch_size: int = 1) -> None:
        self._data = data
        self._bs = batch_size
        self._channels = list(data[0].keys())

    def __getitem__(self, index: int) -> Dict:
        batch = {ch: [] for ch in self._channels}

        start_index = index * self._bs
        end_index = min((index + 1) * self._bs, len(self._data))
        for i in range(start_index, end_index):
            item = self._data[i]
            for ch in self._channels:
                batch[ch].append(item[ch])

        for ch in batch:
            batch[ch] = np.array(batch[ch])

        return batch

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return int(np.ceil(len(self._data) / self._bs))


class KeyedComposer(Composer):
    def __init__(self, datasets, *args, **kwargs) -> None:
        super().__init__(list(datasets.values()), *args, **kwargs)
        self._keys = datasets.keys()

    def __getitem__(self, index: int):
        data_tuple = super().__getitem__(index)
        return {key: val for key, val in zip(self._keys, data_tuple)}


class Filter(Sampler):
    def __init__(self, ds: SizedDataset, indices: List[int], **kwargs) -> None:
        super().__init__(ds, num_samples=len(indices), **kwargs)

        self._indices = indices

    def __getitem__(self, index: int):
        return self._dataset[self._indices[index]]
