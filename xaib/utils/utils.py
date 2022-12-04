from typing import Dict, List
import numpy as np


def rmse(x, y):
    err = x - y
    mse = np.sum(err * err)
    return np.sqrt(mse)


def batch_rmse(bx, by):
    return [rmse(x, y) for (x, y) in zip(bx, by)]


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
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
        return batch

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return int(np.ceil(len(self._data) / self._bs))
