from typing import Dict, List

import cascade.data as cdd
import numpy as np

from ..base import Dataset


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


class KeyedComposer(cdd.Composer):
    def __init__(self, datasets: Dict, *args, **kwargs) -> None:
        super().__init__(list(datasets.values()), *args, **kwargs)
        self._keys = datasets.keys()

    def __getitem__(self, index: int):
        data_tuple = super().__getitem__(index)
        return {key: val for key, val in zip(self._keys, data_tuple)}


class Filter(cdd.Sampler):
    def __init__(self, ds: Dataset, indices: List[int], **kwargs) -> None:
        super().__init__(ds, num_samples=len(indices), **kwargs)

        self._indices = indices

    def __getitem__(self, index: int):
        return self._dataset[self._indices[index]]


class NoiseApplier(cdd.Modifier):
    def __init__(
        self, dataset: Dataset, multiplier: float = 1.0, *args, **kwargs
    ) -> None:
        super().__init__(dataset, *args, **kwargs)
        self._multiplier = multiplier

        data = np.asarray([item["item"] for item in dataset])
        means = (data * data).mean(axis=0)
        self._stds = np.sqrt(means * multiplier)

    def __getitem__(self, index):
        item = self._dataset.__getitem__(index)

        noises = [np.random.normal(0, scale) for scale in self._stds]

        item["item"] = item["item"] + noises
        return item
