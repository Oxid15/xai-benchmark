from typing import Any, Dict, List, Sequence

import cascade.data as cdd
import numpy as np

from ..base import Dataset


class ChannelDataloader(cdd.SimpleDataloader):
    def __init__(self, data, batch_size: int = 1) -> None:
        super().__init__(data, batch_size)
        self._channels = list(data[0].keys())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        batch = super().__getitem__(index)
        new_batch = dict()
        for ch in batch:
            new_batch[ch] = np.array(batch[ch])
        return new_batch


class KeyedComposer(cdd.Composer):
    def __init__(self, datasets: Dict[str, Dataset], *args, **kwargs) -> None:
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
