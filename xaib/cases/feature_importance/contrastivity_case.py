from typing import Iterable

from tqdm import tqdm
import numpy as np
from cascade.data import Sampler

from ...utils import batch_rmse, SimpleDataloader
from ...base import Case, Explainer


class Filter(Sampler):
    def __init__(self, ds, indices: Iterable[int], **kwargs) -> None:
        super().__init__(ds, num_samples=len(indices), **kwargs)

        self._indices = indices
    
    def __getitem__(self, index):
        return self._dataset[self._indices[index]]


class ContrastivityCase(Case):
    def evaluate(self, name: str, expl: Explainer, batch_size: int = 1) -> None:
        # Obtain all the labels
        labels = np.asarray([item['label'] for item in
            tqdm(self._ds, desc='Obtaining labels', leave=False)])

        # Determine how much of an intersection different labels have
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_len = np.min(counts)

        # For each label get indexes of its items
        coords = {u: np.argwhere(labels == u)[0][:min_len] for u in unique_labels}

        # Obtain all explanations by batches
        explanations = {u: [] for u in unique_labels}
        for u in unique_labels:
            dl = SimpleDataloader(Filter(self._ds, coords[u]), batch_size=batch_size)
            for batch in tqdm(dl):
                explanations[u].append(expl.predict(batch['item'], self._model))

        # Compare explanations of different labels
        diffs = {u: [] for u in unique_labels}
        for u in unique_labels:
            for other_label in unique_labels:
                if u == other_label:
                    continue

                for batch, other_batch in zip(
                    explanations[u],
                    explanations[other_label]
                ):
                    rmse = batch_rmse(batch, other_batch)
                    diffs[u] += list(rmse)
        for u in unique_labels:
            diffs[u] = np.nanmean(diffs[u])

        self.metrics[name] = {}
        self.metrics[name]['contrastivity'] = {
            'label_difference': np.nanmean([diffs[u] for u in diffs])
        }
