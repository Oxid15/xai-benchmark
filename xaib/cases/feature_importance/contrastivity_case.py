from typing import List, Union, Dict, Any

from tqdm import tqdm
import numpy as np
from cascade.data import Sampler

from ...utils import batch_rmse, minmax_normalize, SimpleDataloader
from ...base import Dataset, Case, Explainer


class Filter(Sampler):
    def __init__(self, ds: Dataset, indices: List[int], **kwargs) -> None:
        super().__init__(ds, num_samples=len(indices), **kwargs)

        self._indices = indices

    def __getitem__(self, index: int) -> Any:
        return self._dataset[self._indices[index]]


class ContrastivityCase(Case):
    """
    Measures how different explanations
    are actually different from each other
    """
    def evaluate(
        self,
        name: str,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {} 

        # Obtain all the labels
        labels = np.asarray([item['label'] for item in
                             tqdm(self._ds, desc='Obtaining labels', leave=False)])

        # Determine how much of an intersection different labels have
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_len = np.min(counts)

        # For each label get indexes of its items
        coords = {u: np.argwhere(labels == u).T[0][:min_len] for u in unique_labels}

        # Obtain all explanations by batches
        explanations = {u: [] for u in unique_labels}
        for u in unique_labels:
            dl = SimpleDataloader(Filter(self._ds, coords[u]), batch_size=batch_size)
            for batch in tqdm(dl):
                ex = expl.predict(batch['item'], self._model, **expl_kwargs)
                ex = minmax_normalize(ex)
                explanations[u].append(ex)

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

        self.params['name'] = name
        self.metrics['label_difference'] = np.nanmean([diffs[u] for u in diffs])
