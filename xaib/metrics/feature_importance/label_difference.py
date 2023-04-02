from typing import Any, Dict, Union

import numpy as np
from cascade.data import Sampler
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import Filter, SimpleDataloader, batch_rmse, minmax_normalize


class LabelDifference(Metric):
    """
    Obtain explanations for all targets and compare them.
    In binary case one metric can be computed - the difference
    between explanations of positive and negative targets.
    In case of multiclass classification we can compare one explanation
    to explanations of all other classes and obtain a number of metrics
    which can be averaged.

    **The more the better**

    - **Worst case:** same explanations for different targets - constant explainer
    - **Best case:** different explanations for different targets
    """

    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "label_difference"
        self.direction = "up"

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        # Obtain all the labels
        labels = np.asarray(
            [
                item["label"]
                for item in tqdm(self._ds, desc="Obtaining labels", leave=False)
            ]
        )

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
                ex = expl.predict(batch["item"], self._model, **expl_kwargs)
                ex = minmax_normalize(ex)
                explanations[u].append(ex)

        # Compare explanations of different labels
        diffs = {u: [] for u in unique_labels}
        for u in unique_labels:
            for other_label in unique_labels:
                if u == other_label:
                    continue

                for batch, other_batch in zip(
                    explanations[u], explanations[other_label]
                ):
                    rmse = batch_rmse(batch, other_batch)
                    diffs[u] += list(rmse)
        for u in unique_labels:
            diffs[u] = np.nanmean(diffs[u])

        return np.nanmean([diffs[u] for u in diffs])
