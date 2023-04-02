from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import Filter, SimpleDataloader, entropy, minmax_normalize


class CovariateRegularity(Metric):
    """
    Measures how noisy the features in the examples provided
    by the method.

    More simple explanations are considered better.
    This is measured by average Shannon entropy over batch-normalized explanations.

    **The less the better**

      - **Worst case:** constant explainer that gives examples with same importance to each feature, that is equal to 1/N where N is the number of features
      - **Best case:** constant explainer that gives examples with one feature with maximum value and others zero
    """

    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "covariate_regularity"
        self.direction = "down"

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

                explanations[u] += ex.tolist()

        # Compare explanations of different labels
        entropies_of_features = {u: [] for u in unique_labels}
        for u in unique_labels:
            expls = np.asarray(explanations[u])
            for f in range(expls.shape[1]):
                e = entropy(expls[:, f])
                entropies_of_features[u].append(e)

        return np.nanmean([entropies_of_features[u] for u in entropies_of_features])
