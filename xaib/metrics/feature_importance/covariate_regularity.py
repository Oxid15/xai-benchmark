from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, entropy, minmax_normalize


class CovariateRegularity(Metric):
    """
    Covariate Regularity using entropy over explanations

    This measures how comprehensible the explanations are in average.
    More simple explanations are considered better.
    This is measured by average Shannon entropy over batch-normalized explanations.

    **The less the better**
     - **Worst case:** constant explainer that gives same importance to each feature, that is equal to 1/N where N is the number of features
     - **Best case:** constant explainer that gives one feature maximum value and others zero
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
    ):
        if expl_kwargs is None:
            expl_kwargs = {}

        explanations = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch["item"]

            e = expl.predict(item, self._model, **expl_kwargs)
            e = minmax_normalize(e)
            explanations += e.tolist()

        explanations = np.array(explanations, dtype=float)

        entropies_of_features = []
        for f in range(explanations.shape[1]):
            e = entropy(explanations[:, f])
            entropies_of_features.append(e)

        return np.nanmean(entropies_of_features)
