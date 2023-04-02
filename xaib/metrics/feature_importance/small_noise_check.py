from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, batch_rmse, minmax_normalize


class SmallNoiseCheck(Metric):
    """
    Apply noise of small magnitude to the input data.
    Obtain original and perturbed explanations.
    Compare them using RMSE and average.
    **The less the better**
     - **Worst case:** is when explanations are hugely changed by the small variations in input
     - **Best case:** is no variations, so constant explainer should achieve best results
    """

    def __init__(
        self, ds: Dataset, noisy_ds: Dataset, model: Model, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self._noisy_ds = noisy_ds
        self.name = "small_noise_check"
        self.direction = "down"

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        rmses = []
        dls = zip(
            SimpleDataloader(self._ds, batch_size),
            SimpleDataloader(self._noisy_ds, batch_size),
        )
        for batch, noisy_batch in tqdm(dls):
            item = batch["item"]
            noisy_item = noisy_batch["item"]

            explanation_batch = expl.predict(item, self._model, **expl_kwargs)
            noisy_explanation_batch = expl.predict(
                noisy_item, self._model, **expl_kwargs
            )

            explanation_batch = minmax_normalize(explanation_batch)
            noisy_explanation_batch = minmax_normalize(noisy_explanation_batch)

            rmses += batch_rmse(explanation_batch, noisy_explanation_batch)

        return np.nanmean(rmses)
