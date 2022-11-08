from typing import Dict

import numpy as np
from tqdm import tqdm

from ...base import Case, Explainer, Model, Dataset
from ...utils import rmse, SimpleDataloader


class ContinuityCase(Case):
    """
    Apply noise of small magnitude to the input data.
    Obtain original and perturbed explanations.
    Compare them using **MSE or RMSE** and average.
    """
    def __init__(self, ds: Dataset, noisy_ds: Dataset, model: Model) -> None:
        super().__init__(ds, model)
        self._noisy_ds = noisy_ds

    def evaluate(self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Dict = None) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        rmses = []
        dls = zip(
            SimpleDataloader(self._ds, batch_size),
            SimpleDataloader(self._noisy_ds, batch_size)
        )
        for batch, noisy_batch in tqdm(dls):
            item = batch['item']
            noisy_item = noisy_batch['item']

            explanation = expl.predict(item, self._model, **expl_kwargs)
            noisy_explanation = expl.predict(noisy_item, self._model, **expl_kwargs)

            rmses.append(rmse(explanation, noisy_explanation))

        self.metrics['continuity'] = {
            'small_noise_check': np.nanmean(rmses)
        }
