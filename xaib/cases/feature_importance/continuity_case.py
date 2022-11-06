from typing import Dict

import numpy as np
from tqdm import tqdm

from ..base import Case, Explainer, Model, Dataset
from ..utils import rmse


class ContinuityCase(Case):
    """
    Apply noise of small magnitude to the input data.
    Obtain original and perturbed explanations.
    Compare them using **MSE or RMSE** and average.
    """
    def __init__(self, ds: Dataset, noisy_ds: Dataset, model: Model) -> None:
        super().__init__(ds, model)
        self.noisy_ds = noisy_ds

    def evaluate(self, expl: Explainer, expl_kwargs: Dict = None) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        rmses = np.zeros(len(self.ds))
        for i in tqdm(range(len(self.ds))):
            item = self.ds[i]
            noisy_item = self.noisy_ds[i]

            explanation = expl.predict(item, self.model, **expl_kwargs)
            noisy_explanation = expl.predict(noisy_item, self.model, **expl_kwargs)

            rmses[i] = rmse(explanation, noisy_explanation)

        self.metrics['continuity'] = {
            'small_noise_check': rmses.mean()
        }
