from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, batch_rmse, minmax_normalize


class ModelRandomizationCheck(Metric):
    """
    Model randomization check is a sanity-check.
    To ensure that the model influence explanations the
    following is done. The model is changed and it is expected that
    explanations should not stay the same is model changed.
    This check uses random model baselines instead of same models
    with randomized internal states.
    Then the explanations on the original data are obtained.
    They are compared with explanations done with the original model using
    average RMSE on the whole dataset.
    The further original explanations from the explanations on
    the randomized model the better.

    **The greater the better**
     - **Worst case:** explanations are the same, so it is Constant explainer.
     - **Best case:** is reached when explanations are the opposite, distance between them maximized. The problem with this kind of metric is with its maximization. It seems redundant to maximize it because more different explanations on random states do not mean that the model is more correct.
    It is difficult to define best case explainer in this case - the metric has no maximum value.
    """

    def __init__(
        self, ds: Dataset, model: Model, noisy_model: Model, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, **kwargs)
        self._noisy_model = noisy_model
        self.name = "model_randomization_check"
        self.direction = "up"

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
        expl_noisy_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expl_noisy_kwargs is None:
            expl_noisy_kwargs = {}

        diffs_expl = []

        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch["item"]

            explanation_batch = expl.predict(item, self._model, **expl_kwargs)
            noisy_explanation_batch = expl.predict(
                item, self._noisy_model, **expl_noisy_kwargs
            )

            explanation_batch = minmax_normalize(explanation_batch)
            noisy_explanation_batch = minmax_normalize(noisy_explanation_batch)

            diffs_expl += batch_rmse(explanation_batch, noisy_explanation_batch)

        return np.nanmean(diffs_expl)
