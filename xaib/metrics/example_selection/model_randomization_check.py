from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Explainer, Metric, Model
from ...utils import ChannelDataloader, batch_count_eq


class ModelRandomizationCheck(Metric):
    """
    Model randomization check is a sanity-check.
    To ensure that the model influence explanations the
    following is done. The model is changed and it is expected that
    explanations should not stay the same is model changed.
    This check uses random model baselines instead of same models
    with randomized internal states.
    Then the explanations on the original data are obtained.
    They are compared with explanations done with the original model by
    counting how many examples were the same for same data points.

    **The less the better**
     - **Worst case:** explanations are the same, so it is Constant explainer
     - **Best case:** is reached when explanations are the opposite,
     distance between them maximized.
    The problem with this kind of metric is
    with its maximization. It seems redundant to maximize it because more
    different explanations on random states do not mean that the model is
    more correct.
    It is difficult to define best case explainer in this case - the metric has no maximum value.
    """

    def __init__(self, ds, model, explainer, noisy_model: Model, *args: Any, **kwargs: Any) -> None:
        self._noisy_model = noisy_model
        super().__init__(
            name="model_randomization_check",
            direction="down",
            ds=ds,
            model=model,
            explainer=explainer,
            *args,
            **kwargs
        )

    def compute(
        self,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
        expl_noisy_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expl_noisy_kwargs is None:
            expl_noisy_kwargs = {}

        diffs_expl = []

        for batch in tqdm(ChannelDataloader(self._ds, batch_size)):
            item = batch["item"]

            explanation_batch = self._explainer.predict(item, self._model, **expl_kwargs)
            noisy_explanation_batch = self._explainer.predict(item, self._noisy_model, **expl_noisy_kwargs)

            explanation_batch = np.asarray([item["item"] for item in explanation_batch])
            noisy_explanation_batch = np.asarray([item["item"] for item in noisy_explanation_batch])

            diffs_expl += batch_count_eq(explanation_batch, noisy_explanation_batch)

        self.value = sum(diffs_expl) / len(diffs_expl)
        return self.value
