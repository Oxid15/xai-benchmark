from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, batch_count_eq


class SmallNoiseCheck(Metric):
    """
    Since the nature of example selection methods is discrete - we choose an example
    from finite set, the use of RMSE measure may be not appropriate.
    This means that the metric that is similar to the one that was used in feature
    importance case is not suitable.
    More appropriate metric could be the following: take the test dataset and add
    small amount of noise to the items as was done in the feature importance case.
    Then count the number of item pairs when the example provided didn't change and
    divide by the total number of items.
    This ratio then is how continuous example generator is - if it provides the same
    examples for slightly changed inputs, then it is continuous.

    **The less the better**
     - **Worst case:** Constant explainer
     - **Best case:** Random explainer
    """

    def __init__(
        self, ds: Dataset, noisy_ds: Dataset, model: Model, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self._noisy_ds = noisy_ds
        self.name = "small_noise_check"
        self.direction = "up"

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        counts = []
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

            counts += batch_count_eq(explanation_batch, noisy_explanation_batch)

        return np.sum(counts) / len(self._ds)
