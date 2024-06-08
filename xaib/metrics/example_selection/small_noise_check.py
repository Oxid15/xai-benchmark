from typing import Any

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Metric
from ...utils import ChannelDataloader, batch_count_eq


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

    def __init__(self, ds, model, explainer, noisy_ds: Dataset, *args, **kwargs: Any) -> None:
        self._noisy_ds = noisy_ds
        super().__init__("small_noise_check", "up", ds, model, explainer, *args, **kwargs)

    def compute(
        self,
        batch_size: int = 1,
    ) -> None:
        counts = []
        dls = zip(
            ChannelDataloader(self._ds, batch_size),
            ChannelDataloader(self._noisy_ds, batch_size),
        )
        for batch, noisy_batch in tqdm(dls):
            item = batch["item"]
            noisy_item = noisy_batch["item"]

            explanation_batch = self._explainer.predict(item, self._model, **self._explainer_kwargs)
            noisy_explanation_batch = self._explainer.predict(
                noisy_item, self._model, **self._explainer_kwargs
            )

            explanation_batch = np.asarray([item["item"] for item in explanation_batch])
            noisy_explanation_batch = np.asarray([item["item"] for item in noisy_explanation_batch])

            counts += batch_count_eq(explanation_batch, noisy_explanation_batch)

        self.value = np.sum(counts) / len(self._ds)
        return self.value
