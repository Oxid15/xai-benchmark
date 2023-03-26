from typing import Dict, Any, Union

import numpy as np
from tqdm import tqdm

from ...base import Metric, Explainer, Model, Dataset
from ...utils import batch_count_eq, SimpleDataloader


class SmallNoiseCheck(Metric):
    """
    Continuity measures how continuous
    explanation function is
    """

    def __init__(
        self,
        ds: Dataset,
        noisy_ds: Dataset,
        model: Model,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self._noisy_ds = noisy_ds
        self.name = 'small_noise_check'
        self.direction = 'up'

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        counts = []
        dls = zip(
            SimpleDataloader(self._ds, batch_size),
            SimpleDataloader(self._noisy_ds, batch_size)
        )
        for batch, noisy_batch in tqdm(dls):
            item = batch['item']
            noisy_item = noisy_batch['item']

            explanation_batch = expl.predict(item, self._model, **expl_kwargs)
            noisy_explanation_batch = expl.predict(noisy_item, self._model, **expl_kwargs)

            counts += batch_count_eq(explanation_batch, noisy_explanation_batch)

        return np.sum(counts) / (len(self._ds) * len(self._ds[0]['item']))
