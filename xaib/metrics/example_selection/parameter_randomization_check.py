from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, batch_count_eq, minmax_normalize


class ParameterRandomizationCheck(Metric):
    def __init__(
        self, ds: Dataset, model: Model, noisy_model: Model, **kwargs: Any
    ) -> None:
        super().__init__(ds, model, **kwargs)
        self._noisy_model = noisy_model
        self.name = "parameter_randomization_check"
        self.direction = "down"

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

            diffs_expl += batch_count_eq(explanation_batch, noisy_explanation_batch)

        return sum(diffs_expl) / len(diffs_expl)
