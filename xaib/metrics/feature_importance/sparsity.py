from typing import Any, Dict, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, batch_gini, minmax_normalize


class Sparsity(Metric):
    """
    Considering Gini-index as a measure of sparsity, one can give an
    average of it as a measure of sparsity for explanations.
    **The greater the better**
      - **Worst case:** is achieved by constant explainer that gives same
      importance to each feature that is equal to 1/N where N is the number
      of features will obtain best gini index and hence worst sparsity
      - **Best case:** is when explainer is constant and gives one feature
      maximum value and others zero, which is the most unequal distribution
      and is the sparsest explanation that can be given

    """

    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "sparsity"
        self.direction = "up"

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        ginis = []

        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch["item"]

            explanation_batch = expl.predict(item, self._model, **expl_kwargs)
            explanation_batch = minmax_normalize(explanation_batch)

            ginis += batch_gini(explanation_batch)

        return np.nanmean(ginis)
