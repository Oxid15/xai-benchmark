from typing import Dict, Union, Any

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Model, Metric, Explainer
from ...utils import batch_gini, minmax_normalize, SimpleDataloader


class Sparsity(Metric):
    """
    Compactness measures how compact
    representations are. Explanations are more
    understandable if they are short.
    """
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = 'sparsity'

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        ginis = []

        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch['item']

            explanation_batch = expl.predict(item, self._model, **expl_kwargs)
            explanation_batch = minmax_normalize(explanation_batch)

            ginis += batch_gini(explanation_batch)

        return np.nanmean(ginis)
