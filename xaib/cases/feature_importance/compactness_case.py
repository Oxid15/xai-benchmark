from typing import Dict, Union, Any

import numpy as np
from tqdm import tqdm

from ...base import Case, Explainer
from ...utils import batch_gini, minmax_normalize, SimpleDataloader


class CompactnessCase(Case):
    """
    Compactness measures how compact
    representations are. Explanations are more
    understandable if they are short.
    """
    def evaluate(
        self,
        name: str,
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

        self.params['name'] = name
        self.metrics['sparsity'] = np.nanmean(ginis)
