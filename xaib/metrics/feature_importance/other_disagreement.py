from typing import Dict, List, Any, Union

import numpy as np
from tqdm import tqdm

from ...base import Metric, Explainer
from ...utils import batch_rmse, minmax_normalize, SimpleDataloader


class OtherDisagreement(Metric):
    """
    Coherence measures how method
    complies with domain knowledge, ground-truth
    or other methods
    """
    def evaluate(
        self,
        name: str,
        expl: Explainer,
        expls: List[Explainer],
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
        expls_kwargs: Union[List[Dict[Any, Any]], None] = None
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expls_kwargs is None:
            expls_kwargs = [{} for _ in range(len(expls))]

        diffs = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch['item']

            e = expl.predict(item, self._model, **expl_kwargs)
            e = minmax_normalize(e)
            other_e = [minmax_normalize(other_expl.predict(item, self._model, **other_kwargs))
                       for other_expl, other_kwargs in zip(expls, expls_kwargs)]

            for oe in other_e:
                diffs += batch_rmse(e, oe)

        self.params['name'] = name
        self.metrics['other_disagreement'] = np.nanmean(diffs)
