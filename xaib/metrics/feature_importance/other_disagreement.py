from typing import Any, Dict, List, Union

import numpy as np
from tqdm import tqdm

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, batch_rmse, minmax_normalize


class OtherDisagreement(Metric):
    """
    Measures how distant explanations on the same data points for
    this particular method from explanations of all others.
    Average RMSE is used as a metric.
    **The less the better**
    """

    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "other_disagreement"
        self.direction = "down"

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expls: Union[List[Explainer], None] = None,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
        expls_kwargs: Union[List[Dict[Any, Any]], None] = None,
    ) -> None:
        # Default initialization in case of empty
        # parameter to not to break default argument order
        # TODO: can revise this later
        if expls is None:
            expls = [expl]
        if expl_kwargs is None:
            expl_kwargs = {}
        if expls_kwargs is None:
            expls_kwargs = [{} for _ in range(len(expls))]

        diffs = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch["item"]

            e = expl.predict(item, self._model, **expl_kwargs)
            e = minmax_normalize(e)
            other_e = [
                minmax_normalize(other_expl.predict(item, self._model, **other_kwargs))
                for other_expl, other_kwargs in zip(expls, expls_kwargs)
            ]

            for oe in other_e:
                diffs += batch_rmse(e, oe)

        return np.nanmean(diffs)
