from typing import Dict, Iterable

import numpy as np
from tqdm import tqdm

from ...base import Case, Explainer, Model, Dataset
from ...utils import batch_rmse, SimpleDataloader


class CoherenceCase(Case):
    def __init__(self, ds: Dataset, model: Model) -> None:
        super().__init__(ds, model)

    def evaluate(
            self,
            name: str,
            expl: Explainer,
            expls: Iterable[Explainer],
            batch_size: int = 1,
            expl_kwargs: Dict = None,
            expls_kwargs: Iterable[Dict] = None
        ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expls_kwargs is None:
            expls_kwargs = [{} for _ in range(len(expls))]

        diffs = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch['item']

            e = expl.predict(item, self._model, **expl_kwargs)
            other_e = [other_expl.predict(item, self._model, **other_kwargs) 
                for other_expl, other_kwargs in zip(expls, expls_kwargs)]

            for oe in other_e:
                diffs += batch_rmse(e, oe)

        self.params['name'] = name
        self.metrics['other_disagreement'] = np.nanmean(diffs)
