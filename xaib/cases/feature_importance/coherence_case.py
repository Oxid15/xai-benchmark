from typing import Dict, Iterable

import numpy as np
from tqdm import tqdm

from ..base import Case, Explainer, Model, Dataset
from ..utils import rmse


class CoherenceCase(Case):
    def __init__(self, ds: Dataset, model: Model) -> None:
        super().__init__(ds, model)

    def evaluate(
            self, 
            expl: Explainer,
            expls: Iterable[Explainer],
            expl_kwargs: Dict = None,
            expls_kwargs: Iterable[Dict] = None
        ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expls_kwargs is None:
            expls_kwargs = [{} for _ in range(len(expls))]

        diffs = []
        for i in tqdm(range(len(self.ds))):
            item = self.ds[i]

            e = expl.predict(item, self.model, **expl_kwargs)
            other_e = [other_expl.predict(item, self.model, **other_kwargs) 
                for other_expl, other_kwargs in zip(expls, expls_kwargs)]

            d = [rmse(e, oe) for oe in other_e]
            d = np.mean(d)
            diffs.append(d)

        self.metrics['coherence'] = {
            'other_disaggreement': np.nanmean(diffs)
        }
