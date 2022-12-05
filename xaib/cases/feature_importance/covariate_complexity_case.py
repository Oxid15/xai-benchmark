from typing import Dict, Iterable

import numpy as np
from tqdm import tqdm

from ...base import Case, Explainer, Model, Dataset
from ...utils import batch_entropy, minmax_normalize, SimpleDataloader


class CovariateComplexityCase(Case):
    def evaluate(
            self,
            name: str,
            expl: Explainer,
            batch_size: int = 1,
            expl_kwargs: Dict = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        diffs = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch['item']

            e = expl.predict(item, self._model, **expl_kwargs)
            e = minmax_normalize(e)

            diffs.append(batch_entropy(e))

        self.params['name'] = name
        self.metrics['covariate_regularity'] = np.nanmean(diffs)
