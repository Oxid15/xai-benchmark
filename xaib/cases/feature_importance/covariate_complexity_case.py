from typing import Dict, Union, Any
import numpy as np
from tqdm import tqdm

from ...base import Case, Explainer
from ...utils import entropy, minmax_normalize, SimpleDataloader


class CovariateComplexityCase(Case):
    def evaluate(
            self,
            name: str,
            expl: Explainer,
            batch_size: int = 1,
            expl_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}

        explanations = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch['item']

            e = expl.predict(item, self._model, **expl_kwargs)
            e = minmax_normalize(e)
            explanations += e.tolist()

        explanations = np.array(explanations, dtype=float)

        entropies_of_features = []
        for f in range(explanations.shape[1]):
            e = entropy(explanations[:, f])
            entropies_of_features.append(e)

        self.params['name'] = name
        self.metrics['covariate_regularity'] = np.nanmean(entropies_of_features)
