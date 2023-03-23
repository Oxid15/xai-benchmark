from typing import Dict, Union, Any
import numpy as np
from tqdm import tqdm

from ...base import Dataset, Model, Metric, Explainer
from ...utils import entropy, minmax_normalize, SimpleDataloader


class CovariateRegularity(Metric):
    """
    CovariateComplexity Measures how complex explanation features are, their
    consistency. If explanation features are noisy, then
    they are harder to remember.
    """
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = 'covariate_regularity'
        self.direction = 'down'

    def compute(
            self,
            expl: Explainer,
            batch_size: int = 1,
            expl_kwargs: Union[Dict[Any, Any], None] = None,
    ):
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

        return np.nanmean(entropies_of_features)
