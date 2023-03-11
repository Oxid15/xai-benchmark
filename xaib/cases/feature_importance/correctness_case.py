from typing import Any, Union, Dict
import numpy as np
from tqdm import tqdm
from ...base import Case, Explainer, Dataset, Model
from ...utils import batch_rmse, minmax_normalize, SimpleDataloader


class CorrectnessCase(Case):
    """
    Measures truthfullness of the method
    to the underlying model - whether it is
    sensitive to the changes in model
    """
    def __init__(self, ds: Dataset, model: Model, noisy_model: Model, **kwargs: Any) -> None:
        super().__init__(ds, model, **kwargs)
        self._noisy_model = noisy_model

    def evaluate(
        self,
        name: str,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
        expl_noisy_kwargs: Union[Dict[Any, Any], None] = None
    ) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expl_noisy_kwargs is None:
            expl_noisy_kwargs = {}

        diffs_expl = []

        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch['item']

            explanation_batch = expl.predict(item, self._model, **expl_kwargs)
            noisy_explanation_batch = expl.predict(item, self._noisy_model, **expl_noisy_kwargs)

            explanation_batch = minmax_normalize(explanation_batch)
            noisy_explanation_batch = minmax_normalize(noisy_explanation_batch)

            diffs_expl += batch_rmse(explanation_batch, noisy_explanation_batch)

        self.params['name'] = name
        self.metrics['parameter_randomization_check'] = np.nanmean(diffs_expl)
