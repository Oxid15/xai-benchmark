import numpy as np
from tqdm import tqdm
from ...base import Case, Explainer, Dataset, Model
from ...utils import rmse, SimpleDataloader


class CorrectnessCase(Case):
    def __init__(self, ds: Dataset, model: Model, noisy_model:Model, **kwargs):
        super().__init__(ds, model, **kwargs)
        self._noisy_model = noisy_model

    def evaluate(self, expl: Explainer, batch_size: int = 1, expl_kwargs=None, expl_noisy_kwargs=None) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expl_noisy_kwargs is None:
            expl_noisy_kwargs = {}

        diffs_expl = []

        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch['item']

            e = expl.predict(item, self._model, **expl_kwargs)
            ne = expl.predict(item, self._noisy_model, **expl_noisy_kwargs)

            diffs_expl += [rmse(rowe, rowne) for (rowe, rowne) in zip(e, ne)]

        self.metrics['correctness'] = {
                'parameter_randomization_check': np.nanmean(diffs_expl)
        }
