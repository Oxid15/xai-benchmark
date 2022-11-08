import numpy as np
from tqdm import tqdm
from ...base import Case, Explainer, Dataset, Model
from ...utils import rmse


class CorrectnessCase(Case):
    def __init__(self, ds: Dataset, model: Model, noisy_model:Model, **kwargs):
        super().__init__(ds, model, **kwargs)
        self.noisy_model = noisy_model

    def evaluate(self, expl: Explainer, expl_kwargs=None, expl_noisy_kwargs=None) -> None:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expl_noisy_kwargs is None:
            expl_noisy_kwargs = {}

        diffs_expl = np.zeros(len(self.ds))

        for i, item in tqdm(enumerate(self.ds)):
            e = expl.predict(item, self.model, **expl_kwargs)
            ne = expl.predict(item, self.noisy_model, **expl_noisy_kwargs)

            diffs_expl[i] = rmse(e, ne)
        
        self.metrics['correctness'] = {
                'parameter_randomization_check': np.nanmean(diffs_expl)
        }
