from typing import Dict, Any

import numpy as np
from sklearn.datasets import make_classification
from cascade import data as cdd
from cascade import models as cdm


class MakeClassificationDataset(cdd.Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.X, self.y = make_classification(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {'item': self.X[index], 'label': self.y[index]}


class NoiseApplier(cdd.Modifier):
    def __init__(self, dataset, multiplier: float = 1., *args, **kwargs) -> None:
        super().__init__(dataset, *args, **kwargs)

        self._multiplier = multiplier

    def __getitem__(self, index):
        item = self._dataset.__getitem__(index)
        item['item'] = item['item'] \
            + np.random.random(item['item'].shape) * self._multiplier
        return item


class RandomBinBaseline(cdm.BasicModel):
    def __init__(self, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)

    def predict(self, x):
        return np.array([np.random.choice((0, 1)) for _ in range(len(x))])

    def predict_proba(self, x):
        proba = np.array([np.random.random() for _ in range(len(x))])
        return np.stack((proba, 1.0 - proba), axis=1)


def case(explainers, batch_size=1):
    def wrapper(case_init):
        def wrap_case():
            c = case_init()

            repo = cdm.ModelRepo('repo')
            line = repo.add_line('correctness')

            for name in explainers:
                c.evaluate(name, explainers[name], batch_size=batch_size)
                line.save(c, only_meta=True)

        return wrap_case

    return wrapper
