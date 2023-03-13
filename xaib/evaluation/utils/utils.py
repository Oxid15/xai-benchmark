from typing import Dict, Any
import os

import numpy as np
from sklearn.datasets import make_classification
from cascade import data as cdd
from cascade import models as cdm

from cascade.meta import MetaViewer
import pandas as pd
from plotly import graph_objects as go


class MakeClassificationDataset(cdd.SizedDataset):
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


def case(root, explainers, *args, batch_size=1, **kwargs):
    def wrapper(case_init):
        def wrap_case():
            c = case_init()

            repo = cdm.ModelRepo(os.path.join(root))
            line = repo.add_line()

            for name in explainers:
                c.evaluate(name, explainers[name], *args, batch_size=batch_size, **kwargs)
                line.save(c, only_meta=True)

        return wrap_case

    return wrapper


def visualize_results(path, output_path) -> None:
    m = MetaViewer(path, filt={'type': 'model'})

    data = []
    for p in m:
        for metric_name in p[0]['metrics']:
            data.append(
                {
                    'name': p[0]['params']['name'],
                    'case': p[0]['params']['case'],
                    'metric': metric_name,
                    'value': p[0]['metrics'][metric_name]
                }
            )

    df = pd.DataFrame(data)
    df = pd.pivot_table(df, values='value', columns=['name'], index=['case', 'metric'])

    df = df.reset_index()
    df.loc[df['case'].duplicated(), 'case'] = ''

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                align='left'))
    ])
    fig.write_image(output_path)
