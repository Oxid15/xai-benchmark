from typing import Dict, Any
import os

import numpy as np
from cascade import data as cdd
from cascade import models as cdm

from cascade.meta import MetaViewer
import pandas as pd
from plotly import graph_objects as go


class WrapperModel(cdm.ModelModifier):
    def __init__(self, model, name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(model, *args, **kwargs)
        self.name = name

    def save(self, filepath: str, *args: Any, **kwargs: Any) -> None:
        return self._model.save(filepath, *args, **kwargs)

    def load(self, filepath: str, *args: Any, **kwargs: Any) -> None:
        return self._model.load(filepath, *args, **kwargs)

class WrapperDataset(cdd.Modifier):
    def __init__(self, ds, name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset=ds, *args, **kwargs)
        self.name = name


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


class KNeighborsTransformer:
    def __init__(self, dataset_length) -> None:
        self._dataset_length = dataset_length

    def kneighbors(self, x, n_neighbors):
        return None, np.asarray([[np.random.randint(0, self._dataset_length)]
            for _ in range(n_neighbors)])


class RandomNeighborsBaseline(cdm.BasicModel):
    def __init__(self, dataset_length, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)
        self._pipeline = [KNeighborsTransformer(dataset_length)]




def experiment(root, explainers, *args, batch_size=1, **kwargs):
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


def visualize_results(path, output_path=None, title=None):
    m = MetaViewer(path, filt={'type': 'model'})

    data = []
    for p in m:
        for metric_name in p[0]['metrics']:
            data.append(
                {
                    'name': p[0]['params']['metric_params'][metric_name]['name'],
                    'case': p[0]['params']['case'],
                    'dataset': p[0]['params']['metric_params'][metric_name]['dataset'],
                    'model': p[0]['params']['metric_params'][metric_name]['model'],
                    'metric': metric_name,
                    'direction': p[0]['params']['metric_params'][metric_name]['direction'],
                    'value': p[0]['metrics'][metric_name]
                }
            )

    df = pd.DataFrame(data)
    df = pd.pivot_table(df, values='value', columns=['name'], index=['dataset', 'model', 'case', 'metric', 'direction'])

    df = df.reset_index()
    df.loc[df['dataset'].duplicated(), 'dataset'] = ''
    df.loc[df['model'].duplicated(), 'model'] = ''
    df.loc[df['case'].duplicated(), 'case'] = ''
    df.loc[df['metric'].duplicated(), 'metric'] = ''

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                align='left'))
        ]
    )
    fig.update_layout(title=title)

    if output_path is not None:
        fig.write_image(output_path)

    return fig