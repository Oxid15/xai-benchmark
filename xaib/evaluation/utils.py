import os
from typing import Any

import numpy as np
import pandas as pd
from cascade import data as cdd
from cascade import models as cdm
from cascade.meta import MetaViewer
from plotly import express as px
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
    def __init__(self, dataset, multiplier: float = 1.0, *args, **kwargs) -> None:
        super().__init__(dataset, *args, **kwargs)

        self._multiplier = multiplier

    def __getitem__(self, index):
        item = self._dataset.__getitem__(index)
        item["item"] = (
            item["item"] + np.random.random(item["item"].shape) * self._multiplier
        )
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
        return None, np.asarray(
            [[np.random.randint(0, self._dataset_length)] for _ in range(n_neighbors)]
        )


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
                c.evaluate(
                    name, explainers[name], *args, batch_size=batch_size, **kwargs
                )
                line.save(c, only_meta=True)

        return wrap_case

    return wrapper


def table(df):
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(df.columns), align="left"),
                cells=dict(values=[df[col] for col in df.columns], align="left"),
            )
        ]
    )
    fig.update_layout(title="Table with metric values on all setups")

    return fig


def relative_bar(df):
    metrics = df.metric.unique()

    df["viz_value"] = df["value"]
    for metric in metrics:
        df.loc[df["metric"] == metric, "viz_value"] /= df.loc[
            df["metric"] == metric, "value"
        ].max()
    df.loc[df["direction"] == "down", "viz_value"] *= -1

    fig = go.Figure(
        data=[
            go.Bar(
                name=metric,
                x=df.name.unique(),
                y=df.loc[df["metric"] == metric]["viz_value"],
                hovertext=df.loc[df["metric"] == metric]["direction"],
            )
            for metric in df.metric.unique()
        ]
    )
    # Change the bar mode
    fig.update_layout(barmode="relative")

    return fig


def scatter(df, metric):
    direction = df.loc[df.metric == metric]["direction"].iloc[0]

    return px.scatter(
        df.loc[df.metric == metric].sort_values("value", ascending=direction == "up"),
        x="name",
        y="value",
        title=metric.replace("_", " ").capitalize() + ", direction - " + direction,
        hover_data=["dataset", "model"],
    )


def visualize_results(path, output_dir=None):
    m = MetaViewer(path, filt={"type": "model"})

    data = []
    for p in m:
        for metric_name in p[0]["metrics"]:
            data.append(
                {
                    "name": p[0]["params"]["metric_params"][metric_name]["name"],
                    "case": p[0]["params"]["case"],
                    "dataset": p[0]["params"]["metric_params"][metric_name]["dataset"],
                    "model": p[0]["params"]["metric_params"][metric_name]["model"],
                    "metric": metric_name,
                    "direction": p[0]["params"]["metric_params"][metric_name][
                        "direction"
                    ],
                    "value": p[0]["metrics"][metric_name],
                }
            )

    df = pd.DataFrame(data)
    pv = pd.pivot_table(
        df,
        values="value",
        columns=["name"],
        index=["dataset", "model", "case", "metric", "direction"],
    )

    pv = pv.reset_index()
    pv.loc[pv["dataset"].duplicated(), "dataset"] = ""
    pv.loc[pv["model"].duplicated(), "model"] = ""
    pv.loc[pv["case"].duplicated(), "case"] = ""
    pv.loc[pv["metric"].duplicated(), "metric"] = ""

    metrics = pv.metric.unique()

    figs = dict()
    figs["table"] = table(pv)

    for name in metrics:
        figs[f"scatter_{name}"] = scatter(df, name)

    figs["bar"] = relative_bar(df)

    for name in figs:
        if output_dir is not None:
            figs[name].write_image(os.path.join(output_dir, name + ".png"))

    return figs
