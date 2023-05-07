import os
from typing import Any, Union, Literal, List

import numpy as np
import pandas as pd
from cascade import data as cdd
from cascade import models as cdm
from cascade.meta import MetaViewer
from plotly import express as px
from plotly import graph_objects as go
from scipy.special import softmax
from cascade.base import Traceable, PipeMeta
from xaib.base import Factory


class Setup(Traceable):
    def __init__(
        self,
        dataset_factory: Factory,
        model_factory: Factory,
        explainer_factory: Factory,
        case_factory: Factory,
        datasets: Union[List[str], Literal["all"]] = "all",
        models: Union[List[str], Literal["all"]] = "all",
        explainers: Union[List[str], Literal["all"]] = "all",
        cases: Union[List[str], Literal["all"]] = "all",
        datasets_except: Union[List[str], None] = None,
        models_except: Union[List[str], None] = None,
        explainers_except: Union[List[str], None] = None,
        cases_except: Union[List[str], None] = None,
    ) -> None:
        super().__init__()

        self._cfg = {
            "datasets": {
                "input": datasets,
                "factory": dataset_factory,
                "output": None,
                "except": datasets_except,
            },
            "models": {
                "input": models,
                "factory": model_factory,
                "output": None,
                "except": models_except,
            },
            "explainers": {
                "input": explainers,
                "factory": explainer_factory,
                "output": None,
                "except": explainers_except,
            },
            "cases": {
                "input": cases,
                "factory": case_factory,
                "output": None,
                "except": cases_except,
            },
        }

        for name in self._cfg:
            output = self._cfg[name]["input"]
            if output == "all":
                output = self._cfg[name]["factory"].get_names()

            if self._cfg[name]["except"]:
                output = list(set(output) - set(self._cfg[name]["except"]))

            self._cfg[name]["output"] = output

        self.datasets = self._cfg["datasets"]["output"]
        self.models = self._cfg["models"]["output"]
        self.explainers = self._cfg["explainers"]["output"]
        self.cases = self._cfg["cases"]["output"]

    def __repr__(self) -> str:
        for name in self._cfg:
            if self._cfg[name]["input"] == "all":
                rep = "all"
            else:
                rep = ", ".join(self._cfg[name]["output"])

            if self._cfg[name]["except"]:
                rep += f" except {', '.join(self._cfg[name]['except'])}"
            self._cfg[name]["repr"] = rep

        return (
            f"Experiment setup with {self._cfg['datasets']['repr']} Datasets,"
            f" {self._cfg['models']['repr']} Models,"
            f" {self._cfg['explainers']['repr']} Explainers,"
            f" {self._cfg['cases']['repr']} Cases"
        )

    def get_meta(self) -> PipeMeta:
        meta = super().get_meta()
        meta[0]["datasets"] = self.datasets
        meta[0]["models"] = self.models
        meta[0]["explainers"] = self.explainers
        meta[0]["cases"] = self.cases
        return meta


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


class RandomBaseline(cdm.BasicModel):
    def __init__(self, labels, *args, meta_prefix=None, **kwargs) -> None:
        super().__init__(*args, meta_prefix=meta_prefix, **kwargs)
        self._labels = labels

    def predict(self, x):
        return np.array([np.random.choice(self._labels) for _ in range(len(x))])

    def predict_proba(self, x):
        proba = np.random.random((len(x), len(self._labels)))
        return softmax(proba, axis=1)


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
    for col in df:
        if df[col].dtype == np.float64:
            df[col] = df[col].apply(lambda x: f"{x:0.2f}")

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

    df = df.groupby(["name", "metric", "direction"], as_index=False).agg("mean")

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
                width=0.25,
            )
            for metric in metrics
        ]
    )

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
        color="dataset",
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

    for col in ["dataset", "model", "case", "metric"]:
        prev_val = ""
        for i in range(len(pv[col])):
            val = pv.loc[i][col]
            if val == prev_val:
                pv.loc[i, col] = ""
            prev_val = val

    figs = []
    figs.append(
        {
            "name": "table",
            "fig": table(pv),
            "desc": "",
        }
    )

    figs.append(
        {
            "name": "bar",
            "fig": relative_bar(df),
            "desc": "This barchart is simple representation of average scores for each method evaluated.<br>"
            "Since metrics can be directed differenty e.g. in some cases 'the greater the better' or"
            " 'the less the better' on this chart all metrics directed down (the less the better) were "
            "multiplied by -1.<br>This means that on the side below zero better method will be higher, "
            "closer to zero and on the size above zero better method will be also higher as usual.",
        }
    )

    metrics = df.metric.unique()
    for name in metrics:
        figs.append({"name": f"scatter_{name}", "fig": scatter(df, name), "desc": ""})

    for fig in figs:
        if output_dir is not None:
            fig["fig"].write_image(os.path.join(output_dir, fig["name"] + ".png"))

    return figs
