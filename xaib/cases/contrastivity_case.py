import numpy as np

from xaib.utils import rmse
from ..base import Dataset, Model, Case, Explainer


class ContrastivityCase(Case):
    def __init__(
        self, ds: Dataset, model: Model, *args, **kwargs) -> None:

        super().__init__(ds, model, *args, **kwargs)

    def _compare(self, coords, explanations):
        diffs = []

        for u in range(len(coords)):
            for i in range(len(coords[u])):
                e = explanations[u][i]

                # Calculate the RMSE between this explanation and
                # others of other classes
                d = [rmse(e, explanations[x][i]) for x in range(len(coords)) if x != u]
                d = np.mean(d)
                diffs.append(d)
        return np.mean(diffs)

    def evaluate(self, expl: Explainer) -> None:
        labels = np.asarray([label for _, label in self.ds])
        unique, counts = np.unique(labels, return_counts=True)
        min_len = np.min(counts)

        unique_coords = [np.argwhere(labels == ul)[:min_len] for ul in unique]

        explanations = []
        for u in range(len(unique)):
            for i in range(len(unique_coords[u])):
                e = expl.predict(self.ds[unique_coords[u][i][0]], self.model)
                explanations.append(e)

        self.metrics['contrastivity'] = self._compare(unique_coords, explanations)
