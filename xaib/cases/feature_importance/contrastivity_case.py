import numpy as np

from xaib.utils import batch_rmse
from ...base import Case, Explainer


class ContrastivityCase(Case):
    def _compare(self, coords, explanations):
        diffs = []

        for u in range(len(coords)):
            for i in range(len(coords[u])):
                e = explanations[u][i]

                # Calculate the RMSE between this explanation and
                # others of other classes
                d = []
                for x in range(len(coords)):
                    if x != u:
                        d += batch_rmse(e, explanations[x][i])
                diffs += d
        return diffs

    def evaluate(self, expl: Explainer) -> None:
        # TODO: refactor the process
        labels = np.asarray([item['label'] for item in self._ds])
        unique, counts = np.unique(labels, return_counts=True)
        min_len = np.min(counts)

        # List of coordinates in ds for every unique label
        coords = [np.argwhere(labels == ul)[:min_len] for ul in unique]

        explanations = [[] for _ in range(len(coords))]
        for u in range(len(coords)):
            for i in range(len(coords[u])):
                item = self._ds[coords[u][i][0]]
                e = expl.predict([item['item']], self._model)
                explanations[u].append(e)

        diffs = self._compare(coords, explanations)
        self.metrics['contrastivity'] = {
            'label_difference': np.nanmean(diffs)
        }
