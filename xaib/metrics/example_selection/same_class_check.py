from typing import Any, Dict, Union

from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

from ...base import Dataset, Explainer, Metric, Model
from ...utils import SimpleDataloader, batch_count_eq


class SameClassCheck(Metric):
    def __init__(self, ds: Dataset, model: Model, **kwargs: Any) -> None:
        super().__init__(ds, model, **kwargs)
        self.name = "same_class_check"
        self.direction = "up"

    def compute(
        self,
        expl: Explainer,
        batch_size: int = 1,
        expl_kwargs: Union[Dict[Any, Any], None] = None,
        expl_noisy_kwargs: Union[Dict[Any, Any], None] = None,
    ) -> float:
        if expl_kwargs is None:
            expl_kwargs = {}
        if expl_noisy_kwargs is None:
            expl_noisy_kwargs = {}

        y_true = []
        y_pred = []

        for batch in tqdm(SimpleDataloader(self._ds, batch_size)):
            item = batch["item"]

            explanation_batch = expl.predict(item, self._model, **expl_kwargs)
            explanation_labels = [x["label"] for x in explanation_batch]

            y_true += batch["label"].tolist()
            y_pred += explanation_labels

        return f1_score(y_true, y_pred, average="macro")
