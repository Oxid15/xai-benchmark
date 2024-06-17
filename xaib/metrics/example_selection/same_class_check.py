from typing import Any

from sklearn.metrics import f1_score
from tqdm import tqdm

from ...base import Metric
from ...utils import ChannelDataloader


class SameClassCheck(Metric):
    """
    Counts how many times the class of the input and the class of the
    example produced are the same and computes a classification F1-score
    based on it
    """

    def __init__(self, ds, model, explainer, *args, **kwargs: Any) -> None:
        super().__init__(
            name="same_class_check",
            direction="up",
            ds=ds,
            model=model,
            explainer=explainer,
            *args,
            **kwargs
        )

    def compute(
        self,
        batch_size: int = 1,
    ) -> float:
        y_model = []
        y_pred = []

        for batch in tqdm(ChannelDataloader(self._ds, batch_size)):
            item = batch["item"]

            explanation_batch = self._explainer.predict(item, self._model, **self._explainer_kwargs)
            explanation_labels = [x["label"] for x in explanation_batch]

            y_model += self._model.predict(item).tolist()
            y_pred += explanation_labels

        self.value = f1_score(y_model, y_pred, average="macro")
        return self.value
