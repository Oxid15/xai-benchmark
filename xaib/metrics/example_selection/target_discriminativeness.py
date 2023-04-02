from cascade import data as cdd
from tqdm import tqdm

from ...base import Metric
from ...evaluation import ModelFactory
from ...utils import KeyedComposer, SimpleDataloader


class TargetDiscriminativeness(Metric):
    """
    Given true labels and explanations in form of the examples, train another model to discriminate between labels.
    The quality of the model on the examples given describes the quality of the explanations.
    The quality can be measured by any performance metric,
    but it is better to adopt to imbalanced data and use F1-measure for example.

    **The greater the better**

    **Best case:** examples are descriptive of labels so that the model reaches best performance
    **Worst case:** constant or random baseline - giving insufficient information to grasp labels
    """

    def __init__(self, ds, model, *args, **kwargs):
        super().__init__(ds, model, *args, **kwargs)
        self.name = "target_discriminativeness"
        self.direction = "up"

    def compute(self, explainer, batch_size=1, expl_kwargs=None):
        if expl_kwargs is None:
            expl_kwargs = {}

        labels = []
        explanations = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size=batch_size)):
            item, label = batch["item"], batch["label"]

            explanation_batch = explainer.predict(item, self._model, **expl_kwargs)

            explanations += explanation_batch.tolist()
            labels += label.tolist()

        ds = KeyedComposer(
            {"item": cdd.Wrapper(explanations), "label": cdd.Wrapper(labels)}
        )

        expl_train_ds, expl_test_ds = cdd.split(ds, frac=0.8)

        user_model = ModelFactory(expl_train_ds, expl_test_ds).get("svm")

        return user_model.metrics["f1"]
