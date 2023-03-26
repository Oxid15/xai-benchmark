from tqdm import tqdm
from cascade import data as cdd
from ...base import Metric
from ...utils import SimpleDataloader, KeyedComposer
from ...evaluation import ModelFactory


class TargetDiscriminativeness(Metric):
    def __init__(self, ds, model, *args, **kwargs):
        super().__init__(ds, model, *args, **kwargs)
        self.name = 'target_discriminativeness'
        self.direction = 'up'

    def compute(self, explainer, batch_size=1, expl_kwargs=None):
        if expl_kwargs is None:
            expl_kwargs = {}

        labels = []
        explanations = []
        for batch in tqdm(SimpleDataloader(self._ds, batch_size=batch_size)):
            item, label = batch['item'], batch['label']

            explanation_batch = explainer.predict(item, self._model, **expl_kwargs)

            explanations += explanation_batch.tolist()
            labels += label.tolist()

        ds = KeyedComposer(
            {
                'item': cdd.Wrapper(explanations),
                'label': cdd.Wrapper(labels)
            }
        )

        expl_train_ds, expl_test_ds = cdd.split(ds, frac=0.8)

        user_model = ModelFactory(expl_train_ds, expl_test_ds).get('svm')

        return user_model.metrics['f1']
