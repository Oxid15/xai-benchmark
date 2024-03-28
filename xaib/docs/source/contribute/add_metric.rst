Add metric
==========
Metrics are ways to numerically assess the quality of explainers and are parts of
Cases

Create metric
-------------
First you need to create a Metric object - which will accept and explainer and data
and return some value

.. note::

    You should define `self.name` and `self.direction` of the metric in order for
    it to be displayed correctly in results.  
    `self.name` is the short name of what is measured and `self.direction`
    denotes what values are considered better - greater or less.

.. code-block:: python

    from xaib import Metric

    class NewMetric(Metric):
        def __init__(self, ds, model *args, **kwargs):
            super().__init__(ds, model, *args, **kwargs)

            self.name = 'new_metric'
            self.direction = 'down'

        def compute(self, explainer, *args, batch_size=1, expl_kwargs=None, **kwargs):
            if expl_kwargs is None:
                expl_kwargs = {}

            dl = SimpleDataloader(self._ds, batch_size=batch_size)
            for batch in tqdm(dl):
                # get explanations
                ex = expl.predict(batch['item'], self._model, **expl_kwargs)

            # Here compute and return your metric

            return np.random.rand()

Test new metric
---------------
Before adding your implementation directly into source code, it would be useful to
test how it will work with standard XAIB setup  
  
Since metrics are more low-level objects, they need special treatment
when tested. Basically you need to create metric and append it to the existing
Case of choice.

.. code-block:: python

    from xaib.evaluation import DatasetFactory, ModelFactory
    from xaib.evaluation.feature_importance import ExplainerFactory
    from xaib.evaluation.utils import visualize_results, experiment


    train_ds, test_ds = DatasetFactory().get('synthetic')
    model = ModelFactory(train_ds, test_ds).get('svm')

    explainers = ExplainerFactory(train_ds, model, labels=[0, 1]).get('all')

    metric = NewMetric(test_ds, model)


    @experiment(
        'results',
        explainers=explainers,
        metrics_kwargs={
            'other_disagreement': dict(expls=list(explainers.values()))
        }
    )
    def coherence():
        case = CoherenceCase(test_ds, model)
        case.add_metric('new_metric', metric)
        return case

    coherence()

    visualize_results('results', 'results/results.png')

Integrate new metric
--------------------

.. code-block:: python

    # xaib/cases/feature_importance/coherence_case.py
    # ...
    from ...metrics.feature_importance import NewMetric


    class CoherenceCase(Case):
        def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
            super().__init__(ds, model, *args, **kwargs)
            # ...

            self.metrics(NewMetric(ds, model))
