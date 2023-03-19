Add explainer
=============

Explainers are the heart of this benchmark, they are being thorougly tested
and the more of them added, the more we know

Create explainer wrapper
------------------------

Explainers wrappers are less demanding than model's which makes them
easier to be implemented

.. code-block:: python

    import numpy as np
    from xaib import Explainer


    class NewExplainer(Explainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # name is essential to put explainer into
            # results table correctly
            self.name = 'new_explainer'

        def predict(self, x, model, *args, **kwargs):
            return np.random.rand(len(x), len(x[0]))

Test new explainer
------------------
Before adding your implementation directly into source code, it would be useful to
test how it will work with standard XAIB setup

.. code-block:: python

    from xaib.evaluation import DatasetFactory, ModelFactory
    from xaib.evaluation.feature_importance import ExperimentFactory
    from xaib.evaluation.utils import visualize_results


    train_ds, test_ds = DatasetFactory().get('synthetic')
    model = ModelFactory(train_ds, test_ds).get('svm')

    explainers = {'new_explainer': NewExplainer()}

    experiment_factory = ExperimentFactory(
        repo_path='results',
        explainers=explainers,
        test_ds=test_ds,
        model=model,
        batch_size=10
    )

    experiments = experiment_factory.get('all')
    for name in experiments:
        experiments[name]()

    visualize_results('results', 'results/results.png')

Integrate new explainer
-----------------------
Finally you can integrate your explainer into the source code.  
To do that you need to add it into `xaib.explainers` module
and then make a constructor for the Factory.

.. code-block:: python

    # xaib/evaluation/feature_importance/explainer_factory.py
    # ...
    from ...explainers.feature_importance.new_explainer import NewExplainer
    # ...

    # Create a constructor - function that will build your explainer
    def new_explainer():
        return NewExplainer()


    class ExplainerFactory(Factory):
        def __init__(self) -> None:
            
            # ...
            # add it to the factory
            self._constructors['new_explainer'] = lambda: new_explainer()
