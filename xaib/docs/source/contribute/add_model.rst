Add model
=========
New models and model classes provide information on how good explainers
are in some particular cases.

Create model wrapper
--------------------
First model wrapper should be implemented. It has many
required methods that should be implemented.
For example `fit` and `evaluate` methods are needed
to be able to train the model on different datasets
see specification in `xaib/base` and examples in
`xaib/evaluation/model_factory.py`

.. code-block:: python

    import numpy as np
    from xaib.base import Model


    class NewModel(Model):
        """
        Here the documentation on model should be filled
        """
        def __init__(self, const, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.const = const

            # It is important to set the name
            # the name will be used to identify a model
            self.name = 'new_model'

        def fit(self, x, y):
            pass

        def evaluate(self, x, y):
            pass

        def predict(self, x):
            return np.array([self.const for _ in range(len(x))])

        def save(self, filepath, *args, **kwargs):
            with open(filepath, 'w') as f:
                f.write(str(self.const))

        def load(self, filepath, *args, **kwargs):
            with open(filepath, 'r') as f:
                self.const = float(f.read())
            # load does not return anything - just fills
            # internal state


Test new model
--------------

Before adding your implementation directly into source code, it would be useful to
test how it will work with standard XAIB setup

.. code-block:: python

    from xaib.evaluation import DatasetFactory
    from xaib.evaluation.feature_importance import ExplainerFactory, ExperimentFactory
    from xaib.evaluation.utils import visualize_results


    # Create your model
    model = NewModel(const=1)

    train_ds, test_ds = DatasetFactory().get('synthetic')
    explainers = {'shap': ExplainerFactory(train_ds, model, labels=[0, 1]).get('shap')}

    experiment_factory = ExperimentFactory(
        repo_path='results',
        explainers=explainers,
        test_ds=test_ds,
        model=model, # and put it here
        batch_size=10
    )

    experiments = experiment_factory.get('all')
    for name in experiments:
        experiments[name]()


    visualize_results('results', 'results/results.png')

Integrate new model
-------------------

Finally you can integrate your model into the source code.  
To do that you need to add it into `xaib.models` module
and then make a constructor for the Factory.

.. code-block:: python

    # xaib/evaluation/model_factory.py
    # ...
    from ..models.new_model import NewModel
    # ...

    # Create a constructor - function that will build your model
    def new_model(const):
        return NewModel(const=const)


    class ModelFactory(Factory):
        def __init__(self) -> None:
            
            # ...
            # add it to the factory
            self._constructors['new_model'] = lambda: new_model(const=1)
