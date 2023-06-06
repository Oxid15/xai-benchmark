Try a method
============
All explanation methods in XAIB have the same input and output interface which allows to
use them easily and compare.  
If you want to run an Explainer and see the results you can do this:

.. code-block:: python

    from xaib.explainers.feature_importance.lime_explainer import LimeExplainer
    from xaib.evaluation import DatasetFactory, ModelFactory

    # Get the dataset and train the model
    train_ds, test_ds = DatasetFactory().get('synthetic')
    model = ModelFactory(train_ds, test_ds).get('svm')


.. code-block:: python

    # You can also get the default one using ExplainerFactory
    explainer = LimeExplainer(train_ds, labels=[0, 1])


.. code-block:: python

    # Obtain batch from dataset
    sample = [test_ds[i]['item'] for i in range(10)]


.. code-block:: python

    # Obtain explanations
    explanations = explainer.predict(sample, model)

    print(explanations)
