Evaluate a method
=================

To evaluate some existing method on all
cases you should create a default setup and run it

.. code-block:: python

    from xaib.evaluation import DatasetFactory, ModelFactory
    from xaib.evaluation.example_selection import ExplainerFactory, ExperimentFactory
    from xaib.evaluation.utils import visualize_results


    train_ds, test_ds = DatasetFactory().get('synthetic')
    model = ModelFactory(train_ds, test_ds).get('knn')

    explainer = ExplainerFactory(train_ds, model).get('knn')

    # Run all experiments on chosen method
    experiment_factory = ExperimentFactory(
        repo_path='results',
        explainers={'knn': explainer},
        test_ds=test_ds,
        model=model,
        batch_size=10
    )

    experiments = experiment_factory.get('all')
    for name in experiments:
        experiments[name]()

    # Save plot to the results folder
    visualize_results('results', 'results/results.png')
