Add dataset
===========

New datasets may extend our understanding of how different explainers
behave in context of different domains and tasks.  
To add your dataset, you should provide a Wrapper, which will
download or access prepared data from disk.

Create data wrapper
-------------------

First you need to create a wrapper with required interface and fields

.. code-block:: python
    
    import numpy as np
    from xaib import Dataset


    class NewDataset(Dataset):
        """
        Here the documentation on data should be filled
        """
        def __init__(self, split, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            # It is important to set the name
            # the name will be used to identify a dataset
            self.name = 'new_dataset'

            # While creating you can download and cache data,
            # define splits, etc
            if split == 'train':
                self._data = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
                self._labels = [0, 1, 0]
            elif split == 'test':
                self._data = [(9, 10, 11), (12, 13, 14)]
                self._labels = [1, 0]

        def __getitem__(self, index):
            # This form of returning items is required - Dict[str, np.ndarray[Any]]
            return {
                'item': np.asarray(self._data[index]),
                'label': np.asarray(self._labels[index])
            }

        def __len__(self):
            return len(self._data)

Test new dataset
----------------
Before adding your implementation directly into source code, it would be useful to
test how it will work with standard XAIB setup

.. code-block:: python

    from xaib.evaluation import DatasetFactory, ModelFactory
    from xaib.evaluation.feature_importance import ExplainerFactory, ExperimentFactory
    from xaib.evaluation.utils import visualize_results


    # Here you create your data
    train_ds, test_ds = NewDataset('train'), NewDataset('test')

    # And then pass it further
    model = ModelFactory(train_ds, test_ds).get('svm')

    explainers = ExplainerFactory(train_ds, model, labels=[0, 1]).get('all')

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

Integrate new dataset
---------------------
Finally you can integrate your dataset into the source code.  
To do that you need to add it into `xaib.datasets` module
and then make a constructor for the Factory.

.. code-block:: python

    # xaib/evaluation/dataset_factory.py
    # ...
    from xaib.datasets import NewDataset
    # ...

    # Create a constructor - function that will build your dataset
    # it should provide all defaults needed
    def new_dataset():
        return NewDataset('train'), NewDataset('test')


    class DatasetFactory(Factory):
        def __init__(self) -> None:
            # ...
            # add it to the factory
            self._constructors['new_dataset'] = lambda: new_dataset()
