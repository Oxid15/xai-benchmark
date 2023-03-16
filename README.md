# xai-benchmark

Open and extensible benchmark for XAI methods

## Description

XAIB is an open benchmark that provides a way to compare different XAI methods using broad set of metrics that aimed to measure different aspects of interpretability

## Installation

```bash
pip3 install xai-benchmark
```

Remember to create virtual environment if you need one.  

After the installation you can verify the package by printing out its version:

```python
import xaib
print(xaib.__version__)
```

To use all explainers you should also install `explainers_requirements.txt` which can be done
directly

```bash
pip3 install -r https://raw.githubusercontent.com/oxid15/xai-benchmark/master/explainers_requirements.txt
```

## Results

`Coming soon`

## How to use

### Reproduce

`Coming soon`

### Try method

```python
from xaib.explainers.feature_importance.lime_explainer import LimeExplainer
from xaib.evaluation import DatasetFactory, ModelFactory

# Get the dataset and train the model
train_ds, test_ds = DatasetFactory().get('synthetic')
model = ModelFactory(train_ds, test_ds).get('svm')

explainer = LimeExplainer(train_ds, labels=[0, 1])

# Obtain batch from dataset
sample = [test_ds[i]['item'] for i in range(10)]

# Obtain explanations
explanations = explainer.predict(sample, model)

print(explanations)
```

### Evaluate method

```python
from xaib.evaluation import DatasetFactory, ModelFactory
from xaib.evaluation.example_selection import ExplainerFactory, ExperimentFactory
from xaib.evaluation.utils import visualize_results


train_ds, test_ds = DatasetFactory().get('synthetic')
model = ModelFactory(train_ds, test_ds).get('knn')

explainer = ExplainerFactory(train_ds, model).get('knn')

experiment_factory = ExperimentFactory(
    repo_path='results',
    explainers={'knn': explainer},
    test_ds=test_ds,
    model=model,
    batch_size=10
)

# Run all experiments on chosen method
experiments = experiment_factory.get('all')
for name in experiments:
    experiments[name]()

# Save plot to the results folder
visualize_results('results', 'results/results.png')
```

## How to contribute

### Add dataset

#### Create wrapper

```python
import numpy as np
from xaib.base import Dataset


class CoolNewDataset(Dataset):
    """
    Here the documentation on data should be filled
    """
    def __init__(self, split, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # It is important to set the name
        # the name will be used to identify a dataset
        self.name = 'cool_new_dataset'

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
```

#### Test new dataset

```python

from xaib.evaluation import DatasetFactory, ModelFactory
from xaib.evaluation.feature_importance import ExplainerFactory, ExperimentFactory
from xaib.evaluation.utils import visualize_results


train_ds, test_ds = CoolNewDataset('train'), CoolNewDataset('test')
model = ModelFactory(train_ds, test_ds).get('svm')

explainers = {'const': ExplainerFactory(train_ds, model, labels=[0, 1]).get('const')}

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
```

#### Integrate into package

```python
# xaib/evaluation/dataset_factory.py
# ...

# Create a constructor - function that will build your dataset
def my_cool_dataset():
    return CoolNewDataset('train'), CoolNewDataset('test')


class DatasetFactory(Factory):
    def __init__(self) -> None:
        # ...
        # add it to the factory
        self._constructors['cool_new_dataset'] = lambda: my_cool_dataset()
```

### Add model

`Docs coming soon`

### Add explainer

`Docs coming soon`

### Add metric

`Docs coming soon`
