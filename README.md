# xai-benchmark
Public and extensible benchmark for XAI methods

## Description
XAIB is an open benchmark that provides a way to compare different XAI methods using broad set of metrics that aimed to measure different aspects of interpretability

## Installation
```
git clone https://github.com/Oxid15/xai-benchmark.git
cd xai-benchmark
pip install .
```
Then you can verify your installation by importing:
```python
import xaib
print(xaib.__version__)
```

## Usage
To test your own XAI method, you need to wrap it into `Explainer` interface (for examples of doing that see `explainers` folder) and pass it into `explainers` dict of evaluation notebook. For example for feature importance methods go to `evaluation/feature_importance/feature_importance.ipynb`.

## Results
Metric values of all tested algorithms and baselines can be found in `evaluation` folder in the folder corresponding to the type of method for example for feature importance methods you can go to `evaluation/feature_importance/feature_importance.ipynb` and found metric values there.
