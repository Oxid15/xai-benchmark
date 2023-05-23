Datasets
########
:py:class:`xaib.base.Dataset`

The defining property of XAIB is that it brings XAI evaluation closer to practice - allowing evaluation of
methods on any custom datasets.
Here are the description of datasets that are incorporated into XAIB experiments.

breast_cancer
*************
:py:class:`xaib.datasets.sk_dataset.SkDataset`

.. tags:: sk_dataset, toy_dataset, classification

`breast_cancer` is a collection of 569 records of different
properties of cell nuclei from digitized images of a fine needle aspirate (FNA) of a
breast mass. Each record contains 30 numeric features and the class. Classes
represent the state of nuclei – malignant or benign. Distribution is imbalanced –
there are 212 malignant and 357 benign samples.
  
Source: `sklearn <https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset>`_

digits
******
:py:class:`xaib.datasets.sk_dataset.SkDataset`

.. tags:: sk_dataset, toy_dataset, classification

Digits dataset is a collection of 1797 images of the hand-written digits 8x8
pixels each where each pixel can have intensity values from 0 to 16. Number of
classes is 10 with each number corresponding to a digit. This dataset can help to
understand how methods deal with high-dimensional sparse data, since each row is
64 values most of which are zeros being the background pixels.

Source: `sklearn <https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset>`_

wine
****
:py:class:`xaib.datasets.sk_dataset.SkDataset`

.. tags:: sk_dataset, toy_dataset, classification

Wine dataset is another classification dataset in which data obtained from
chemical analysis is intended to be used to recognize the type of wine. The data
has 13 features corresponding to different chemical properties of wines and 3
classes which are somewhat imbalanced.

Source: `sklearn <https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset>`_

iris
****
:py:class:`xaib.datasets.sk_dataset.SkDataset`

.. tags:: sk_dataset, toy_dataset, classification

Iris is another classical toy dataset. It contains balanced 150 samples of three
classes where each class corresponds to the type of the iris plant. Each sample
consists of four features namely: sepal length in cm, sepal width in cm, petal length
in cm, petal width in cm. It is well known in the literature and is included mainly
by that reason.

Source: `sklearn <https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset>`_

synthetic_noisy
***************
:py:class:`xaib.datasets.synthetic_dataset.SyntheticDataset`

.. tags:: toy_dataset, classification

Source: `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification>`_

synthetic
*********
:py:class:`xaib.datasets.synthetic_dataset.SyntheticDataset`

.. tags:: toy_dataset, classification

Source: `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification>`_
