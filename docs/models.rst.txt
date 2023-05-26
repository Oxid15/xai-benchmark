Models
######
:py:class:`xaib.base.Model`

Models are defined as generic as possible to be able to cover any use-case - from classification and regression to segmentation and language modelling.
Here are the list of ones that are available in the moment.
  
.. note::
    The requirement is that all models should be able to work with batches.

SVC
***
:py:mod:`xaib.evaluation.model_factory.py`

.. tags:: classification, black_box

MLPClassifier
*************
:py:mod:`xaib.evaluation.model_factory.py`

.. tags:: classification, black_box

KNeighborsClassifier
********************
:py:mod:`xaib.evaluation.model_factory.py`

.. tags:: classification, white_box
