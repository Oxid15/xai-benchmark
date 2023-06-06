Explainers
##########
:py:class:`xaib.base.Explainer`

Explainer is very similar to the Model and in fact it inherits the same
superclass, but it has different inference interface – accepting not only data, but
also another Model, it should generate an explanation of some format. The idea
here is similar to the model's – to be able to incorporate in the benchmark as many
types as possible the interface made very broad. One can have feature importance,
example selection/synthesis or trees under one umbrella-class which allows
common interface for handling different entities similarly.

.. toctree::
    :maxdepth: 2

    explainers/feature_importance
    explainers/example_selection
