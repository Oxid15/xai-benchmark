Cases
=====
:py:class:`xaib.base.Case`

Case is the core element of XAIB – it provides an interface to
compute metric values using all entities defined above.
  
It defines one abstract method which accepts name of an explainer, and
an explainer to evaluate. It receives a model and a dataset on creation. This is done
intentionally to insist on the particular usage workflow. User first defines all
settings – the model and data, which should be constant for one set of experiments
and then they can evaluate several explainer in the same settings using the same
model and data, which is correct way to compare different methods.
  
Another important aspect of Cases is that each one of them represents one abstract property
that is measured in it. These theoretically defined properties are complex and very
difficult to define numerically in exact way. This means that any metric will by
definition miss the property that we should measure. However, if several ways to
measure one property will be found (and they were) one property can and should
be represented using several exact metrics, which will save users from overfitting
even on the scale of one property.

Each case corresponds to one of the :doc:`Co-12 properties <framework>`.

.. toctree::
    :maxdepth: 2

    cases/feature_importance
    cases/example_selection
