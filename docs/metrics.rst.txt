Metrics
=======
:py:class:`xaib.base.Metric`

Metrics are the way to numerically measure some desired or undesired
properties of an object. Common example is performance metrics in machine
learning. They are defined to show quality of the solution. When computed on
various validation sets they can provide engineers with information on what to
expect from the model in real-life settings.
  
In XAIB each metric should correspond to some :doc:`Case <cases>`

To further read on metrics in this benchmark, please proceed to the following pages.

To see how different XAI methods perform on the different datasets, please proceed to the :doc:`results page. <./results>`

.. toctree::
    :maxdepth: 2

    metrics/feature_importance
    metrics/example_selection
