Installation
============

To install latest version of a package, run:

.. code-block:: bash

    pip3 install xai-benchmark

Remember to create virtual environment if you need one.  

After the installation you can verify the package by printing out its version:

.. code-block:: python

    import xaib
    print(xaib.__version__)


To use all explainers you should also install `explainers_requirements.txt` which can be done
directly

.. code-block:: bash

    pip3 install -r https://raw.githubusercontent.com/oxid15/xai-benchmark/master/explainers_requirements.txt
