.. include:: <isonum.txt>

###########
Quick Start
###########

Programming Language:
`Python`_ 3.5, 3.6, 3.7

Python Packages:
`numpy`_, `scipy`_, `scikit-learn`_

.. _`Python`: https://www.python.org
.. _`numpy`: http://www.numpy.org
.. _`scipy`: https://www.scipy.org
.. _`scikit-learn`: http://scikit-learn.org

You just want to use it
=======================

1. conda or pip installation
----------------------------

The ``koho`` package is available through `conda-forge`_ and `PyPi`_ and
can be installed using `conda`_ or `pip`_::

    $: conda install -c conda-forge koho
    $: pip install koho

.. _`pip`: https://pypi.org/project/pip/
.. _`PyPi`: https://pypi.org/
.. _`conda`: http://conda.pydata.org/
.. _`conda-forge`: https://conda-forge.org/

2. Read the Docs
----------------

The `koho documentation`_ is hosted on `Read the Docs`_::

    Firefox: http://koho.readthedocs.io

.. _`Read the Docs`: https://readthedocs.org/
.. _`koho documentation`: http://koho.readthedocs.io/

3. Jupyter notebook
-------------------

The ``koho`` package can be used with `jupyter notebook`_::

    $: conda install jupyter
    $: conda install tornado=5.1.1  # downgrade if connection problems with jupyter
    $: jupyter notebook
    [] from koho.sklearn import DecisionTreeClassifier, DecisionForestClassifier
    [] ...

.. _`jupyter notebook`: http://jupyter.org/

Download
========

1. Download and install repository
----------------------------------

Clone the `koho repository`_ from `GitHub`_::

    $: sudo apt install git
    $: git clone https://github.com/aiwerkstatt/koho.git

.. _`GitHub`: https://github.com/
.. _`koho repository`: https://github.com/AIWerkstatt/koho

Build and install the ``koho`` package::

    koho$: pip install -e .

2. Generate the documentation
-----------------------------

Generate the ``koho`` documentation using `sphinx`_::

    $: conda install sphinx sphinx-gallery sphinx_rtd_theme matplotlib numpydoc pillow
    koho/doc$: make html

View the ``koho`` documentation::

    Firebox: file:///home/<user>/<...>/koho/doc/_build/html/index.html

.. _`sphinx`: http://www.sphinx-doc.org/
.. _`doxygen`: http://www.doxygen.nl/

3. Run tests
------------

Test the ``koho`` package using `pytest`_ with `pytest-cov`_ plugin::

    $: conda install pytest pytest-cov
    koho$: pytest --disable-pytest-warnings -v --cov=koho --pyargs koho

.. _`pytest`: http://doc.pytest.org
.. _`pytest-xdist`: https://pypi.python.org/pypi/pytest-xdist
.. _`pytest-cov`: https://pypi.python.org/pypi/pytest-cov

