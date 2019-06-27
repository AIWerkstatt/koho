.. -*- mode: rst -*-

|Travis|_ |Codecov|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/AIWerkstatt/koho.svg?branch=master
.. _Travis: https://travis-ci.org/AIWerkstatt/koho

.. |Codecov| image:: https://codecov.io/gh/AIWerkstatt/koho/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/AIWerkstatt/koho

.. |ReadTheDocs| image:: https://readthedocs.org/projects/koho/badge/?version=latest
.. _ReadTheDocs: https://koho.readthedocs.io/en/latest/

koho (TM)
=========

**koho** (Hawaiian word for 'to estimate') is a **Decision Forest** **C++ library**
with a `scikit-learn`_ compatible **Python interface**.

- Classification
- Numerical (dense) data
- Missing values (Not Missing At Random (NMAR))
- Class balancing
- Multi-Class
- Multi-Output (single model)
- Build order: depth first
- Impurity criteria: gini
- n Decision Trees with soft voting
- Split a. features: best over k (incl. all) random features
- Split b. thresholds: 1 random or all thresholds
- Stop criteria: max depth, (pure, no improvement)
- Bagging (Bootstrap AGGregatING) with out-of-bag estimates
- Important Features
- Export Graph

`ReadTheDocs`_

`New BSD License <LICENSE>`_

**Change Log:**
1.1.0 Multi-Output (single model)
1.0.0 Missing Values (NMAR) : Python, Cython(bindings), C++
0.0.2 Criterion implemented in Cython
0.0.1 Classification : Python only

Copyright 2019, `AI Werkstatt (TM)`_. All rights reserved.

.. _`scikit-learn`: http://scikit-learn.org
.. _`AI Werkstatt (TM)`: http://www.aiwerkstatt.com
