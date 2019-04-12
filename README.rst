.. include:: <isonum.txt>
.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/AIWerkstatt/koho.svg?branch=master
.. _Travis: https://travis-ci.org/AIWerkstatt/koho

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/7mpfa6oulfasp99w/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/AIWerkstatt/koho

.. |Codecov| image:: https://codecov.io/gh/AIWerkstatt/koho/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/AIWerkstatt/koho

.. |CircleCI| image:: https://circleci.com/gh/AIWerkstatt/koho.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/AIWerkstatt/koho

.. |ReadTheDocs| image:: https://readthedocs.org/projects/koho/badge/?version=latest
.. _ReadTheDocs: https://koho.readthedocs.io/en/latest/

koho\ |trade|
=============

**koho** (Hawaiian word for 'to estimate') is a **Decision Forest** **C++ library**
with a `scikit-learn`_ compatible **Python interface**.

**Python only implementation!**

- Classification
- Numerical (dense) data
- Class balancing
- Multi-Class
- Single-Output
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

\ |copy| Copyright 2019, `AI Werkstatt`_\ |trade|. All rights reserved.

.. _`scikit-learn`: http://scikit-learn.org
.. _`AI Werkstatt`: http://www.aiwerkstatt.com
