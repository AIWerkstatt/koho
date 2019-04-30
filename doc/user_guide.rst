.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

**koho** (Hawaiian word for 'to estimate') is a **Decision Forest** **C++ library**
with a `scikit-learn`_ compatible **Python interface**.

**Python implementation with Criterion implemented in Cython!**

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

.. _`scikit-learn`: http://scikit-learn.org

Python
======

We provide a `scikit-learn`_ compatible **Python interface**.

.. currentmodule:: koho.sklearn

Classification
--------------

The koho library provides the following classifiers:

:class:`DecisionTreeClassifier`
:class:`DecisionForestClassifier`

    >>> from koho.sklearn import DecisionTreeClassifier, DecisionForestClassifier
    >>> clf = DecisionForestClassifier(random_state=0)

Decision Tree: ``max_features=None`` and ``max_thresholds=None``,
Random Tree: ``max_features<n_features`` and ``max_thresholds=None``,
Extreme Randomized Trees (ET): ``max_thresholds=1``,
Totally Randomized Trees: ``max_features=1`` and ``max_thresholds=1``
very similar to Perfect Random Trees (PERT).

We use the `iris`_ dataset provided by `scikit-learn`_ for illustration purposes.

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target

.. _`iris`: https://en.wikipedia.org/wiki/Iris_flower_data_set

**Training**

    >>> clf.fit(X, y)
    DecisionForestClassifier(bootstrap=False, class_balance='balanced',
             max_depth=3, max_features='auto', max_thresholds=None,
             n_estimators=100, n_jobs=None, oob_score=False,
             random_state=0)

Feature Importances

    >>> feature_importances = clf.feature_importances_
    >>> print(feature_importances)
    [0.06882834 0.00890739 0.41872655 0.50353772]

Visualize Trees

Export a tree in `graphviz`_ format and visualize it using `graphviz`_::

$: conda install python-graphviz

    >>> import graphviz
    >>> tree_idx = 0
    >>> dot_data = clf.estimators_[tree_idx].export_graphviz(
    ...         feature_names=iris.feature_names,
    ...         class_names=iris.target_names,
    ...         rotate=True)
    >>> graph = graphviz.Source(dot_data)
    >>> graph

.. only:: html

    .. figure:: ./_static/iris.png
       :align: center

Convert the tree to different file formats (e.g. pdf, png)::

    >>> graph.render("iris", format='pdf')
    iris.pdf

Export a tree in a compact textual format::

    >>> t = clf.estimators_[tree_idx].export_text()
    >>> print(t)
    0 X[2]<=2.45 [50, 50, 50]; 0->1; 0->2; 1 [50, 0, 0]; 2 X[3]<=1.75 [0, 50, 50]; 2->3; 2->6; 3 X[2]<=4.95 [0, 49, 5]; 3->4; 3->5; 4 [0, 47, 1]; 5 [0, 2, 4]; 6 X[3]<=1.85 [0, 1, 45]; 6->7; 6->8; 7 [0, 1, 11]; 8 [0, 0, 34];

.. _`graphviz`: http://www.graphviz.org/

Persistence

    >>> import pickle
    >>> with open("clf.pkl", "wb") as f:
    ...     pickle.dump(clf, f)
    >>> with open("clf.pkl", "rb") as f:
    ...     clf2 = pickle.load(f)

**Classification**

    >>> c = clf2.predict(X)
    >>> print(c)
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

    >>> cp = clf2.predict_proba(X)
    >>> print(cp)
    [[1.         0.         0.        ]
     [1.         0.         0.        ]
     [1.         0.         0.        ]
     ...
     [0.         0.01635202 0.98364798]
     [0.         0.01635202 0.98364798]
     [0.         0.06594061 0.93405939]]

**Testing**

    >>> score = clf2.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.966667

**scikit-learn's ecosystem**

Pipeline

    >>> from sklearn.pipeline import make_pipeline
    >>> pipe = make_pipeline(DecisionForestClassifier(random_state=0))
    >>> pipe.fit(X, y)
    >>> pipe.predict(X)
    >>> pipe.predict_proba(X)
    >>> score = pipe.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.966667

Grid Search

    >>> from sklearn.model_selection import GridSearchCV
    >>> parameters = [{'n_estimators': [10, 20],
    ...                'bootstrap': [False, True],
    ...                'max_features': [None, 1],
    ...                'max_thresholds': [None, 1]}]
    >>> grid_search = GridSearchCV(DecisionForestClassifier(random_state=0), parameters, cv=3)
    >>> grid_search.fit(X, y)
    >>> print(grid_search.best_params_)
    {'bootstrap': True, 'max_features': None, 'max_thresholds': None, 'n_estimators': 10}
    >>> clf = DecisionForestClassifier(random_state=0)
    >>> clf.set_params(**grid_search.best_params_)
    >>> clf.fit(X, y)
    >>> score = clf.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.973333

Parallel Processing (joblib + dask)

Install and setup dask::

$: conda install dask distributed

    >>> from dask.distributed import Client
    >>> client = Client()

    >>> clf = DecisionForestClassifier(random_state=0)
    >>> from sklearn.externals.joblib import parallel_backend
    >>> with parallel_backend('dask', n_jobs=-1):  # 'loky' when not using dask
    ...     clf.fit(X, y)
    ...     score = clf.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.966667

View progress with dask::

    Firefox: http://localhost:8787/status

.. only:: html

    .. figure:: ./_static/dask.png
       :align: center

Tested Version
==============

``koho`` 0.0.2,
python 3.7.3,
cython 0.29.7,
git 2.17.1,
conda 4.6.8,
pip 19.0.3,
numpy 1.16.2,
scipy 1.2.1,
scikit-learn 0.20.3,
python-graphviz 0.10.1,
jupyter 1.0.0,
tornado 5.1.1,
sphinx 2.0.1,
sphinx-gallery 0.3.1,
sphinx_rtd_theme 0.4.3,
matplotlib 3.0.3,
numpydoc 0.8.0,
pillow 6.0.0,
pytest 4.4.0,
pytest-cov 2.6.1,
dask 1.1.5,
distributed 1.26.1,

