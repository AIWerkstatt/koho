.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

**koho** (Hawaiian word for 'to estimate') is a **Decision Forest** **C++ library**
with a `scikit-learn`_ compatible **Python interface**.

- Classification
- Numerical (dense) data
- Missing values (Not Missing At Random (NMAR))
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

We use the `iris`_ dataset provided by `scikit-learn`_ for illustration purposes.

.. code-block:: python

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target

.. _`iris`: https://en.wikipedia.org/wiki/Iris_flower_data_set

.. code-block:: python

    >>> from koho.sklearn import DecisionTreeClassifier, DecisionForestClassifier
    >>> clf = DecisionForestClassifier(random_state=0)

| Decision Tree: ``max_features=None`` and ``max_thresholds=None``
| Random Tree: ``max_features<n_features`` and ``max_thresholds=None``
| Extreme Randomized Trees (ET): ``max_thresholds=1``
| Totally Randomized Trees: ``max_features=1`` and ``max_thresholds=1`` very similar to Perfect Random Trees (PERT).

**Training**

.. code-block:: python

    >>> clf.fit(X, y)
    DecisionForestClassifier(bootstrap=False, class_balance='balanced',
             max_depth=3, max_features='auto', max_thresholds=None,
             missing_values=None, n_estimators=100, n_jobs=None,
             oob_score=False, random_state=0)

Feature Importances

.. code-block:: python

    >>> feature_importances = clf.feature_importances_
    >>> print(feature_importances)
    [0.09045256 0.00816573 0.38807981 0.5133019]

Visualize Trees

Export a tree in `graphviz`_ format and visualize it using `graphviz`_:

.. code-block:: text

    $: conda install python-graphviz

.. code-block:: python

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

Convert the tree to different file formats (e.g. pdf, png):

.. code-block:: python

    >>> graph.render("iris", format='pdf')
    iris.pdf

Export a tree in a compact textual format:

.. code-block:: python

    >>> t = clf.estimators_[tree_idx].export_text()
    >>> print(t)
    0 X[3]<=0.8 [50, 50, 50]; 0->1; 0->2; 1 [50, 0, 0]; 2 X[3]<=1.75 [0, 50, 50]; 2->3; 2->6; 3 X[2]<=4.95 [0, 49, 5]; 3->4; 3->5; 4 [0, 47, 1]; 5 [0, 2, 4]; 6 X[3]<=1.85 [0, 1, 45]; 6->7; 6->8; 7 [0, 1, 11]; 8 [0, 0, 34];

.. _`graphviz`: http://www.graphviz.org/

Persistence

.. code-block:: python

    >>> import pickle
    >>> with open("clf.pkl", "wb") as f:
    ...     pickle.dump(clf, f)
    >>> with open("clf.pkl", "rb") as f:
    ...     clf2 = pickle.load(f)

**Classification**

.. code-block:: python

    >>> c = clf2.predict(X)
    >>> print(c)
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

.. code-block:: python

    >>> cp = clf2.predict_proba(X)
    >>> print(cp)
    [[1.         0.         0.        ]
     [1.         0.         0.        ]
     [1.         0.         0.        ]
     ...
     [0.         0.01935722 0.98064278]
     [0.         0.01935722 0.98064278]
     [0.         0.09155897 0.90844103]]

**Testing**

.. code-block:: python

    >>> score = clf2.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.966667

**scikit-learn's ecosystem**

Pipeline

.. code-block:: python

    >>> from sklearn.pipeline import make_pipeline
    >>> pipe = make_pipeline(DecisionForestClassifier(random_state=0))
    >>> pipe.fit(X, y)
    >>> pipe.predict(X)
    >>> pipe.predict_proba(X)
    >>> score = pipe.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.966667

Grid Search

.. code-block:: python

    >>> from sklearn.model_selection import GridSearchCV
    >>> parameters = [{'n_estimators': [10, 20],
    ...                'bootstrap': [False, True],
    ...                'max_features': [None, 1],
    ...                'max_thresholds': [None, 1]}]
    >>> grid_search = GridSearchCV(DecisionForestClassifier(random_state=0), parameters, iid=False)
    >>> grid_search.fit(X, y)
    >>> print(grid_search.best_params_)
    {'bootstrap': False, 'max_features': None, 'max_thresholds': 1, 'n_estimators': 10}
    >>> clf = DecisionForestClassifier(random_state=0)
    >>> clf.set_params(**grid_search.best_params_)
    >>> clf.fit(X, y)
    >>> score = clf.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.966667

Parallel Processing (joblib + dask)

Install and setup dask:

.. code-block:: text

    $: conda install dask distributed

.. code-block:: python

    >>> from dask.distributed import Client
    >>> client = Client()

.. code-block:: python

    >>> clf = DecisionForestClassifier(random_state=0)
    >>> from sklearn.externals.joblib import parallel_backend
    >>> with parallel_backend('dask', n_jobs=-1):  # 'loky' when not using dask
    ...     clf.fit(X, y)
    ...     score = clf.score(X, y)
    >>> print("Score: %f" % score)
    Score: 0.966667

View progress with dask:

.. code-block:: text

    Firefox: http://localhost:8787/status

.. only:: html

    .. figure:: ./_static/dask.png
       :align: center

C++
===

We provide a **C++ library**.

Classification
--------------

The koho library provides the following classifiers:

`DecisionTreeClassifier <_static/cpp/html/classkoho_1_1DecisionTreeClassifier.html#://>`_
`DecisionForestClassifier <_static/cpp/html/classkoho_1_1DecisionForestClassifier.html#://>`_

We use a simple example for illustration purposes.

.. code-block:: text

    vector<string>  classes = {"A", "B"};
    long            n_classes = classes.size();
    vector<string>  features = {"a", "b", "c"};
    long            n_features = features.size();

    vector<double>  X = {0, 0, 0,
                         0, 0, 1,
                         0, 1, 0,
                         0, 1, 1,
                         0, 1, 1,
                         1, 0, 0,
                         1, 0, 0,
                         1, 0, 0,
                         1, 0, 0,
                         1, 1, 1};
    vector<long>    y = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    unsigned long   n_samples = y.size();

.. code-block:: text

    #include <decision_tree.h>
    #include <decision_forest.h>

    using namespace koho;

    // Hyperparameters
    string          class_balance  = "balanced";
    long            max_depth = 3;
    long            max_features = n_features;
    long            max_thresholds = 0;
    string          missing_values = "None";
    // Random Number Generator
    long            random_state = 0;

    DecisionTreeClassifier dtc(classes, n_classes,
                               features, n_features,
                               class_balance, max_depth,
                               max_features, max_thresholds,
                               missing_values,
                               random_state);

**Training**

.. code-block:: text

    dfc.fit(&X[0], &y[0], n_samples);

Feature Importances

.. code-block:: text

    vector<double> importances(n_features);
    dtc.calculate_feature_importances(&importances[0]);
    for (auto i: importances) cout << i << ' ';
    // 0.454545 0.545455 0

Visualize Trees

Export a tree in `graphviz`_ format and visualize it using `graphviz`_:

.. code-block:: text

    $: sudo apt install graphviz
    $: sudo apt install xdot

.. code-block:: text

    dtc.export_graphviz("simple_example", true);

.. code-block:: text

    $: xdot simple_example.gv

.. only:: html

    .. figure:: ./_static/simple_example.png
       :align: center

Convert the tree to different file formats (e.g. pdf, png):

.. code-block:: text

    $: dot -Tpdf simple_example.gv -o simple_example.pdf

Export a tree in a compact textual format:

.. code-block:: text

    cout << dtc.export_text() << endl;
    // 0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.875]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.875]; 4 [0, 3.125];

Persistence

.. code-block:: text

    dtc.export_serialize("simple_example");
    DecisionTreeClassifier dtc2 = DecisionTreeClassifier::import_deserialize("simple_example");
    // simple_example.dtc

**Classification**

.. code-block:: text

    vector<long>    c(n_samples, 0);
    dtc2.predict(&X[0], n_samples, &c[0]);
    for (auto i: c) cout << i << ' ';
    // 0 0 1 1 1 1 1 1 1 1

**Testing**

.. code-block:: text

    double score = dtc2.score(&X[0], &y[0], n_samples);
    cout << score
    // 1.0

Tested Version
==============

``koho`` 1.0.0,
python 3.7.3,
cython 0.29.7,
gcc 7.3.0 C++ 17,
git 2.17.1,
conda 4.6.8,
pip 19.0.3,
numpy 1.16.2,
scipy 1.2.1,
scikit-learn 0.20.3,
python-graphviz 0.10.1,
jupyter 1.0.0,
tornado 5.1.1,
doxygen 1.8.13,
sphinx 2.0.1,
sphinx-gallery 0.3.1,
sphinx_rtd_theme 0.4.3,
matplotlib 3.0.3,
numpydoc 0.8.0,
pillow 6.0.0,
pytest 4.4.0,
pytest-cov 2.6.1,
dask 1.1.5,
distributed 1.26.1

