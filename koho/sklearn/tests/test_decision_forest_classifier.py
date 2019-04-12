""" Testing of the Decision Forest Classifier.
"""

# Author: AI Werkstatt (TM)
# (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

import pytest

import numpy as np
import pickle

from sklearn.datasets import load_iris

from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
from sklearn.externals.joblib import parallel_backend

from koho.sklearn import DecisionForestClassifier

precision = 1e-7  # used for floating point "==" test

# iris dataset
@pytest.fixture
def iris():
    return load_iris()

# sklearn compatible
# ==================

# sklearn's check_estimator()
def test_sklearn_check_estimator():
    check_estimator(DecisionForestClassifier)

# sklearn's pipeline
def test_sklearn_pipeline(iris):
    X, y = iris.data, iris.target
    pipe = make_pipeline(DecisionForestClassifier(random_state=0))
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    score = pipe.score(X, y)
    assert score > 0.9666666666666667 - precision and score < 0.9666666666666667 + precision

# sklearn's grid search
def test_sklearn_grid_search(iris):
    X, y = iris.data, iris.target
    parameters = [{'n_estimators': [10, 20],
                   'bootstrap': [False, True],
                   'max_features': [None, 1],
                   'max_thresholds': [None, 1]
                   }]
    grid_search = GridSearchCV(DecisionForestClassifier(random_state=0), parameters, cv=3)
    grid_search.fit(X, y)
    assert grid_search.best_params_['n_estimators'] == 10
    assert grid_search.best_params_['bootstrap'] == True
    assert grid_search.best_params_['max_features'] is None
    assert grid_search.best_params_['max_thresholds'] is None
    clf = DecisionForestClassifier(random_state=0)
    clf.set_params(**grid_search.best_params_)
    assert clf.n_estimators == 10
    assert clf.bootstrap == True
    assert clf.class_balance == 'balanced'
    assert clf.max_depth == 3
    assert clf.max_features is None
    assert clf.max_thresholds is None
    assert clf.random_state == 0
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.9733333333333334 - precision and score < 0.9733333333333334 + precision

# sklearn's persistence
def test_sklearn_persistence(iris):
    X, y = iris.data, iris.target
    clf = DecisionForestClassifier(random_state=0)
    clf.fit(X, y)
    with open("clf.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("clf.pkl", "rb") as f:
        clf2 = pickle.load(f)
    score = clf2.score(X, y)
    assert score > 0.9666666666666667 - precision and score < 0.9666666666666667 + precision

# sklearn's parallel processing (joblib)
def test_sklearn_parallel_processing(iris):
    X, y = iris.data, iris.target
    clf = DecisionForestClassifier(random_state=0)
    with parallel_backend('loky', n_jobs=-1):
        clf.fit(X, y)
        score = clf.score(X, y)
    assert score > 0.9666666666666667 - precision and score < 0.9666666666666667 + precision

# iris dataset
# ============

def test_iris(iris):
    X, y = iris.data, iris.target
    clf = DecisionForestClassifier(n_estimators = 10, bootstrap = True, random_state=0)
    assert clf.n_estimators == 10
    assert clf.bootstrap == True
    assert clf.class_balance == 'balanced'
    assert clf.max_depth == 3
    assert clf.max_features == 'auto'
    assert clf.max_thresholds is None
    assert clf.random_state == 0
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.03089892, 0.02331612, 0.4162988,  0.52948616]
    for i1, i2 in zip(feature_importances, feature_importances_target):
        assert i1 > i2 - precision and i1 < i2 + precision
    tree_idx = 0
    dot_data = clf.estimators_[tree_idx].export_graphviz(
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        rotate=True)
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'rankdir=LR ;' '\n' \
        r'0 [label="petal length (cm) <= 2.45\np(class) = [0.33, 0.33, 0.33]\nclass, n = 150", fillcolor="#00FF0000"] ;' '\n' \
        r'0 -> 1 [penwidth=3.333333, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 2 [penwidth=6.666667, headlabel="False", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'1 [label="[1, 0, 0]\nsetosa", fillcolor="#FF0000FF"] ;' '\n' \
        r'2 [label="petal width (cm) <= 1.75\n[0, 0.5, 0.5]", fillcolor="#00FF003F"] ;' '\n' \
        r'2 -> 3 [penwidth=3.566218] ;' '\n' \
        r'2 -> 6 [penwidth=3.100449] ;' '\n' \
        r'3 [label="petal length (cm) <= 4.95\n[0, 0.91, 0.09]", fillcolor="#00FF00C2"] ;' '\n' \
        r'3 -> 4 [penwidth=3.106061] ;' '\n' \
        r'3 -> 5 [penwidth=0.460157] ;' '\n' \
        r'4 [label="[0, 1, 0]\nversicolor", fillcolor="#00FF00FF"] ;' '\n' \
        r'5 [label="[0, 0.33, 0.67]\nvirginica", fillcolor="#0000FF56"] ;' '\n' \
        r'6 [label="sepal width (cm) > 3.1\n[0, 0.02, 0.98]", fillcolor="#0000FFEC"] ;' '\n' \
        r'6 -> 8 [penwidth=0.878227] ;' '\n' \
        r'6 -> 7 [penwidth=2.222222] ;' '\n' \
        r'8 [label="[0, 0.09, 0.91]\nvirginica", fillcolor="#0000FFC2"] ;' '\n' \
        r'7 [label="[0, 0, 1]\nvirginica", fillcolor="#0000FFFF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target
    t = clf.estimators_[tree_idx].export_text()
    t_target = r'0 X[2]<=2.45 [50, 50.0, 50]; 0->1; 0->2; 1 [50, 0, 0]; 2 X[3]<=1.75 [0, 50.0, 50]; 2->3; 2->6; 3 X[2]<=4.95 [0, 48.86, 4.63]; 3->4; 3->5; 4 [0, 46.59, 0]; 5 [0, 2.27, 4.63]; 6 X[1]<=3.1 [0, 1.14, 45.37]; 6->7; 6->8; 7 [0, 0, 33.33]; 8 [0, 1.14, 12.04]; '
    assert t == t_target
    c = clf.predict(X)
    assert sum(c) == 148
    cp = clf.predict_proba(X)
    assert sum(sum(cp)) > 150 - precision and sum(sum(cp)) < 150 + precision
    score = clf.score(X, y)
    assert score > 0.96 - precision and score < 0.96 + precision
