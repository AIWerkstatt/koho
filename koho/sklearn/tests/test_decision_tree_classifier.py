""" Testing of the Decision Tree Classifier.
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

from koho.sklearn import DecisionTreeClassifier

precision = 1e-7  # used for floating point "==" test

# iris dataset
@pytest.fixture
def iris():
    return load_iris()

# sklearn compatible
# ==================

# sklearn's check_estimator()
def test_sklearn_check_estimator():
    check_estimator(DecisionTreeClassifier)

# sklearn's pipeline
def test_sklearn_pipeline(iris):
    X, y = iris.data, iris.target
    pipe = make_pipeline(DecisionTreeClassifier(random_state=0))
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    score = pipe.score(X, y)
    assert score > 1.0 - precision and score < 1.0 + precision

# sklearn's grid search
def test_sklearn_grid_search(iris):
    X, y = iris.data, iris.target
    parameters = [{'class_balance': ['balanced'],
                   'max_depth': [1, 3, 5]}]
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), parameters, cv=3)
    grid_search.fit(X, y)
    assert grid_search.best_params_['class_balance'] == 'balanced'
    assert grid_search.best_params_['max_depth'] == 5
    clf = DecisionTreeClassifier(random_state=0)
    clf.set_params(**grid_search.best_params_)
    assert clf.class_balance == 'balanced'
    assert clf.max_depth == 5
    assert clf.max_features is None
    assert clf.max_thresholds is None
    assert clf.random_state == 0
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 1.0 - precision and score < 1.0 + precision

# sklearn's persistence
def test_sklearn_persistence(iris):
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    with open("clf.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("clf.pkl", "rb") as f:
        clf2 = pickle.load(f)
    score = clf2.score(X, y)
    assert score > 1.0 - precision and score < 1.0 + precision

# iris dataset
# ============

def test_iris(iris):
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    assert clf.class_balance == 'balanced'
    assert clf.max_depth == 5
    assert clf.max_features is None
    assert clf.max_thresholds is None
    assert clf.random_state == 0
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.,         0.01333333, 0.06405596, 0.92261071]
    for i1, i2 in zip(feature_importances, feature_importances_target):
        assert i1 > i2 - precision and i1 < i2 + precision
    dot_data = clf.export_graphviz(
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        rotate=True)
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'rankdir=LR ;' '\n' \
        r'0 [label="petal width (cm) <= 0.8\np(class) = [0.33, 0.33, 0.33]\nclass, n = 150", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=3.333333, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 2 [penwidth=6.666667, headlabel="False", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'1 [label="[1, 0, 0]\nsetosa", fillcolor="#FF0000FF"] ;' '\n' \
        r'2 [label="petal width (cm) <= 1.75\n[0, 0.5, 0.5]", fillcolor="#00FF003F"] ;' '\n' \
        r'2 -> 3 [penwidth=3.600000] ;' '\n' \
        r'2 -> 12 [penwidth=3.066667] ;' '\n' \
        r'3 [label="petal length (cm) <= 4.95\n[0, 0.91, 0.09]", fillcolor="#00FF00BE"] ;' '\n' \
        r'3 -> 4 [penwidth=3.200000] ;' '\n' \
        r'3 -> 7 [penwidth=0.400000] ;' '\n' \
        r'4 [label="petal width (cm) <= 1.65\n[0, 0.98, 0.02]", fillcolor="#00FF00EF"] ;' '\n' \
        r'4 -> 5 [penwidth=3.133333] ;' '\n' \
        r'4 -> 6 [penwidth=0.066667] ;' '\n' \
        r'5 [label="[0, 1, 0]\nversicolor", fillcolor="#00FF00FF"] ;' '\n' \
        r'6 [label="[0, 0, 1]\nvirginica", fillcolor="#0000FFFF"] ;' '\n' \
        r'7 [label="petal width (cm) > 1.55\n[0, 0.33, 0.67]", fillcolor="#0000FF55"] ;' '\n' \
        r'7 -> 9 [penwidth=0.200000] ;' '\n' \
        r'7 -> 8 [penwidth=0.200000] ;' '\n' \
        r'9 [label="petal length (cm) <= 5.45\n[0, 0.67, 0.33]", fillcolor="#00FF0055"] ;' '\n' \
        r'9 -> 10 [penwidth=0.133333] ;' '\n' \
        r'9 -> 11 [penwidth=0.066667] ;' '\n' \
        r'10 [label="[0, 1, 0]\nversicolor", fillcolor="#00FF00FF"] ;' '\n' \
        r'11 [label="[0, 0, 1]\nvirginica", fillcolor="#0000FFFF"] ;' '\n' \
        r'8 [label="[0, 0, 1]\nvirginica", fillcolor="#0000FFFF"] ;' '\n' \
        r'12 [label="petal length (cm) <= 4.85\n[0, 0.02, 0.98]", fillcolor="#0000FFEE"] ;' '\n' \
        r'12 -> 13 [penwidth=0.200000] ;' '\n' \
        r'12 -> 16 [penwidth=2.866667] ;' '\n' \
        r'13 [label="sepal width (cm) > 3.1\n[0, 0.33, 0.67]", fillcolor="#0000FF55"] ;' '\n' \
        r'13 -> 15 [penwidth=0.066667] ;' '\n' \
        r'13 -> 14 [penwidth=0.133333] ;' '\n' \
        r'15 [label="[0, 1, 0]\nversicolor", fillcolor="#00FF00FF"] ;' '\n' \
        r'14 [label="[0, 0, 1]\nvirginica", fillcolor="#0000FFFF"] ;' '\n' \
        r'16 [label="[0, 0, 1]\nvirginica", fillcolor="#0000FFFF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target
    t = clf.export_text()
    t_target = r'0 X[3]<=0.8 [50, 50, 50]; 0->1; 0->2; 1 [50, 0, 0]; 2 X[3]<=1.75 [0, 50, 50]; 2->3; 2->12; 3 X[2]<=4.95 [0, 49, 5]; 3->4; 3->7; 4 X[3]<=1.65 [0, 47, 1]; 4->5; 4->6; 5 [0, 47, 0]; 6 [0, 0, 1]; 7 X[3]<=1.55 [0, 2, 4]; 7->8; 7->9; 8 [0, 0, 3]; 9 X[2]<=5.45 [0, 2, 1]; 9->10; 9->11; 10 [0, 2, 0]; 11 [0, 0, 1]; 12 X[2]<=4.85 [0, 1, 45]; 12->13; 12->16; 13 X[1]<=3.1 [0, 1, 2]; 13->14; 13->15; 14 [0, 0, 2]; 15 [0, 1, 0]; 16 [0, 0, 43]; '
    assert t == t_target
    c = clf.predict(X)
    assert sum(c) == 150
    cp = clf.predict_proba(X)
    assert sum(sum(cp)) > 150 - precision and sum(sum(cp)) < 150 + precision
    score = clf.score(X, y)
    assert score > 1.0 - precision and score < 1.0 + precision

