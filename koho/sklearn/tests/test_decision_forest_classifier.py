""" Testing of the Decision Forest Classifier.
"""

# Author: AI Werkstatt (TM)
# (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

import pytest

import numpy as np
import pickle
import graphviz

from sklearn.datasets import load_iris

from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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
    assert grid_search.best_params_['bootstrap'] == False
    assert grid_search.best_params_['max_features'] is None
    assert grid_search.best_params_['max_thresholds'] == 1
    clf = DecisionForestClassifier(random_state=0)
    clf.set_params(**grid_search.best_params_)
    assert clf.n_estimators == 10
    assert clf.bootstrap == False
    assert clf.class_balance == 'balanced'
    assert clf.max_depth == 3
    assert clf.max_features is None
    assert clf.max_thresholds == 1
    assert clf.random_state == 0
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.9666666666666667 - precision and score < 0.9666666666666667 + precision

# sklearn's persistence
def test_sklearn_persistence(iris):
    X, y = iris.data, iris.target
    clf = DecisionForestClassifier(random_state=0)
    clf.fit(X, y)
    with open("clf_dfc.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("clf_dfc.pkl", "rb") as f:
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

# iris dataset (User's Guide Python)
# ==================================

def test_iris(iris):
    X, y = iris.data, iris.target
    clf = DecisionForestClassifier(random_state=0)
    assert clf.n_estimators == 100
    assert clf.bootstrap == False
    assert clf.oob_score == False
    assert clf.class_balance == 'balanced'
    assert clf.max_depth == 3
    assert clf.max_features == 'auto'
    assert clf.max_thresholds is None
    assert clf.missing_values is None
    assert clf.n_jobs is None
    assert clf.random_state == 0
    # Training
    clf.fit(X, y)
    # Feature Importances
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.09045256, 0.00816573, 0.38807981, 0.5133019]
    for i1, i2 in zip(feature_importances, feature_importances_target):
        assert i1 > i2 - precision and i1 < i2 + precision
    # Visualize Trees
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
        r'0 [label="petal width (cm) <= 0.8\np(class) = [0.33, 0.33, 0.33]\nclass, n = 150", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=3.333333, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 2 [penwidth=6.666667] ;' '\n' \
        r'1 [label="[1, 0, 0]\nsetosa", fillcolor="#FF0000FF"] ;' '\n' \
        r'2 [label="petal width (cm) <= 1.75\n[0, 0.5, 0.5]", fillcolor="#00FF003F"] ;' '\n' \
        r'2 -> 3 [penwidth=3.600000] ;' '\n' \
        r'2 -> 6 [penwidth=3.066667] ;' '\n' \
        r'3 [label="petal length (cm) <= 4.95\n[0, 0.91, 0.09]", fillcolor="#00FF00BE"] ;' '\n' \
        r'3 -> 4 [penwidth=3.200000] ;' '\n' \
        r'3 -> 5 [penwidth=0.400000] ;' '\n' \
        r'4 [label="[0, 0.98, 0.02]\nversicolor", fillcolor="#00FF00EF"] ;' '\n' \
        r'5 [label="[0, 0.33, 0.67]\nvirginica", fillcolor="#0000FF55"] ;' '\n' \
        r'6 [label="petal width (cm) <= 1.85\n[0, 0.02, 0.98]", fillcolor="#0000FFEE"] ;' '\n' \
        r'6 -> 7 [penwidth=0.800000] ;' '\n' \
        r'6 -> 8 [penwidth=2.266667] ;' '\n' \
        r'7 [label="[0, 0.08, 0.92]\nvirginica", fillcolor="#0000FFC4"] ;' '\n' \
        r'8 [label="[0, 0, 1]\nvirginica", fillcolor="#0000FFFF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target
    # Export textual format
    t = clf.estimators_[tree_idx].export_text()
    t_target = r'0 X[3]<=0.8 [50, 50, 50]; 0->1; 0->2; 1 [50, 0, 0]; 2 X[3]<=1.75 [0, 50, 50]; 2->3; 2->6; 3 X[2]<=4.95 [0, 49, 5]; 3->4; 3->5; 4 [0, 47, 1]; 5 [0, 2, 4]; 6 X[3]<=1.85 [0, 1, 45]; 6->7; 6->8; 7 [0, 1, 11]; 8 [0, 0, 34]; '
    assert t == t_target
    # Persistence
    with open("iris_dfc.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("iris_dfc.pkl", "rb") as f:
        clf2 = pickle.load(f)
    assert clf2.estimators_[tree_idx].export_text() == clf.estimators_[tree_idx].export_text()
    # Classification
    c = clf2.predict(X)
    assert sum(c) == 147
    cp = clf2.predict_proba(X)
    assert sum(sum(cp)) > 150 - precision and sum(sum(cp)) < 150 + precision
    # Testing
    score = clf2.score(X, y)
    assert score > 0.9666666666666667 - precision and score < 0.9666666666666667 + precision

# simple example
# ==============

classes = ['A', 'B']
features = ['a', 'b', 'c']
X = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]]).astype(np.double)
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

X_test = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]]).astype(np.double)
y_test = np.array([0, 0, 1, 1, 1, 1, 1, 1])

def test_simple_example():
    clf = DecisionForestClassifier(
        n_estimators=10,
        bootstrap=False,
        oob_score=False,
        class_balance='balanced',
        max_depth=3,
        max_features=2,
        max_thresholds=None,
        missing_values=None,
        random_state=0)
    assert clf.n_estimators == 10
    assert clf.bootstrap == False
    assert clf.oob_score == False
    assert clf.class_balance == 'balanced'
    assert clf.max_depth == 3
    assert clf.max_features == 2
    assert clf.max_thresholds is None
    assert clf.missing_values is None
    assert clf.random_state == 0
    # Training
    clf.fit(X, y)
    # Feature Importances
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.48484848, 0.47939394, 0.03575758]
    for i1, i2 in zip(feature_importances, feature_importances_target):
        assert i1 > i2 - precision and i1 < i2 + precision
    # Visualize Tree
    tree_idx = 0
    dot_data = clf.estimators_[tree_idx].export_graphviz(
        feature_names=features,
        class_names=classes,
        rotate=True)
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'rankdir=LR ;' '\n' \
        r'0 [label="a <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="b <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\nA", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target
    # Export textual format
    t = clf.estimators_[tree_idx].export_text()
    t_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert t == t_target
    # Persistence
    with open("simple_example_dfc.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("simple_example_dfc.pkl", "rb") as f:
        clf2 = pickle.load(f)
    assert clf2.estimators_[tree_idx].export_text() == clf.estimators_[tree_idx].export_text()
    # Classification
    c = clf2.predict(X)
    c_target = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    for i1, i2 in zip(c, c_target):
        assert i1 > i2 - precision and i1 < i2 + precision
    # Testing
    score = clf2.score(X, y)
    assert score > 1.0 - precision and score < 1.0 + precision

# simple multi-output example
# ===========================

# multi-output fed with single-output
# -----------------------------------

def test_simple_multi_output_example_with_single_output():
    classes = [['0', '1', '2', '3', '4', '5', '6', '7']]
    features = ['2^2', '2^1', '2^0']

    X_mo = np.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [0, 1, 1],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0],
                     [1, 1, 1]]).astype(np.double)
    y_mo = np.array([[0],
                     [1],
                     [2],
                     [3],
                     [4],
                     [5],
                     [6],
                     [7]]).astype(np.long)

    clf = DecisionForestClassifier(
        n_estimators=10,
        bootstrap=False,
        oob_score=False,
        class_balance='balanced',
        max_depth=3,
        max_features=1,
        max_thresholds=1,
        missing_values=None,
        random_state=0)
    # Training
    clf.fit(X_mo, y_mo)
    # Feature Importances
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.32857143, 0.32857143, 0.34285714]
    for i1, i2 in zip(feature_importances, feature_importances_target):
        assert i1 > i2 - precision and i1 < i2 + precision
    # Visualize Tree
    dot_data = clf.estimators_[0].export_graphviz(
        feature_names=features,
        class_names=classes,
        rotate=True)
    #filename = "simple_example_multi_output_with_single_output_dfc_dtc_0"
    #graph = graphviz.Source(dot_data)
    #graph.render(filename, format='pdf')
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'rankdir=LR ;' '\n' \
        r'0 [label="2^2 <= 0.874\np(class) = [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]\nclass, n = 8", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=5.000000, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 8 [penwidth=5.000000] ;' '\n' \
        r'1 [label="2^0 <= 0.753\n[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]", fillcolor="#FF000024"] ;' '\n' \
        r'1 -> 2 [penwidth=2.500000] ;' '\n' \
        r'1 -> 5 [penwidth=2.500000] ;' '\n' \
        r'2 [label="2^1 <= 0.368\n[0.5, 0, 0.5, 0, 0, 0, 0, 0]", fillcolor="#FF00006D"] ;' '\n' \
        r'2 -> 3 [penwidth=1.250000] ;' '\n' \
        r'2 -> 4 [penwidth=1.250000] ;' '\n' \
        r'3 [label="[1, 0, 0, 0, 0, 0, 0, 0]\n0", fillcolor="#FF0000FF"] ;' '\n' \
        r'4 [label="[0, 0, 1, 0, 0, 0, 0, 0]\n2", fillcolor="#0000FFFF"] ;' '\n' \
        r'5 [label="2^1 <= 0.31\n[0, 0.5, 0, 0.5, 0, 0, 0, 0]", fillcolor="#00FF006D"] ;' '\n' \
        r'5 -> 6 [penwidth=1.250000] ;' '\n' \
        r'5 -> 7 [penwidth=1.250000] ;' '\n' \
        r'6 [label="[0, 1, 0, 0, 0, 0, 0, 0]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'7 [label="[0, 0, 0, 1, 0, 0, 0, 0]\n3", fillcolor="#FFFF00FF"] ;' '\n' \
        r'8 [label="2^1 <= 0.247\n[0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25]", fillcolor="#00FFFF24"] ;' '\n' \
        r'8 -> 9 [penwidth=2.500000] ;' '\n' \
        r'8 -> 12 [penwidth=2.500000] ;' '\n' \
        r'9 [label="2^0 <= 0.316\n[0, 0, 0, 0, 0.5, 0.5, 0, 0]", fillcolor="#00FFFF6D"] ;' '\n' \
        r'9 -> 10 [penwidth=1.250000] ;' '\n' \
        r'9 -> 11 [penwidth=1.250000] ;' '\n' \
        r'10 [label="[0, 0, 0, 0, 1, 0, 0, 0]\n4", fillcolor="#00FFFFFF"] ;' '\n' \
        r'11 [label="[0, 0, 0, 0, 0, 1, 0, 0]\n5", fillcolor="#FF00FFFF"] ;' '\n' \
        r'12 [label="2^0 <= 0.516\n[0, 0, 0, 0, 0, 0, 0.5, 0.5]", fillcolor="#FF80006D"] ;' '\n' \
        r'12 -> 13 [penwidth=1.250000] ;' '\n' \
        r'12 -> 14 [penwidth=1.250000] ;' '\n' \
        r'13 [label="[0, 0, 0, 0, 0, 0, 1, 0]\n6", fillcolor="#FF8000FF"] ;' '\n' \
        r'14 [label="[0, 0, 0, 0, 0, 0, 0, 1]\n7", fillcolor="#00FF80FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target
    # Export textual format
    t = clf.estimators_[0].export_text()
    t_target = r'0 X[0]<=0.874 [1, 1, 1, 1, 1, 1, 1, 1]; 0->1; 0->8; 1 X[2]<=0.753 [1, 1, 1, 1, 0, 0, 0, 0]; 1->2; 1->5; 2 X[1]<=0.368 [1, 0, 1, 0, 0, 0, 0, 0]; 2->3; 2->4; 3 [1, 0, 0, 0, 0, 0, 0, 0]; 4 [0, 0, 1, 0, 0, 0, 0, 0]; 5 X[1]<=0.31 [0, 1, 0, 1, 0, 0, 0, 0]; 5->6; 5->7; 6 [0, 1, 0, 0, 0, 0, 0, 0]; 7 [0, 0, 0, 1, 0, 0, 0, 0]; 8 X[1]<=0.247 [0, 0, 0, 0, 1, 1, 1, 1]; 8->9; 8->12; 9 X[2]<=0.316 [0, 0, 0, 0, 1, 1, 0, 0]; 9->10; 9->11; 10 [0, 0, 0, 0, 1, 0, 0, 0]; 11 [0, 0, 0, 0, 0, 1, 0, 0]; 12 X[2]<=0.516 [0, 0, 0, 0, 0, 0, 1, 1]; 12->13; 12->14; 13 [0, 0, 0, 0, 0, 0, 1, 0]; 14 [0, 0, 0, 0, 0, 0, 0, 1]; '
    assert t == t_target
    # Persistence
    with open("simple_example_multi_output_with_single_output_dtc.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("simple_example_multi_output_with_single_output_dtc.pkl", "rb") as f:
        clf2 = pickle.load(f)
    assert clf2.estimators_[0].export_text() == clf.estimators_[0].export_text()
    # Classification
    c = clf2.predict(X_mo)
    for i1, i2 in zip(c, y_mo):
        assert i1 > i2 - precision and i1 < i2 + precision
    # Testing
    score = clf2.score(X_mo, y_mo)
    assert score > 1.0 - precision and score < 1.0 + precision

# multi-output
# ------------

def test_simple_multi_output_example():
    classes = [['0', '1', '2', '3', '4', '5', '6', '7'],
               ['0', '4'],
               ['0', '2'],
               ['0', '1']]
    features = ['2^2', '2^1', '2^0']

    X_mo = np.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [0, 1, 1],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0],
                     [1, 1, 1]]).astype(np.double)
    y_mo = np.array([[0, 0, 0, 0],
                     [1, 0, 0, 1],
                     [2, 0, 1, 0],
                     [3, 0, 1, 1],
                     [4, 1, 0, 0],
                     [5, 1, 0, 1],
                     [6, 1, 1, 0],
                     [7, 1, 1, 1]]).astype(np.long)

    clf = DecisionForestClassifier(
        n_estimators=10,
        bootstrap=False,
        oob_score=False,
        class_balance='balanced',
        max_depth=3,
        max_features=1,
        max_thresholds=1,
        missing_values=None,
        random_state=0)
    # Training
    clf.fit(X_mo, y_mo)
    # Feature Importances
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.33157895, 0.33157895, 0.33684211]
    for i1, i2 in zip(feature_importances, feature_importances_target):
        assert i1 > i2 - precision and i1 < i2 + precision
    # Visualize Tree
    dot_data = clf.estimators_[0].export_graphviz(
        feature_names=features,
        class_names=classes,
        rotate=True)
    #filename = "simple_example_multi_output_dfc_dtc_0"
    #graph = graphviz.Source(dot_data)
    #graph.render(filename, format='pdf')
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'rankdir=LR ;' '\n' \
        r'0 [label="2^2 <= 0.874\np(class) = [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]\n[0.5, 0.5]\n[0.5, 0.5]\n[0.5, 0.5]\nclass, n = 8", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=5.000000, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 8 [penwidth=5.000000] ;' '\n' \
        r'1 [label="2^0 <= 0.753\n[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]\n[1, 0]\n[0.5, 0.5]\n[0.5, 0.5]\n", fillcolor="#FF000043"] ;' '\n' \
        r'1 -> 2 [penwidth=2.500000] ;' '\n' \
        r'1 -> 5 [penwidth=2.500000] ;' '\n' \
        r'2 [label="2^1 <= 0.368\n[0.5, 0, 0.5, 0, 0, 0, 0, 0]\n[1, 0]\n[0.5, 0.5]\n[1, 0]\n", fillcolor="#FF000093"] ;' '\n' \
        r'2 -> 3 [penwidth=1.250000] ;' '\n' \
        r'2 -> 4 [penwidth=1.250000] ;' '\n' \
        r'3 [label="[1, 0, 0, 0, 0, 0, 0, 0]\n[1, 0]\n[1, 0]\n[1, 0]\n0\n0\n0\n0\n", fillcolor="#FF0000FF"] ;' '\n' \
        r'4 [label="[0, 0, 1, 0, 0, 0, 0, 0]\n[1, 0]\n[0, 1]\n[1, 0]\n2\n0\n2\n0\n", fillcolor="#00FFFFFF"] ;' '\n' \
        r'5 [label="2^1 <= 0.31\n[0, 0.5, 0, 0.5, 0, 0, 0, 0]\n[1, 0]\n[0.5, 0.5]\n[0, 1]\n", fillcolor="#FFFF0093"] ;' '\n' \
        r'5 -> 6 [penwidth=1.250000] ;' '\n' \
        r'5 -> 7 [penwidth=1.250000] ;' '\n' \
        r'6 [label="[0, 1, 0, 0, 0, 0, 0, 0]\n[1, 0]\n[1, 0]\n[0, 1]\n1\n0\n0\n1\n", fillcolor="#FFFF00FF"] ;' '\n' \
        r'7 [label="[0, 0, 0, 1, 0, 0, 0, 0]\n[1, 0]\n[0, 1]\n[0, 1]\n3\n0\n2\n1\n", fillcolor="#00FF80FF"] ;' '\n' \
        r'8 [label="2^1 <= 0.247\n[0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25]\n[0, 1]\n[0.5, 0.5]\n[0.5, 0.5]\n", fillcolor="#FF400043"] ;' '\n' \
        r'8 -> 9 [penwidth=2.500000] ;' '\n' \
        r'8 -> 12 [penwidth=2.500000] ;' '\n' \
        r'9 [label="2^0 <= 0.316\n[0, 0, 0, 0, 0.5, 0.5, 0, 0]\n[0, 1]\n[1, 0]\n[0.5, 0.5]\n", fillcolor="#FF400093"] ;' '\n' \
        r'9 -> 10 [penwidth=1.250000] ;' '\n' \
        r'9 -> 11 [penwidth=1.250000] ;' '\n' \
        r'10 [label="[0, 0, 0, 0, 1, 0, 0, 0]\n[0, 1]\n[1, 0]\n[1, 0]\n4\n4\n0\n0\n", fillcolor="#FF4000FF"] ;' '\n' \
        r'11 [label="[0, 0, 0, 0, 0, 1, 0, 0]\n[0, 1]\n[1, 0]\n[0, 1]\n5\n4\n0\n1\n", fillcolor="#BFFF00FF"] ;' '\n' \
        r'12 [label="2^0 <= 0.516\n[0, 0, 0, 0, 0, 0, 0.5, 0.5]\n[0, 1]\n[0, 1]\n[0.5, 0.5]\n", fillcolor="#00BFFF93"] ;' '\n' \
        r'12 -> 13 [penwidth=1.250000] ;' '\n' \
        r'12 -> 14 [penwidth=1.250000] ;' '\n' \
        r'13 [label="[0, 0, 0, 0, 0, 0, 1, 0]\n[0, 1]\n[0, 1]\n[1, 0]\n6\n4\n2\n0\n", fillcolor="#00BFFFFF"] ;' '\n' \
        r'14 [label="[0, 0, 0, 0, 0, 0, 0, 1]\n[0, 1]\n[0, 1]\n[0, 1]\n7\n4\n2\n1\n", fillcolor="#00FFC0FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target
    # Export textual format
    t = clf.estimators_[0].export_text()
    t_target = r'0 X[0]<=0.874 [1, 1, 1, 1, 1, 1, 1, 1][4, 4][4, 4][4, 4]; 0->1; 0->8; 1 X[2]<=0.753 [1, 1, 1, 1, 0, 0, 0, 0][4, 0][2, 2][2, 2]; 1->2; 1->5; 2 X[1]<=0.368 [1, 0, 1, 0, 0, 0, 0, 0][2, 0][1, 1][2, 0]; 2->3; 2->4; 3 [1, 0, 0, 0, 0, 0, 0, 0][1, 0][1, 0][1, 0]; 4 [0, 0, 1, 0, 0, 0, 0, 0][1, 0][0, 1][1, 0]; 5 X[1]<=0.31 [0, 1, 0, 1, 0, 0, 0, 0][2, 0][1, 1][0, 2]; 5->6; 5->7; 6 [0, 1, 0, 0, 0, 0, 0, 0][1, 0][1, 0][0, 1]; 7 [0, 0, 0, 1, 0, 0, 0, 0][1, 0][0, 1][0, 1]; 8 X[1]<=0.247 [0, 0, 0, 0, 1, 1, 1, 1][0, 4][2, 2][2, 2]; 8->9; 8->12; 9 X[2]<=0.316 [0, 0, 0, 0, 1, 1, 0, 0][0, 2][2, 0][1, 1]; 9->10; 9->11; 10 [0, 0, 0, 0, 1, 0, 0, 0][0, 1][1, 0][1, 0]; 11 [0, 0, 0, 0, 0, 1, 0, 0][0, 1][1, 0][0, 1]; 12 X[2]<=0.516 [0, 0, 0, 0, 0, 0, 1, 1][0, 2][0, 2][1, 1]; 12->13; 12->14; 13 [0, 0, 0, 0, 0, 0, 1, 0][0, 1][0, 1][1, 0]; 14 [0, 0, 0, 0, 0, 0, 0, 1][0, 1][0, 1][0, 1]; '
    assert t == t_target
    # Persistence
    with open("simple_example_multi_output_dtc.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("simple_example_multi_output_dtc.pkl", "rb") as f:
        clf2 = pickle.load(f)
    assert clf2.estimators_[0].export_text() == clf.estimators_[0].export_text()
    # Classification
    c = clf2.predict(X_mo)
    for i1, i2 in zip(c.ravel(), y_mo.ravel()):
        assert i1 > i2 - precision and i1 < i2 + precision
    # Testing
    score = clf2.score(X_mo, y_mo)
    assert score > 1.0 - precision and score < 1.0 + precision

# DecisionForestClassifier.fit()
# ==============================

def test_fit():

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [2, 8]; 0->1; 0->4; 1 X[1]<=0.5 [2, 3]; 1->2; 1->3; 2 [2, 0]; 3 [0, 3]; 4 [0, 5]; '
    assert data == data_target

# n_estimators
# ------------

def test_fit_nestimators():

    clf = DecisionForestClassifier(n_estimators=-1, bootstrap=False, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'n_estimators' in str(excinfo.value)

    clf = DecisionForestClassifier(n_estimators=0, bootstrap=False, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'n_estimators' in str(excinfo.value)

    clf = DecisionForestClassifier(n_estimators=1, bootstrap=False, class_balance=None, max_depth=1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [2, 8]; 0->1; 0->2; 1 [2, 3]; 2 [0, 5]; '
    assert data == data_target

    clf = DecisionForestClassifier(n_estimators=10, bootstrap=False, class_balance=None, max_depth=1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [2, 8]; 0->1; 0->2; 1 [2, 3]; 2 [0, 5]; '
    assert data == data_target

    data = clf.estimators_[9].export_text()
    data_target = r'0 X[1]<=0.5 [2, 8]; 0->1; 0->2; 1 [2, 4]; 2 [0, 4]; '
    assert data == data_target

    with pytest.raises(IndexError):
        data = clf.estimators_[10].export_text()

    # misc

    clf = DecisionForestClassifier(n_estimators=[], bootstrap=False, random_state=0)
    with pytest.raises(TypeError):
        clf.fit(X, y)

# bootstrap
# ---------

def test_fit_bootstrap(iris):

    # simple example

    # data set it too small for bootstrap = True

    # misc

    clf = DecisionForestClassifier(bootstrap=[], random_state=0)
    with pytest.raises(TypeError):
        clf.fit(X, y)

    clf = DecisionForestClassifier(bootstrap=1, random_state=0)
    with pytest.raises(TypeError):
        clf.fit(X, y)

    clf = DecisionForestClassifier(bootstrap=2, random_state=0)
    with pytest.raises(TypeError):
        clf.fit(X, y)

    clf = DecisionForestClassifier(bootstrap=1.0, random_state=0)
    with pytest.raises(TypeError):
        clf.fit(X, y)

    # iris dataset

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4, random_state=0)

    clf = DecisionForestClassifier(bootstrap=True,
                                   n_estimators=20,
                                   class_balance='balanced',
                                   max_depth=None,
                                   max_features=None,
                                   max_thresholds=None,
                                   random_state=0)

    clf.set_params(n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score > .95 - precision and score < .95 + precision

# bootstrap + oob_score
# ---------------------

def test_fit_bootstrap_oob_score(iris):

    # simple example

    clf = DecisionForestClassifier(bootstrap=False, oob_score=True)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'oob_score: only available if bootstrap=True' in str(excinfo.value)

    clf = DecisionForestClassifier(oob_score=[])
    with pytest.raises(TypeError) as excinfo:
        clf.fit(X, y)
    assert 'oob_score: must be boolean.' in str(excinfo.value)

    clf = DecisionForestClassifier(oob_score='abc')
    with pytest.raises(TypeError) as excinfo:
        clf.fit(X, y)
    assert 'oob_score: must be boolean.' in str(excinfo.value)

    # iris dataset

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4, random_state=0)

    clf = DecisionForestClassifier(bootstrap=True,
                                   oob_score=True,
                                   n_estimators=1,
                                   class_balance='balanced',
                                   max_depth=None,
                                   max_features=None,
                                   max_thresholds=None,
                                   random_state=0)

    clf.set_params(n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score > 0.9333333333333333 - precision and score < 0.9333333333333333 + precision
    score = clf.oob_score_
    assert score < precision

    clf = DecisionForestClassifier(bootstrap=True,
                                   oob_score=True,
                                   n_estimators=25,
                                   class_balance='balanced',
                                   max_depth=None,
                                   max_features=None,
                                   max_thresholds=None,
                                   random_state=0)

    clf.set_params(n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score > 0.95 - precision and score < 0.95 + precision
    score = clf.oob_score_
    assert score > 0.9555555555555556 - precision and score < 0.9555555555555556 + precision

# max_depth
# ---------

def test_fit_maxdepth():

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, max_depth='abc', random_state=0)
    with pytest.raises(TypeError):
        clf.fit(X, y)

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, max_depth=-999, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_depth' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, max_depth=0, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_depth' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, max_depth=1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [2, 8]; 0->1; 0->2; 1 [2, 3]; 2 [0, 5]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, max_depth=2, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [2, 8]; 0->1; 0->4; 1 X[1]<=0.5 [2, 3]; 1->2; 1->3; 2 [2, 0]; 3 [0, 3]; 4 [0, 5]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, max_depth=999, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [2, 8]; 0->1; 0->4; 1 X[1]<=0.5 [2, 3]; 1->2; 1->3; 2 [2, 0]; 3 [0, 3]; 4 [0, 5]; '
    assert data == data_target

# class_balance
# -------------

def test_fit_classbalance():

    clf = DecisionForestClassifier(bootstrap=False, class_balance=0, random_state=0)
    with pytest.raises(TypeError):
        clf.fit(X, y)

    clf = DecisionForestClassifier(bootstrap=False, class_balance='auto', random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'class_balance' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

# max_depth + class_balance
# -------------------------

def test_fit_maxdepth_classbalance():

    clf = DecisionForestClassifier(bootstrap=False, max_depth=2, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

# max_features
# ------------

def test_fit_maxfeatures():

    clf = DecisionForestClassifier(bootstrap=False, max_features=None, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    # integers

    clf = DecisionForestClassifier(bootstrap=False, max_features=-1, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, max_features=0, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, max_features=1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_features=2, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_features=4, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

    # floats

    clf = DecisionForestClassifier(bootstrap=False, max_features=-1.0, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, max_features=0.0, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, max_features=0.1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_features=0.67, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_features=1.0, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_features=1.1, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

    # strings

    clf = DecisionForestClassifier(bootstrap=False, max_features='xxx', random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, max_features='auto', random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_features='sqrt', random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_features='log2', random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    # misc

    clf = DecisionForestClassifier(bootstrap=False, max_features=[], random_state=0)
    with pytest.raises(TypeError) as excinfo:
        clf.fit(X, y)
    assert 'max_features' in str(excinfo.value)

# max_thresholds
# --------------

def test_fit_maxthresholds():

    # None

    clf = DecisionForestClassifier(bootstrap=False, max_thresholds=None, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.88]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.88]; 4 [0, 3.12]; '
    assert data == data_target

    # integers

    clf = DecisionForestClassifier(bootstrap=False, max_thresholds=0, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_thresholds' in str(excinfo.value)

    clf = DecisionForestClassifier(bootstrap=False, max_thresholds=1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.874 [5, 5]; 0->1; 0->8; 1 X[2]<=0.753 [5, 1.88]; 1->2; 1->5; 2 X[1]<=0.368 [2.5, 0.62]; 2->3; 2->4; 3 [2.5, 0]; 4 [0, 0.62]; 5 X[1]<=0.31 [2.5, 1.25]; 5->6; 5->7; 6 [2.5, 0]; 7 [0, 1.25]; 8 [0, 3.12]; '
    assert data == data_target

    clf = DecisionForestClassifier(bootstrap=False, max_thresholds=99, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X, y)
    assert 'max_thresholds' in str(excinfo.value)

    # misc

    clf = DecisionForestClassifier(bootstrap=False, max_thresholds=[], random_state=0)
    with pytest.raises(TypeError) as excinfo:
        clf.fit(X, y)
    assert 'max_thresholds' in str(excinfo.value)

# max_features and max_thresholds
# -------------------------------

def test_fit_maxfeatures_maxthresholds():

    # decision tree: max_features=None, max_thresholds=None ... covered before
    # random tree: max_features<n_features, max_thresholds=None ... covered before

    # extreme randomized tree: max_features<n_features, max_thresholds=1

    clf = DecisionForestClassifier(bootstrap=False, max_depth=2, max_features=2, max_thresholds=1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.874 [5, 5]; 0->1; 0->4; 1 X[2]<=0.563 [5, 1.88]; 1->2; 1->3; 2 [2.5, 0.62]; 3 [2.5, 1.25]; 4 [0, 3.12]; '
    assert data == data_target

    # totally randomized tree: max_features=1, max_thresholds=1

    clf = DecisionForestClassifier(bootstrap=False, max_depth=2, max_features=1, max_thresholds=1, random_state=0)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.874 [5, 5]; 0->1; 0->4; 1 X[2]<=0.753 [5, 1.88]; 1->2; 1->3; 2 [2.5, 0.62]; 3 [2.5, 1.25]; 4 [0, 3.12]; '
    assert data == data_target

# missing_values
# --------------

def test_fit_missingvalues():

    # training

    # - no NaN in y ever

    X_train_mv = np.array([
        [np.NaN],
        [np.NaN]
    ]).astype(np.double)
    y_train_mv = np.array([0, np.NaN])

    clf = DecisionForestClassifier(missing_values=None, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X_train_mv, y_train_mv)
    assert 'NaN' in str(excinfo.value)

    clf = DecisionForestClassifier(missing_values='NMAR', random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X_train_mv, y_train_mv)
    assert 'NaN' in str(excinfo.value)

    # - no NaN in X when missing values None

    y_train_mv = np.array([0, 1])

    clf = DecisionForestClassifier(missing_values=None, random_state=0)
    with pytest.raises(ValueError) as excinfo:
        clf.fit(X_train_mv, y_train_mv)
    assert 'NaN' in str(excinfo.value)

    # - only NaN(s)

    y_train_mv = np.array([0, 1])
    clf = DecisionForestClassifier(missing_values='NMAR', random_state=0)
    clf.fit(X_train_mv, y_train_mv)

    data = clf.estimators_[0].export_text()
    data_target = r'0 [1, 1]; '
    assert data == data_target

    # - 1 value : 0, 1 NaN : 1

    X_train_mv = np.array([
        [0],
        [np.NaN]
    ]).astype(np.double)
    y_train_mv = np.array([0, 1])

    clf = DecisionForestClassifier(missing_values='NMAR', random_state=0)
    clf.fit(X_train_mv, y_train_mv)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0] NA [1, 1]; 0->1; 0->2; 1 [0, 1]; 2 [1, 0]; '
    assert data == data_target

    # - 1 value : 0, 1 value and 1 NaN : 1

    X_train_mv = np.array([
        [0],
        [1],
        [np.NaN]
    ]).astype(np.double)
    y_train_mv = np.array([0, 1, 1])

    clf = DecisionForestClassifier(missing_values='NMAR', random_state=0)
    clf.fit(X_train_mv, y_train_mv)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 not NA [1.5, 1.5]; 0->1; 0->2; 1 [1.5, 0]; 2 [0, 1.5]; '
    assert data == data_target

    # testing

    # - simple dataset - no NaN(s) in training, all 1s are NaN(s) in testing

    X_train_mv = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]).astype(np.double)
    y_train_mv = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    X_test_mv = np.array([
        [0, 0, 0],
        [0, 0, np.NaN],
        [0, np.NaN, 0],
        [0, np.NaN, np.NaN],
        [np.NaN, 0, 0],
        [np.NaN, 0, np.NaN],
        [np.NaN, np.NaN, 0],
        [np.NaN, np.NaN, np.NaN]
    ]).astype(np.double)

    clf = DecisionForestClassifier(missing_values='NMAR', random_state=0)
    clf.fit(X_train_mv, y_train_mv)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [1, 1, 1, 1, 1, 1, 1, 1]; 0->1; 0->8; 1 X[1]<=0.5 [1, 1, 1, 1, 0, 0, 0, 0]; 1->2; 1->5; 2 X[2]<=0.5 [1, 1, 0, 0, 0, 0, 0, 0]; 2->3; 2->4; 3 [1, 0, 0, 0, 0, 0, 0, 0]; 4 [0, 1, 0, 0, 0, 0, 0, 0]; 5 X[2]<=0.5 [0, 0, 1, 1, 0, 0, 0, 0]; 5->6; 5->7; 6 [0, 0, 1, 0, 0, 0, 0, 0]; 7 [0, 0, 0, 1, 0, 0, 0, 0]; 8 X[2]<=0.5 [0, 0, 0, 0, 1, 1, 1, 1]; 8->9; 8->12; 9 X[1]<=0.5 [0, 0, 0, 0, 1, 0, 1, 0]; 9->10; 9->11; 10 [0, 0, 0, 0, 1, 0, 0, 0]; 11 [0, 0, 0, 0, 0, 0, 1, 0]; 12 X[1]<=0.5 [0, 0, 0, 0, 0, 1, 0, 1]; 12->13; 12->14; 13 [0, 0, 0, 0, 0, 1, 0, 0]; 14 [0, 0, 0, 0, 0, 0, 0, 1]; '
    assert data == data_target

    predict_proba = clf.predict_proba(X_test_mv)
    predict_proba_target = [
         [ 1.,     0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
         [ 0.5,    0.5,    0.,     0.,     0.,     0.,     0.,     0.   ],
         [ 0.5,    0.,     0.5,    0.,     0.,     0.,     0.,     0.   ],
         [ 0.25,   0.25,   0.25,   0.25,   0.,     0.,     0.,     0.   ],
         [ 0.5,    0.,     0.,     0.,     0.5,    0.,     0.,     0.   ],
         [ 0.25,   0.25,   0.,     0.,     0.25,   0.25,   0.,     0.   ],
         [ 0.25,   0.,     0.25,   0.,     0.25,   0.,     0.25,   0.   ],
         [ 0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125]
    ]

    for a, b in zip(predict_proba, predict_proba_target):
        for ai, bi in zip(a, b):
            assert ai > bi - precision and ai < bi + precision

    # - simple dataset - NaN(s) in training replacing some 1s, all 1s are NaN(s) in testing

    X_train_mv = np.array([
        [0, 0, 0],
        [0, 0, np.NaN],
        [0, np.NaN, 0],
        [0, np.NaN, np.NaN],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]).astype(np.double)
    y_train_mv = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    X_test_mv = np.array([
        [0, 0, 0],
        [0, 0, np.NaN],
        [0, np.NaN, 0],
        [0, np.NaN, np.NaN],
        [np.NaN, 0, 0],
        [np.NaN, 0, np.NaN],
        [np.NaN, np.NaN, 0],
        [np.NaN, np.NaN, np.NaN]
    ]).astype(np.double)

    clf = DecisionForestClassifier(missing_values='NMAR', random_state=0)
    clf.fit(X_train_mv, y_train_mv)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[0]<=0.5 [1, 1, 1, 1, 1, 1, 1, 1]; 0->1; 0->8; 1 X[1] NA [1, 1, 1, 1, 0, 0, 0, 0]; 1->2; 1->5; 2 X[2] NA [0, 0, 1, 1, 0, 0, 0, 0]; 2->3; 2->4; 3 [0, 0, 0, 1, 0, 0, 0, 0]; 4 [0, 0, 1, 0, 0, 0, 0, 0]; 5 X[2] NA [1, 1, 0, 0, 0, 0, 0, 0]; 5->6; 5->7; 6 [0, 1, 0, 0, 0, 0, 0, 0]; 7 [1, 0, 0, 0, 0, 0, 0, 0]; 8 X[2]<=0.5 [0, 0, 0, 0, 1, 1, 1, 1]; 8->9; 8->12; 9 X[1]<=0.5 [0, 0, 0, 0, 1, 0, 1, 0]; 9->10; 9->11; 10 [0, 0, 0, 0, 1, 0, 0, 0]; 11 [0, 0, 0, 0, 0, 0, 1, 0]; 12 X[1]<=0.5 [0, 0, 0, 0, 0, 1, 0, 1]; 12->13; 12->14; 13 [0, 0, 0, 0, 0, 1, 0, 0]; 14 [0, 0, 0, 0, 0, 0, 0, 1]; '
    assert data == data_target

    predict_proba = clf.predict_proba(X_test_mv)
    predict_proba_target = [
        [0.75,       0.04333333, 0.04333333, 0.,         0.1,        0.02666667,  0.03666667, 0.        ],
        [0.04333333, 0.93333333, 0.,         0.,         0.02333333, 0.,          0.,         0.        ],
        [0.04333333, 0.,         0.94333333, 0.,         0.01333333, 0.,          0.,         0.        ],
        [0.,         0.,         0.,         1.,         0.,         0.,          0.,         0.        ],
        [0.37333333, 0.03333333, 0.02833333, 0.,         0.37333333, 0.08666667,  0.105,      0.        ],
        [0.03333333, 0.65833333, 0.,         0.,         0.1525,     0.12416667,  0.01833333, 0.01333333],
        [0.02833333, 0.,         0.69666667, 0.,         0.13583333, 0.01333333,  0.1175,     0.00833333],
        [0.,         0.,         0.,         0.855 ,     0.03625,    0.03625,     0.03625,    0.03625   ]
    ]
    for a, b in zip(predict_proba, predict_proba_target):
        for ai, bi in zip(a, b):
            assert ai > bi - precision and ai < bi + precision

# random_state
# ------------

def test_fit_randomstate():

    # integers

    clf = DecisionForestClassifier(bootstrap=False, max_features='auto', random_state=-1)
    with pytest.raises(OverflowError) as excinfo:
        clf.fit(X, y)

    clf = DecisionForestClassifier(bootstrap=False, max_depth=2, max_features=1, max_thresholds=1, random_state=999)
    clf.fit(X, y)

    data = clf.estimators_[0].export_text()
    data_target = r'0 X[2]<=0.929 [5, 5]; 0->1; 0->4; 1 X[1]<=0.249 [2.5, 3.12]; 1->2; 1->3; 2 [2.5, 2.5]; 3 [0, 0.62]; 4 X[1]<=0.21 [2.5, 1.88]; 4->5; 4->6; 5 [2.5, 0]; 6 [0, 1.88]; '
    assert data == data_target

    # misc

    clf = DecisionForestClassifier(bootstrap=False, max_features='auto', random_state=[])
    with pytest.raises(TypeError) as excinfo:
        clf.fit(X, y)

# DecisionForestClassifier.predict_proba()
# ========================================

def test_predict_proba():

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    clf.fit(X, y)

    predict_proba = clf.predict_proba(X_test)
    predict_proba_target = [
        [1., 0.],
        [1., 0.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0.38, 0.62],
        [0., 1.],
        [0., 1.]
    ]

    for a, b in zip(predict_proba, predict_proba_target):
        for ai, bi in zip(a, b):
            assert ai > bi - precision and ai < bi + precision

    # not fitted

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    with pytest.raises(NotFittedError):
        predict_proba = clf.predict_proba(X_test)

# class_balance
# -------------

def test_predict_proba_classbalance():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    predict_proba = clf.predict_proba(X_test)
    predict_proba_target = [
        [1., 0.],
        [1., 0.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0.38, 0.62],
        [0., 1.],
        [0., 1.]
    ]

    for a, b in zip(predict_proba, predict_proba_target):
        for ai, bi in zip(a, b):
            assert ai > bi - precision and ai < bi + precision

# DecisionForestClassifier.predict()
# ==================================

def test_predict():

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    clf.fit(X, y)

    predict = clf.predict(X_test)
    predict_target = [0, 0, 1, 1, 1, 1, 1, 1]

    for a, b in zip(predict, predict_target):
        assert a > b - precision and a < b + precision

    # not fitted

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    with pytest.raises(NotFittedError):
        predict = clf.predict(X_test)

# class_balance
# -------------

def test_predict_classbalance():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    predict = clf.predict(X_test)
    predict_target = [0, 0, 1, 1, 1, 1, 1, 1]

    for a, b in zip(predict, predict_target):
        assert a > b - precision and a < b + precision

# DecisionForestClassifier.feature_importances_
# =============================================

def test_feature_importances():

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, random_state=0)
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.46291667, 0.46802083, 0.0690625]
    for a, b in zip(feature_importances, feature_importances_target):
        assert a > b - precision

    # not fitted

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, random_state=0)
    with pytest.raises(NotFittedError):
        feature_importances = clf.feature_importances_

# class_balance
# -------------

def test_feature_importances_classbalance():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    feature_importances_target = [0.51302453, 0.44893362, 0.03804185]
    for a, b in zip(feature_importances, feature_importances_target):
        assert a > b - precision

# DecisionForestClassifier.estimators_[i].export_graphviz()
# ========================================================

def test_export_graphviz():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    dot_data = clf.estimators_[0].export_graphviz()
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'0 [label="X[0] <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="X[1] <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\n0", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

# feature_names
# -------------

def test_export_graphviz_featurenames():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    with pytest.raises(TypeError):
        dot_data = clf.estimators_[0].export_graphviz(feature_names=0)

    with pytest.raises(IndexError):
        dot_data = clf.estimators_[0].export_graphviz(feature_names=[ ])

    with pytest.raises(IndexError):
        dot_data = clf.estimators_[0].export_graphviz(feature_names=["f1"])

    dot_data = clf.estimators_[0].export_graphviz(feature_names=["f1", "f2", "f3"])
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'0 [label="f1 <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="f2 <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\n0", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

    dot_data = clf.estimators_[0].export_graphviz(feature_names=["f1", "f2", "f3", "f4"])
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'0 [label="f1 <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="f2 <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\n0", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

# class_names
# -----------

def test_export_graphviz_classnames():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    with pytest.raises(TypeError):
        dot_data = clf.estimators_[0].export_graphviz(class_names=0)

    with pytest.raises(IndexError):
        dot_data = clf.estimators_[0].export_graphviz(class_names=[ ])

    with pytest.raises(IndexError):
        dot_data = clf.estimators_[0].export_graphviz(class_names=['A'])

    dot_data = clf.estimators_[0].export_graphviz(class_names=['A', 'B'])
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'0 [label="X[0] <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="X[1] <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\nA", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

    dot_data = clf.estimators_[0].export_graphviz(class_names=['A', 'B', 'C'])
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'0 [label="X[0] <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="X[1] <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\nA", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

# rotate
# ------

def test_export_graphviz_rotate():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)

    dot_data = clf.estimators_[0].export_graphviz(rotate=True)
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'rankdir=LR ;' '\n' \
        r'0 [label="X[0] <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="X[1] <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\n0", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

    dot_data = clf.estimators_[0].export_graphviz(rotate=False)
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'0 [label="X[0] <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="X[1] <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\n0", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\n1", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

# feature_names + class_names
# ---------------------------

def test_export_graphviz_featurenames_classnames():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)
    dot_data = clf.estimators_[0].export_graphviz(feature_names=["f1", "f2", "f3"],
                                                  class_names=['A', 'B'])
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'0 [label="f1 <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="f2 <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\nA", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

# feature_names + class_names + rotate
# ------------------------------------

def test_export_graphviz_featurenames_classnames_rotate():

    clf = DecisionForestClassifier(bootstrap=False, class_balance='balanced', random_state=0)
    clf.fit(X, y)
    dot_data = clf.estimators_[0].export_graphviz(feature_names=["f1", "f2", "f3"],
                                                  class_names=['A', 'B'],
                                                  rotate=True)
    dot_data_target = \
        r'digraph Tree {' '\n' \
        r'node [shape=box, style="rounded, filled", color="black", fontname=helvetica, fontsize=14] ;' '\n' \
        r'edge [fontname=helvetica, fontsize=12] ;' '\n' \
        r'rankdir=LR ;' '\n' \
        r'0 [label="f1 <= 0.5\np(class) = [0.5, 0.5]\nclass, n = 10", fillcolor="#FF000000"] ;' '\n' \
        r'0 -> 1 [penwidth=6.875000, headlabel="True", labeldistance=2.5, labelangle=-45] ;' '\n' \
        r'0 -> 4 [penwidth=3.125000] ;' '\n' \
        r'1 [label="f2 <= 0.5\n[0.73, 0.27]", fillcolor="#FF000034"] ;' '\n' \
        r'1 -> 2 [penwidth=5.000000] ;' '\n' \
        r'1 -> 3 [penwidth=1.875000] ;' '\n' \
        r'2 [label="[1, 0]\nA", fillcolor="#FF0000FF"] ;' '\n' \
        r'3 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'4 [label="[0, 1]\nB", fillcolor="#00FF00FF"] ;' '\n' \
        r'}'
    assert dot_data == dot_data_target

# Extreme Data
# ============

# Empty X, y training data
# ------------------------

def test_empty_Xy_train():

    X_train = np.array([]).astype(np.double).reshape(1, -1)
    y_train = np.array([])

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    with pytest.raises(ValueError):
        clf.fit(X_train, y_train)

# 1 X, y training data
# --------------------

def test_1_Xy_train():

    X_train = np.array([[0, 0, 0]]).astype(np.double).reshape(1, -1)
    y_train = np.array([0])

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    clf.fit(X_train, y_train)

    data = clf.estimators_[0].export_text()
    data_target = r'0 [1]; '
    assert data == data_target

    X_test = np.array([[1, 1, 1]]).astype(np.double).reshape(1, -1)
    predict = clf.predict(X_test)
    predict_target = [0]
    for a, b in zip(predict, predict_target):
        assert a > b - precision and a < b + precision

# All X = 0 training data
# -----------------------

def test_X_0_train():

    X_train = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).astype(np.double)
    y_train = np.array([0, 1, 1])

    clf = DecisionForestClassifier(bootstrap=False, class_balance=None, random_state=0)
    clf.fit(X_train, y_train)

    data = clf.estimators_[0].export_text()
    data_target = r'0 [1, 2]; '
    assert data == data_target

    predict = clf.predict(X_train)
    predict_target = [1, 1, 1]
    for a, b in zip(predict, predict_target):
        assert a > b - precision and a < b + precision

# All y = 0 training data
# -----------------------

def test_y_0_train():

    X_train = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]).astype(np.double)
    y_train = np.array([0, 0, 0])

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    clf.fit(X_train, y_train)

    data = clf.estimators_[0].export_text()
    data_target = r'0 [3]; '
    assert data == data_target

    predict = clf.predict(X_train)
    predict_target = [0, 0, 0]
    for a, b in zip(predict, predict_target):
        assert a > b - precision and a < b + precision

# Number of classes very large
# ----------------------------
# code coverage for duplication of offset_list in create_rgb_LUT in export_graphviz( )

def test_numberclasses_large():

    n_classes = 97 # max number of colors = 96
    X_train = np.array(range(n_classes)).astype(np.double).reshape(-1,1)
    y_train = np.array(range(n_classes))

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    clf.fit(X_train, y_train)
    dot_data = clf.estimators_[0].export_graphviz()
    # no error raised

# Mismatch number of features
# ---------------------------

def test_mismatch_nfeatures():

    X_train = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]).astype(np.double)
    y_train = np.array([0, 1, 2])

    X_test = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]).astype(np.double)
    y_test = np.array([0, 1, 2])

    clf = DecisionForestClassifier(bootstrap=False, random_state=0)
    clf.fit(X_train, y_train)

    with pytest.raises(ValueError) as excinfo:
        predict = clf.predict(X_test)
    assert 'number of features' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        predict_proba = clf.predict_proba(X_test)
    assert 'number of features' in str(excinfo.value)
