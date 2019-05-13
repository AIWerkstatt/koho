# encoding=utf-8
""" Decision Forest module.

- Classification
- n Decision Trees with soft voting
- Important Features

Python interface compatible with scikit-learn.
"""

# Author: AI Werkstatt (TM)
# (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

# Scikit-learn compatible
# http://scikit-learn.org/stable/developers
# Trying to be consistent with scikit-learn's ensemble module
# https://github.com/scikit-learn/scikit-learn
# Basic concepts for the implementation of the classifier are based on
# G. Louppe, “Understanding Random Forests”, PhD Thesis, 2014

import numbers
import numpy as np
from warnings import warn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils._joblib import Parallel, delayed

from ._decision_tree import DecisionTreeClassifier
# Cython binding for C++ implementation
from ._decision_tree_cpp import RandomState

# ==============================================================================
# Decision Forest Classifier
# ==============================================================================


def _DecisionForestClassifier_bagging_fit_and_oob_score(estimator, oob_score, X, y, n_samples, n_classes, data_seed):
    """ Parallel processing helper function for DecisionForestClassifier's fit function.
    """

    # Build a decision tree from the bootstrapped training data
    # drawing random samples with replacement
    random_state = RandomState(data_seed)
    idx = random_state.randint(0, n_samples, size=n_samples)  # includes 0, excludes n_samples
    # make sure training data includes all classes
    while np.unique(y[idx]).shape[0] < n_classes:
        idx = random_state.randint(0, n_samples, size=n_samples)
    estimator.fit(X[idx, :], y[idx])

    # Compute Out-Of-Bag estimates
    # as average error for all samples when not included in bootstrap
    p = np.zeros((n_samples, n_classes))
    if oob_score:
        unsampled_idx = np.bincount(idx, minlength=n_samples) == 0
        p[unsampled_idx, :] = estimator.predict_proba(X[unsampled_idx, :])

    return estimator, p


class DecisionForestClassifier(BaseEstimator, ClassifierMixin):
    """ A decision forest classifier.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of decision trees in the forest.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.
        Out-of-bag samples are used to estimate the generalization accuracy.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    class_balance : str 'balanced' or None, optional (default='balanced')
        Weighting of the classes.

            - If 'balanced', then the values of y are used to automatically adjust class weights
              inversely proportional to class frequencies in the input data.
            - If None, all classes are supposed to have weight one.

    max_depth : int or None, optional (default=3)
        The maximum depth of the tree.

            - If None, the depth of the tree is expanded until all leaves
              are pure or no further impurity improvement can be achieved.

    max_features : int, float, str or None, optional (default=None)
        The number of random features to consider when looking for the best split at each node.

            - If int, then consider ``max_features`` features.
            - If float, then ``max_features`` is a percentage and
              int(``max_features`` * n_features) features are considered.
            - If 'auto', then ``max_features`` = sqrt(n_features).
            - If 'sqrt', then ``max_features`` = sqrt(n_features).
            - If 'log2', then ``max_features`` = log2(n_features).
            - If None, then ``max_features`` = n_features considering all features in random order.

        Note: the search for a split does not stop until at least
        one valid partition of the node samples is found up to the point that
        all features have been considered,
        even if it requires to effectively inspect more than ``max_features`` features.

        `Decision Tree`: ``max_features`` = None and ``max_thresholds`` = None

        `Random Tree`: ``max_features`` < n_features and ``max_thresholds`` = None

    max_thresholds : int 1 or None, optional (default=None)
        The number of random thresholds to consider when looking for the best split at each node.

            - If 1, then consider 1 random threshold, based on the `Extreme Randomized Tree` formulation.
            - If None, then all thresholds, based on the mid-point of the node samples, are considered.

        `Extreme Randomized Trees (ET)`: ``max_thresholds`` = 1

        `Totally Randomized Trees`: ``max_features`` = 1 and ``max_thresholds`` = 1,
        very similar to `Perfect Random Trees (PERT)`.

    missing_values : str 'NMAR' or None, optional (default=None)
        Handling of missing values, defined as np.NaN.

            - If 'NMAR' (Not Missing At Random), then during training: the split criterion considers missing values
              as another category and samples with missing values are passed to either the left or the right child
              depending on which option provides the best split,
              and then during testing: if the split criterion includes missing values,
              a missing value is dealt with accordingly (passed to left or right child),
              or if the split criterion does not include missing values,
              a missing value at a split criterion is dealt with by combining the results from both children
              proportionally to the number of samples that are passed to the children during training.
            - If None, an error is raised if one of the features has a missing value.
              An option is to use imputation (fill-in) of missing values prior to using the decision tree classifier.

    random_state : int or None, optional (default=None)
        A random state to control the pseudo number generation and repetitiveness of fit().

            - If int, random_state is the seed used by the random number generator;
            - If None, the random number generator is seeded with the current system time.

    n_jobs : int, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.

            - None means 1.
            - If -1, then the number of jobs is set to the number of cores.

    Attributes
    ----------
    classes_ : array, shape = [n_classes]
        The classes labels.

    n_classes_ : int
        The number of classes.

    n_features_ : int
        The number of features.

    estimators_ : list of tree objects from DecisionTreeClassifier
        The collection of the underlying sub-estimators.

    feature_importances_ : array, shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    """

    # We use 'class_balance' as the hyperparameter name instead of “class_weight”
    # The “class_weight” hyperparameter name is recognized by 'check_estimator()'
    # and the test “check_class_weight_ classifiers()” is performed that uses the
    # dict parameter and requires for a decision tree the “min_weight_fraction_leaf”
    # hyperparameter to be implemented to pass the test.

    def __init__(self,
                 n_estimators=100,
                 bootstrap=False,
                 oob_score=False,
                 class_balance='balanced',  # pass through hyperparameter to decision trees
                 max_depth=3,  # pass through hyperparameter to decision trees
                 max_features='auto',
                 max_thresholds=None,
                 missing_values=None,
                 random_state=None,
                 n_jobs=None):
        """ Create a new decision forest classifier and initialize it with hyperparameters.
        """

        # Hyperparameters
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_balance = class_balance
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.missing_values = missing_values
        # Random Number Generator
        self.random_state = random_state
        # Parallel Processing
        self.n_jobs = n_jobs

        return

    def fit(self, X, y):
        """ Build a decision forest classifier based on decision tree classifiers from the training data.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            The training input samples.

        y : array, shape = [n_samples]
            The target class labels corresponding to the training input samples.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check and prepare data
        # ----------------------

        # Check X, y

        if self.missing_values == 'NMAR':
            X, y = check_X_y(X, y, dtype=np.float64, order="C", force_all_finite='allow-nan')
        else:
            X, y = check_X_y(X, y, dtype=np.float64, order="C")

        # Determine attributes from training data

        self.classes_ = unique_labels(y)  # Keep to raise required ValueError tested by 'check_estimator()'
        self.classes_, y = np.unique(y, return_inverse=True)  # Encode y from classes to integers
        self.n_classes_ = self.classes_.shape[0]
        n_samples, self.n_features_ = X.shape

        # Check hyperparameters (here, not in __init__)

        # n estimators

        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise TypeError("n_estimators: must be an integer.")

        if self.n_estimators < 1:
            raise ValueError("n_estimators: %s < 1, "
                             "but a decsion forest requires to have at least 1 decision tree."
                             % self.n_estimators)

        # bootstrap

        if not isinstance(self.bootstrap, bool):
            raise TypeError("bootstrap: must be boolean.")

        # oob score

        if not isinstance(self.oob_score, bool):
            raise TypeError("oob_score: must be boolean.")

        if not self.bootstrap and self.oob_score:
            raise ValueError("oob_score: only available if bootstrap=True")

        # Random Number Generator

        random_state = RandomState(self.random_state)

        # Create explicitly different seeds for the decision trees
        # to avoid building the same tree over and over again for the entire decision forest
        # when decision trees are build in parallel.
        algo_seeds = random_state.randint(0, random_state.MAX_INT, size=self.n_estimators)

        # Build a decision forest
        # -----------------------

        # Instantiate decision trees
        estimators = []
        for e in range(self.n_estimators):
            estimator = DecisionTreeClassifier(class_balance=self.class_balance,
                                               max_depth=self.max_depth,
                                               max_features=self.max_features,
                                               max_thresholds=self.max_thresholds,
                                               missing_values=self.missing_values,
                                               random_state=algo_seeds[e])
            estimators.append(estimator)

        # Build decision trees from the training data
        if not self.bootstrap:

            # embarrassing parallelism

            # for estimator in estimators:
            #     estimator.fit(X, y)

            estimators = Parallel(n_jobs=self.n_jobs) \
                (delayed(estimator.fit)(X, y) for estimator in estimators)

        else:  # Bagging & Out_Of-Bag estimate

            # Different seeds for algorithm and data (bagging)
            # to avoid building the same trees multiple times
            # when the same seed comes up again.
            data_seeds = random_state.randint(0, random_state.MAX_INT, size=self.n_estimators)

            # embarrassing parallelism
            ps = []
            n_classes = np.unique(y).shape[0]

            # for estimator, data_seed in zip(estimators, data_seeds):
            #     # Build a decision tree from the bootstrapped training data
            #     # drawing random samples with replacement
            #     random_state = np.random.RandomState(data_seed)
            #     idx = random_state.randint(0, n_samples, size=n_samples)  # includes 0, excludes n_samples
            #     # make sure training data includes all classes
            #     while np.unique(y[idx]).shape[0] < n_classes:
            #         idx = random_state.randint(0, n_samples, size=n_samples)
            #     estimator.fit(X[idx, :], y[idx])
            #     # Compute Out-Of-Bag estimates
            #     # as average error for all samples when not included in bootstrap
            #     p = np.zeros((n_samples, self.n_classes_))
            #     if self.oob_score:
            #         unsampled_idx = np.bincount(idx, minlength=n_samples) == 0
            #         p[unsampled_idx, :] = estimator.predict_proba(X[unsampled_idx, :])
            #     ps.append(p)

            out = Parallel(n_jobs=self.n_jobs) \
                (delayed(_DecisionForestClassifier_bagging_fit_and_oob_score)  # helper function
                 (estimator, self.oob_score, X, y, n_samples, n_classes, data_seed)
                 for estimator, data_seed in zip(estimators, data_seeds))
            estimators, ps = zip(*out)

            if self.oob_score:
                ps = np.array(ps)
                class_probabilities = np.sum(ps, axis=0)  # no normalization needed when using argmax( )
                predictions = np.argmax(class_probabilities, axis=1)
                valid_idx = np.sum(class_probabilities, axis=1) > 0.0  # samples that have an oob score
                oob_n_samples = np.sum(valid_idx)
                if oob_n_samples == n_samples:
                    self.oob_score_ = np.mean(y[valid_idx] == predictions[valid_idx])
                else:
                    self.oob_score_ = 0.0
                    warn("Only %s out of %s samples have an out-of-bag estimate. "
                         "This probably means too few estimators were used "
                         "to compute any reliable oob estimates."
                         % (oob_n_samples, n_samples))

        self.estimators_ = estimators

        # Return the classifier
        return self

    def predict(self, X):
        """ Predict classes for the test data using soft voting.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            The test input samples.

        Returns
        -------
        y : array, shape = [n_samples]
            The predicted classes for the test input samples.
        """

        # Soft voting using
        # predicted class probabilities
        class_probablities = self.predict_proba(X)

        # Determine class based on highest classes probabilities
        predictions = np.argmax(class_probablities, axis=1)

        # Decode y back from integers to classes
        return self.classes_.take(predictions, axis=0)

    def predict_proba(self, X):
        """ Predict classes probabilities for the test data.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            The test input samples.

        Returns
        -------
        p : array, shape = [n_samples, n_classes]
            The predicted classes probablities for the test input samples.
        """

        # Check that fit has been called
        check_is_fitted(self, ['estimators_'])

        # Check X
        if self.missing_values == 'NMAR':
            X = check_array(X, dtype=np.float64, order="C", force_all_finite='allow-nan')
        else:
            X = check_array(X, dtype=np.float64, order="C")

        n_samples, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError("X: number of features %s != number of features of the model %s, "
                             "must match."
                             % (n_features, self.n_features_))

        # Predict class probabilities for all decision trees

        # embarrassing parallelism
        ps = []

        # for estimator in self.estimators_:
        #     p = estimator.predict_proba(X)
        #     ps.append(p)

        ps = Parallel(n_jobs=self.n_jobs)\
            (delayed(estimator.predict_proba)(X) for estimator in self.estimators_)

        # Predict classes probabilities for the decision forest
        # as average of the class probabilities from all decision trees
        return sum(ps) / len(self.estimators_)  # reduce

    @property
    def feature_importances_(self):
        """ Get feature importances from the decision forest.
        """

        # Check that fit has been called
        check_is_fitted(self, ['estimators_'])

        # Calculate feature importances for all decision trees

        # embarrassing parallelism
        fis = []

        # for estimator in self.estimators_:
        #     fi = estimator.feature_importances_
        #     fis.append(fi)

        fis = Parallel(n_jobs=self.n_jobs) \
            (delayed(getattr)(estimator, 'feature_importances_') for estimator in self.estimators_)

        # Calculate feature importances for the decision forest
        # as average of feature importances from all decision trees
        return sum(fis) / len(self.estimators_)  # reduce
