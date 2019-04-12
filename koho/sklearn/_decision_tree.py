# encoding=utf-8
""" Decision Tree module.

- Classification
- Numerical (dense) data
- Class balancing
- Multi-Class
- Single-Output
- Build order: depth first
- Impurity criteria: gini
- Split a. features: best over k (incl. all) random features
- Split b. thresholds: 1 random or all thresholds
- Stop criteria: max depth, (pure, no improvement)
- Important Features
- Export Graph

Implementation Optimizations:
stack, samples LUT with in-place partitioning, incremental histogram updates

Python interface compatible with scikit-learn.
"""

# Author: AI Werkstatt (TM)
# (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

# Compliant with scikit-learn's developer's guide:
# http://scikit-learn.org/stable/developers
# trying to be consistent with scikit-learn's sklearn.tree module implementation
# https://github.com/scikit-learn/scikit-learn
# which is further documented in
# G. Louppe, “Understanding Random Forests”, PhD Thesis, 2014
# and from which the basic principles are implemented.

import numbers
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_random_state, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from io import StringIO

from ._decision_tree_python import Tree, DepthFirstTreeBuilder

# ==============================================================================
# Decision Tree Classifier
# ==============================================================================


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """ A decision tree classifier.

    Parameters
    ----------
    class_balance : string 'balanced' or None, optional (default='balanced')
        Weighting of the classes.

            - If 'balanced', then the values of y are used to automatically adjust class weights
              inversely proportional to class frequencies in the input data.
            - If None, all classes are supposed to have weight one.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree.

            The depth of the tree is expanded until the specified maximum depth of the tree is reached
            or all leaves are pure or no further impurity improvement can be achieved.
            - If None, the maximum depth of the tree is set to max long (2^31-1).

    max_features : int, float, string or None, optional (default=None)
        Note: only to be used by Decision Forest

        The number of random features to consider when looking for the best split at each node.

            - If int, then consider `max_features` features.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features` considering all features in random order.

        Note: the search for a split does not stop until at least
        one valid partition of the node samples is found up to the point that
        all features have been considered,
        even if it requires to effectively inspect more than ``max_features`` features.

        `Decision Tree`: ``max_features=None`` and ``max_thresholds=None``

        `Random Tree`: ``max_features<n_features`` and ``max_thresholds=None``

    max_thresholds : 1 or None, optional (default=None)
        Note: only to be used by Decision Forest

        The number of random thresholds to consider when looking for the best split at each node.

            - If 1, then consider 1 random threshold, based on the `Extreme Randomized Tree` formulation.
            - If None, then all thresholds, based on the mid-point of the node samples, are considered.

        `Extreme Randomized Trees (ET)`: ``max_thresholds=1``

        `Totally Randomized Trees`: ``max_features=1`` and ``max_thresholds=1``,
        very similar to `Perfect Random Trees (PERT)`.

    random_state : int or None, optional (default=None)
        Note: only to be used by Decision Forest

        A random state to control the pseudo number generation and repetitiveness of fit().

            - If int, random_state is the seed used by the random number generator;
            - If None, the random number generator is seeded with the current system time.

    Attributes
    ----------
    classes_ : array, shape = [n_classes]
        The classes labels.

    n_classes_ : int
        The number of classes.

    n_features_ : int
        The number of features.

    max_features_ : int,
        The inferred value of max_features.

    tree_ : tree object
        The underlying estimator.

    feature_importances_ : array, shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.
    """

    # We use "class_balance" as the hyperparameter name instead of “class_weight”
    # The “class_weight” hyperparameter name is recognized by "check_estimator()"
    # and the test “check_class_weight_ classifiers()” is performed that uses the
    # dict parameter and requires for a decision tree the “min_weight_fraction_leaf”
    # hyperparameter to be implemented to pass the test.

    def __init__(self,
                 class_balance='balanced',
                 max_depth=None,
                 max_features=None,
                 max_thresholds=None,
                 random_state=None):
        """Create a new decision tree classifier and initialize it with hyperparameters.
        """

        # Hyperparameters
        self.class_balance = class_balance
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        # Random State
        self.random_state = random_state

        return

    def fit(self, X, y):
        """Build a decision tree classifier from the training data.

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

        X, y = check_X_y(X, y)

        # Determine attributes from training data

        self.classes_ = unique_labels(y)  # Keep to raise required ValueError tested by "check_estimator()"
        self.classes_, y = np.unique(y, return_inverse=True)  # Encode y from classes to integers
        self.n_classes_ = self.classes_.shape[0]
        n_samples, self.n_features_ = X.shape

        # Calculate class weights
        # so that n_samples == sum of all weighted samples
        # Note that scikit-learn provides:
        # "compute_class_weight(self.class_balance, self.classes_, y)"

        mean_samples_per_class = y.shape[0] / self.n_classes_
        if self.class_balance is not None:
            if isinstance(self.class_balance, str):
                if self.class_balance in ['balanced']:
                    # The 'balanced' mode uses the values of y to
                    # automatically adjust weights inversely proportional
                    # to class frequencies in the input data.
                    class_weight = mean_samples_per_class / np.bincount(y)
                else:
                    raise ValueError("class_balance: unsupported string \'%s\', "
                                     "only 'balanced' is supported."
                                     % self.class_balance)
            else:
                raise TypeError("class_balance: %s is not supported."
                                 % self.class_balance)
        else:
            class_weight = np.ones(self.classes_.shape[0], dtype=np.float64)

        # Check hyperparameters (here, not in __init__)

        # max depth

        if self.max_depth is not None:
            if not isinstance(self.max_depth, (numbers.Integral, np.integer)):
                raise TypeError("max_depth: must be an integer.")

        max_depth = self.max_depth if self.max_depth is not None else (2 ** 31) - 1

        if max_depth < 1:
            raise ValueError("max_depth: %s < 1, "
                             "but a decision tree requires to have at least a root node."
                             % max_depth)

        # max features

        if self.max_features is not None:
            if isinstance(self.max_features, str):
                if self.max_features in ['auto', 'sqrt']:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                elif self.max_features in ['log2']:
                    max_features = max(1, int(np.log2(self.n_features_)))
                else:
                    raise ValueError("max_features: unsupported string \'%s\', "
                                     "only 'auto', 'sqrt' and 'log2' are supported."
                                     % self.max_features)
            elif isinstance(self.max_features, (numbers.Integral, np.integer)):
                if self.max_features > 0:
                    max_features = self.max_features
                else:
                    raise ValueError("max_features: %s < 1, "
                                     "but a spit requires to consider a least 1 feature."
                                     % self.max_features)
            elif isinstance(self.max_features, (numbers.Real, np.float)):
                if self.max_features > 0.0:
                    if self.max_features <= 1.0:
                        max_features = max(1,
                                           min(int(self.max_features * self.n_features_),
                                               self.n_features_))
                    else:
                        raise ValueError("max_features: %s > 1.0, "
                                         "only floats <= 1.0 are supported."
                                         % self.max_features)
                else:
                    raise ValueError("max_features: %s <= 0.0, "
                                     "only floats > 0.0 are supported."
                                     % self.max_features)
            else:
                raise TypeError("max_features: %s is not supported, "
                                "only 'None', strings: 'auto', 'sqrt', 'log2', integers and floats are supported."
                                % self.max_features)
        else:
            max_features = self.n_features_

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features: %s not in (0, n_features]"
                             % max_features)

        self.max_features_ = max_features

        # max thresholds

        max_thresholds = None
        if self.max_thresholds is not None:
            if isinstance(self.max_thresholds, (numbers.Integral, np.integer)):
                if self.max_thresholds == 1:
                    max_thresholds = 1
                else:
                    raise ValueError("max_thresholds: %s != 1, "
                                     "only 1 is supported."
                                     % self.max_thresholds)
            else:
                raise TypeError("max_thresholds: %s is not supported, "
                                "only 'None' and '1' are supported."
                                % self.max_thresholds)
        else:
            max_thresholds = 0

        # Random Number Generator

        random_state = check_random_state(self.random_state)

        # Build decision tree
        # -------------------

        # Initialize the tree builder
        builder = DepthFirstTreeBuilder(
                        self.n_classes_, self.n_features_, n_samples, class_weight,
                        max_depth, max_features, max_thresholds, random_state)

        # Create an empty tree
        self.tree_ = Tree(self.n_classes_, self.n_features_)

        # Build a decision tree from the training data X, y
        builder.build(self.tree_, X, y)

        # Return the classifier
        return self

    def predict(self, X):
        """ Predict classes for the test data.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            The test input samples.

        Returns
        -------
        y : array, shape = [n_samples]
            The predicted classes for the test input samples.
        """

        # Predict classes probabilities
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
        check_is_fitted(self, ['tree_'])

        # Check X
        X = check_array(X)

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("X: number of features %s != number of features of the model %s, "
                             "must match."
                             % (n_features, self.n_features_))

        # Predict classes probabilities
        proba = self.tree_.predict(X)

        return proba

    @property
    def feature_importances_(self):
        """ Get feature importances from the decision tree.
        """

        # Check that fit has been called
        check_is_fitted(self, ['tree_'])

        # Calculate feature importances for the decision tree
        return self.tree_.calculate_feature_importances()

    def export_graphviz(self, feature_names=None, class_names=None, rotate=False):
        """Export of a decision tree in GraphViz dot format.

        Parameters
        ----------
        feature_names : list of strings, optional (default=None)
            Names of each of the features.

        class_names : list of strings, optional (default=None)
            Names of each of the classes in ascending numerical order.
            Classes are represented as integers: 0, 1, ... (n_classes-1).
            If y consists of class labels, those class labels need to be provided as class_names again.

        rotate : bool, optional (default=False)
            When set to ``True``, orient tree left to right rather than top-down.

        Returns
        -------
        dot_data : string
            String representation of the decision tree classifier in GraphViz dot format.
        """

        def process_tree_recursively(tree, node_id):
            """ Process tree recursively node by node and provide GraphViz dot format for node."""

            # Current node
            left_child = tree.get_node_left_child(node_id)
            right_child = tree.get_node_right_child(node_id)
            feature = tree.get_node_feature(node_id)
            threshold = tree.get_node_threshold(node_id)
            histogram = tree.get_node_histogram(node_id)
            impurity = tree.get_node_impurity(node_id)

            # Prediction
            n = sum(histogram)
            p_c = histogram / n
            c = np.argmax(p_c)
            # formatting
            p_c = [int(x) if x % 1 == 0 else round(float(x), 2) for x in p_c]

            # Node color and intensity based on classification and impurity
            (r, g, b) = rgb_LUT[c]
            max_impurity = 1.0 - (1.0 / tree.get_n_classes())
            alpha = int(255 * (max_impurity - impurity) / max_impurity)
            color = '#' + ''.join('{:02X}'.format(a) for a in [r, g, b, alpha])  # #RRGGBBAA hex format

            # Leaf node
            if left_child is None:
                # leaf nodes do no have any children
                # so we only need to test for one of the children

                class_name = class_names[c] if class_names is not None else "%d" % c

                # Node
                dot_data.write('%d [label=\"%s\\n%s\", fillcolor=\"%s\"] ;\n'
                               % (node_id, p_c, class_name, color))

            # Split node
            else:

                # Order children nodes by predicted classes (and their probabilities)
                # Switch left_child with right_child and
                # modify test feature <= threshold (default) vs feature > threshold accordingly

                order = True
                test_type = 0   # 0: feature <= threshold (default)
                                # 1: feature >  threshold, when left and right children are switched

                if order:
                    change = False
                    # Left Child Prediction
                    lc_histogram = tree.get_node_histogram(left_child)
                    lc_c = np.argmax(lc_histogram)
                    lc_n = sum(lc_histogram)
                    lc_p_c = lc_histogram[lc_c] / lc_n
                    # Right Child Prediction
                    rc_histogram = tree.get_node_histogram(right_child)
                    rc_c = np.argmax(rc_histogram)
                    rc_n = sum(rc_histogram)
                    rc_p_c = rc_histogram[rc_c] / rc_n
                    # Determine if left_child and right_child should be switched based on predictions
                    if lc_c > rc_c:  # assign left child to lower class index
                        change = True
                    elif lc_c == rc_c:           # if class indices are the same for left and right children
                        if lc_c == 0:            # for the first class index = 0
                            if lc_p_c < rc_p_c:  # assign left child to higher class probability
                                change = True
                        else:                    # for all other class indices > 0
                            if lc_p_c > rc_p_c:  # assign left child to lower class probability
                                change = True
                    if change:
                        test_type = 1
                        left_child, right_child = right_child, left_child

                feature_name = feature_names[feature] if feature_names is not None else "X[%d]" % feature
                threshold = round(threshold, 3)

                # Edge width based on (weighted) number of samples used for training
                n_root = sum(tree.get_node_histogram(0))  # total number of samples used for training
                n_left_child = sum(tree.get_node_histogram(left_child)) / n_root  # normalized
                n_right_child = sum(tree.get_node_histogram(right_child)) / n_root

                max_width = 10
                if node_id == 0:  # Root node with legend
                    # Node
                    if test_type == 0:
                        dot_data.write('%d [label=\"%s <= %s\\np(class) = %s\\nclass, n = %s\", fillcolor=\"%s\"] ;\n'
                                       % (node_id, feature_name, threshold, p_c, int(round(n, 0)), color))
                    else:  # test_type == 1
                        dot_data.write('%d [label=\"%s > %s\\np(class) = %s\\nclass, n = %s\", fillcolor=\"%s\"] ;\n'
                                       % (node_id, feature_name, threshold, p_c, int(round(n, 0)), color))
                    # Edges
                    dot_data.write('%d -> %d [penwidth=%f, headlabel="True", labeldistance=2.5, labelangle=%d] ;\n'
                                   % (node_id, left_child, max_width * n_left_child, -45 if rotate else 45))
                    dot_data.write('%d -> %d [penwidth=%f, headlabel="False", labeldistance=2.5, labelangle=%d] ;\n'
                                   % (node_id, right_child, max_width * n_right_child, 45 if rotate else -45))
                else:
                    # Node
                    if test_type == 0:
                        dot_data.write('%d [label=\"%s <= %s\\n%s\", fillcolor=\"%s\"] ;\n'
                                       % (node_id, feature_name, threshold, p_c, color))
                    else:  # test_type == 1
                        dot_data.write('%d [label=\"%s > %s\\n%s\", fillcolor=\"%s\"] ;\n'
                                       % (node_id, feature_name, threshold, p_c, color))
                    # Edges
                    dot_data.write('%d -> %d [penwidth=%f] ;\n'
                                   % (node_id, left_child, max_width * n_left_child))
                    dot_data.write('%d -> %d [penwidth=%f] ;\n'
                                   % (node_id, right_child, max_width * n_right_child))

                # process the children's sub trees recursively
                process_tree_recursively(tree, left_child)
                process_tree_recursively(tree, right_child)

            return

        def create_rgb_LUT(n_classes):
            """ Create a rgb color look up table (LUT) for all classes.
            """

            # Define rgb colors for the different classes
            # with (somewhat) max differences in hue between nearby classes

            # Number of iterations over the grouping of 2x 3 colors
            n_classes = max(n_classes, 1)  # input check > 0
            n = ((n_classes - 1) // 6) + 1  # > 0

            # Create a list of offsets for the grouping of 2x 3 colors
            # that (somewhat) max differences in hue between nearby classes
            offset_list = [0]  # creates pure R G B - Y C M colors
            d = 128
            n_offset_levels = int(scipy.log2(n - 1) + 1) if n > 1 else 1  # log(0) not defined
            n_offset_levels = min(n_offset_levels, 4)  # limit number of colors to 96
            for i in range(n_offset_levels):
                # Create in between R G B Y C M colors
                # in a divide by 2 pattern per level
                # i=0: + 128,
                # i=1: +  64, 192,
                # i=2: +  32, 160, 96, 224,
                # i=3: +  16, 144, 80, 208, 48, 176, 112, 240
                # abs max i=7 with + 1 ...
                offset_list += ([int(offset + d) for offset in offset_list])
                d /= 2

            # If there are more classes than colors
            # then the offset_list is duplicated,
            # which assigns the same colors to different classes
            # but at least to the most distance classes
            length = len(offset_list)
            if n > length:
                offset_list = int(1 + scipy.ceil((n - length) / length)) * offset_list

            rgb_LUT = []
            for i in range(n):
                # Calculate grouping of 2x 3 rgb colors R G B - Y C M
                # that (somewhat) max differences in hue between nearby classes
                # and makes it easy to define other in between colors
                # using a simple linear offset
                # Based on HSI to RGB calculation with I = 1 and S = 1
                offset = offset_list[i]
                rgb_LUT.append((255, offset, 0))  # 0 <= h < 60 RED ...
                rgb_LUT.append((0, 255, offset))  # 120 <= h < 180 GREEN ...
                rgb_LUT.append((offset, 0, 255))  # 240 <= h < 300 BLUE ...
                rgb_LUT.append((255 - offset, 255, 0))  # 60 <= h < 120 YELLOW ...
                rgb_LUT.append((0, 255 - offset, 255))  # 180 <= h < 240 CYAN ...
                rgb_LUT.append((255, 0, 255 - offset))  # 300 <= h < 360 MAGENTA ...

            return rgb_LUT

        # Check that fit has been called
        check_is_fitted(self, ['tree_'])

        dot_data = StringIO()

        dot_data.write('digraph Tree {\n')
        dot_data.write(
            'node [shape=box, style=\"rounded, filled\", color=\"black\", fontname=helvetica, fontsize=14] ;\n')
        dot_data.write('edge [fontname=helvetica, fontsize=12] ;\n')

        # Rotate (default: top-down)
        if rotate:
            dot_data.write('rankdir=LR ;\n')  # left-right orientation

        # Define rgb colors for the different classes
        rgb_LUT = create_rgb_LUT(self.tree_.get_n_classes())

        # Process the tree recursively
        process_tree_recursively(self.tree_, 0)  # root node = 0

        dot_data.write("}")

        return dot_data.getvalue()

    def export_text(self):
        """Export of a decision tree in a simple text format.

        Returns
        -------
        data : string
            String representation of the decision tree classifier in a simple text format.
        """

        def process_tree_recursively(tree, node_id):
            """ Process tree recursively node by node and provide simple text format for node."""

            # Current node
            left_child = tree.get_node_left_child(node_id)
            right_child = tree.get_node_right_child(node_id)
            feature = tree.get_node_feature(node_id)
            threshold = tree.get_node_threshold(node_id)
            histogram = [int(x) if x % 1 == 0 else round(float(x), 2) for x in tree.get_node_histogram(node_id)]

            # Leaf node
            if left_child is None:
                # leaf nodes do no have any children
                # so we only need to test for one of the children

                data.write('%d' % node_id)
                data.write(' %s; ' % histogram)

            # Split node
            else:

                data.write('%d' % node_id)
                data.write(' X[%d]' % feature)
                data.write('<=%s' % round(float(threshold), 2))
                data.write(' %s; ' % histogram)

                data.write('%d->%d; ' % (node_id, left_child))
                data.write('%d->%d; ' % (node_id, right_child))

                # process the children's sub trees recursively
                process_tree_recursively(tree, left_child)
                process_tree_recursively(tree, right_child)

            return

        # Check that fit has been called
        check_is_fitted(self, ['tree_'])

        data = StringIO()

        # Process the tree recursively
        process_tree_recursively(self.tree_, 0)  # root node = 0

        return data.getvalue()