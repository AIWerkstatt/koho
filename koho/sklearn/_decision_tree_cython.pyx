# encoding=utf-8
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
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

Optimized Implementation:
stack, samples LUT with in-place partitioning, incremental histogram updates

Python implementation using NumPy with Criterion implemented in Cython.
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

import numpy as np

PRECISION = 1e-7  # used for floating point "==" test

# ==============================================================================
# Tree
# ==============================================================================


class Node:
    """ Node of a binary tree.
    """

    def __init__(self, left_child=None, right_child=None,
                 feature=None, threshold=None,
                 histogram=None,
                 impurity=None, improvement=None):
        """ Create a new node.
        """

        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.histogram = np.array(histogram)  # number of samples per class  !!!
        self.impurity = impurity  # for inspection (e.g. graphviz visualization)
        self.improvement = improvement  # for feature importances

        return


class Tree:
    """ Binary tree structure build up of nodes.
    """

    def __init__(self, n_features=None, n_classes=None):
        """ Create a new tree without nodes.
        """

        # Parameters
        self.n_features = n_features
        self.n_classes = n_classes
        # Fields
        self.max_depth = None
        self.node_count = 0
        self.nodes = []  # list of nodes

        return

    def _add_node(self, parent_id, is_left,
                  feature, threshold,
                  histogram,
                  impurity, improvement):
        """ Add a new node to the tree.

        The new node registers itself as the child of its parent.
        """

        node_id = self.node_count

        # Add new node to tree
        node = Node(left_child=None, right_child=None,
                    # children IDs are set when the child nodes are added
                    feature=feature, threshold=threshold,
                    histogram=histogram,
                    impurity=impurity, improvement=improvement)
        self.nodes.append(node)

        # Register new node as the child of its parent
        if parent_id is not None:
            if is_left:
                self.nodes[parent_id].left_child = node_id
            else:
                self.nodes[parent_id].right_child = node_id

        self.node_count += 1

        return node_id

    def predict(self, X):
        """ Predict classes probablities for the test data.
        """

        n_samples = X.shape[0]
        y_prob = np.zeros((n_samples, self.n_classes), dtype=np.float64)

        # Loop: samples
        for i in range(n_samples):

            # Go to the root node
            idx = 0
            # Loop: root to leaf node
            # While node is not a leaf node
            while self.nodes[idx].left_child is not None:
                # As leaf nodes do no have any children
                # only test for one of the children

                # Go to left or right child node depending on split (feature, threshold)
                if X[i, self.nodes[idx].feature] <= self.nodes[idx].threshold:
                    idx = self.nodes[idx].left_child
                else:
                    idx = self.nodes[idx].right_child

            # Calculate classes probabilities
            # based on number of samples per class histogram
            y_prob[i] = self.nodes[idx].histogram / np.sum(self.nodes[idx].histogram)

        return y_prob

    def calculate_feature_importances(self):
        """ Calculate feature importances from the decision tree.
        """

        importances = np.zeros(self.n_features, dtype=np.float64)

        # Loop: all nodes
        for idx in range(self.node_count):

            # Split node
            if self.nodes[idx].left_child is not None:
                # leaf nodes do no have any children
                # so we only need to test for one of the children

                importances[self.nodes[idx].feature] += self.nodes[idx].improvement

        normalizer = np.sum(importances)
        if normalizer > 0.0:  # 0 when root is pure
            importances /= normalizer

        return importances

    def get_n_classes(self):
        return self.n_classes

    def get_node_left_child(self, idx):
        return self.nodes[idx].left_child

    def get_node_right_child(self, idx):
        return self.nodes[idx].right_child

    def get_node_feature(self, idx):
        return self.nodes[idx].feature

    def get_node_threshold(self, idx):
        return self.nodes[idx].threshold

    def get_node_histogram(self, idx):
        return self.nodes[idx].histogram

    def get_node_impurity(self, idx):
        return self.nodes[idx].impurity

# ==============================================================================
# Impurity Criterion
# ==============================================================================

import cython

from libc.stdlib cimport calloc, free
from libc.string cimport memset, memcpy

@cython.boundscheck(False)
@cython.wraparound(False)


cdef class GiniCriterion:
    """Gini Index impurity criterion.
    """

    cdef long n_classes
    cdef long n_samples
    cdef double* class_weight

    cdef double* node_weighted_histogram
    cdef double node_weighted_n_samples
    cdef double node_impurity

    cdef double* node_weighted_histogram_left
    cdef double node_weighted_n_samples_left
    cdef double impurity_left
    cdef double* node_weighted_histogram_right
    cdef double node_weighted_n_samples_right
    cdef double impurity_right
    cdef long node_pos

    def __cinit__(self, long n_classes, long n_samples, double[::1] class_weight):
        """ Create and initialize a new gini criterion.

        Assuming: y is 0, 1, 2, ... (n_classes - 1).
        """

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.class_weight = <double*> calloc(n_classes, sizeof(double))

        self.node_weighted_histogram = <double*> calloc(n_classes, sizeof(double))
        self.node_weighted_histogram_left = <double*> calloc(n_classes, sizeof(double))
        self.node_weighted_histogram_right = <double*> calloc(n_classes, sizeof(double))

        if (self.class_weight == NULL or
            self.node_weighted_histogram == NULL or
            self.node_weighted_histogram_left == NULL or
            self.node_weighted_histogram_right == NULL):
            raise MemoryError()

        cdef long c
        for c in range(n_classes):
            self.class_weight[c] = class_weight[c]

        self.node_weighted_n_samples = 0.0
        self.node_impurity = 0.0

        self.node_weighted_n_samples_left = 0.0
        self.impurity_left = 0.0
        self.node_weighted_n_samples_right = 0.0
        self.impurity_right = 0.0
        self.node_pos = 0

        return

    def __dealloc__(self):
        """Destructor."""

        free(self.class_weight)
        free(self.node_weighted_histogram)
        free(self.node_weighted_histogram_left)
        free(self.node_weighted_histogram_right)

        return

    cpdef calculate_node_histogram(self, long[::1] y, long[::1] samples, long start, long end):
        """ Calculate number of samples per class histogram for current node.
        """

        # Calculate number of samples per class histogram for current node

        cdef double* histogram
        histogram = NULL
        histogram = <double*> calloc(self.n_classes, sizeof(double))
        if (histogram == NULL):
            raise MemoryError

        cdef long p
        for p in range(start, end):
            histogram[y[samples[p]]] += 1.0

        # Apply class weights to balance classes

        self.node_weighted_n_samples = 0.0
        cdef double weighted_cnt
        cdef long c
        for c in range(self.n_classes):
            weighted_cnt = self.class_weight[c] * histogram[c]
            self.node_weighted_histogram[c] = weighted_cnt
            self.node_weighted_n_samples += weighted_cnt

        free(histogram)

        return np.asarray(<double[:self.n_classes]> self.node_weighted_histogram)

    cpdef calculate_node_impurity(self):
        """Calculate the impurity of the current node using the Gini criterion.

        Assuming: calculate_node_histogram(), sum(node_weighted_histogram) > 0
        """

        cdef double cnt
        cdef double sum_sq = 0.0
        cdef long c
        for c in range(self.n_classes):
            cnt = self.node_weighted_histogram[c]
            sum_sq += cnt*cnt

        self.node_impurity = 1.0 - sum_sq / (self.node_weighted_n_samples * \
                                             self.node_weighted_n_samples)

        return self.node_impurity

    cpdef init_children_histograms(self):
        """ Initialize number of samples per class histograms for the children
        of the current node.

        Assuming: calculate_node_histogram()
        """

        # Initialize current position
        self.node_pos = 0

        # Initialize number of samples per class histogram for left child to 0
        memset(self.node_weighted_histogram_left, 0, self.n_classes * sizeof(double))
        self.node_weighted_n_samples_left = 0.0

        # Initialize number of samples per class histogram for right child to
        # samples per class histogram from the current node
        memcpy(self.node_weighted_histogram_right, self.node_weighted_histogram, self.n_classes * sizeof(double))
        self.node_weighted_n_samples_right = self.node_weighted_n_samples

        return

    cpdef update_children_histograms(self, long[::1] y_view, long[::1] sf_view, long new_pos):
        """ Update number of samples per class histograms for the children
        of the current node from current position to a new position.

        Assuming: new_pos > pos.
        Assuming: init_children_histograms()
        """

        # Add number of samples per class histogram for samples[pos, new_pos]
        # to the number of samples per class histogram for the left child
        # including samples[start, pos]

        cdef double* histogram
        histogram = NULL
        histogram = <double*> calloc(self.n_classes, sizeof(double))
        if (histogram == NULL):
            raise MemoryError

        cdef long p
        for p in range(self.node_pos, new_pos):
            histogram[y_view[sf_view[p]]] += 1.0

        # Apply class weights to balance classes

        cdef double weighted_cnt
        cdef long c
        for c in range(self.n_classes):
            weighted_cnt = self.class_weight[c] * histogram[c]
            self.node_weighted_histogram_left[c] += weighted_cnt
            self.node_weighted_n_samples_left += weighted_cnt

        free(histogram)

        # Update number of samples per class histogram for the right child
        # given: histogram_left[x] + histogram_right[x] = histogram[x]

        for c in range(self.n_classes):
            self.node_weighted_histogram_right[c] = self.node_weighted_histogram[c] - \
                                                    self.node_weighted_histogram_left[c]

        self.node_weighted_n_samples_right = self.node_weighted_n_samples - \
                                             self.node_weighted_n_samples_left

        # Update current position
        self.node_pos = new_pos

        return

    cpdef calculate_impurity_children(self):
        """Calculate the impurity of children nodes from the current node using the Gini index
        based on the number of samples per class histograms for the current position.

        Assuming: update_children_histograms()
        """

        cdef double cnt_left
        cdef double sum_sq_left = 0.0
        cdef double cnt_right
        cdef double sum_sq_right = 0.0
        cdef long c
        for c in range(self.n_classes):
            cnt_left = self.node_weighted_histogram_left[c]
            sum_sq_left += cnt_left*cnt_left
            cnt_right = self.node_weighted_histogram_right[c]
            sum_sq_right += cnt_right*cnt_right

        self.impurity_left = 1.0 - sum_sq_left / (self.node_weighted_n_samples_left * \
                                                  self.node_weighted_n_samples_left)
        self.impurity_right = 1.0 - sum_sq_right / (self.node_weighted_n_samples_right * \
                                                    self.node_weighted_n_samples_right)

        return self.impurity_left, self.impurity_right

    cpdef calculate_impurity_improvement(self):
        """Calculate the impurity improvement from the current node to its children
        based on the impurity of the children nodes for the current position.

        Assuming: calculate_node_impurity(), calculate_impurity_children()
        """

        self.node_weighted_n_samples = self.node_weighted_n_samples_left + \
                                       self.node_weighted_n_samples_right

        improvement = (self.node_weighted_n_samples / self.n_samples) * \
                      (self.node_impurity -
                       (self.node_weighted_n_samples_left /
                        self.node_weighted_n_samples) * self.impurity_left -
                       (self.node_weighted_n_samples_right /
                        self.node_weighted_n_samples) * self.impurity_right)
        # Using "n_samples as "n_samples = sum of all weighted samples"
        # given the way the class weights are calculated

        return improvement

@cython.boundscheck(True)
@cython.wraparound(True)

# ==============================================================================
# Node Splitter
# ==============================================================================


class BestSplitter:
    """ Splitter to find the best split for a node.
    """

    def __init__(self, n_classes, n_features, n_samples, class_weight, max_features, max_thresholds, random_state):
        """ Create and initialize a new best splitter.
        """

        # Create amd initialize a gini criterion
        self.criterion = GiniCriterion(n_classes, n_samples, class_weight)

        # Initialize samples[0, n_samples] to the training data X, y
        # samples[start, end] is a global LUT to the training data X, y
        # to handle the recursive partitioning and
        # the sorting of the data efficiently.
        self.samples = np.arange(0, n_samples)  # identity mapping
        self.n_samples = n_samples
        self.start = 0
        self.end = n_samples

        # Features
        self.n_features = n_features

        # Hyperparameters
        self.max_features = max_features  # required: 0 < max_features <= n_features
        self.max_thresholds = max_thresholds  # required: 0, 1

        # Random Number Generator
        self.random_state = random_state

        return

    def init_node(self, y, start, end):
        """ Calculate number of samples per class histogram and impurity for the node.
        """

        self.start = start
        self.end = end

        histogram = self.criterion.calculate_node_histogram(y, self.samples, start, end).tolist()  # !!!
        impurity = self.criterion.calculate_node_impurity()

        return histogram, impurity

    def split_feature(self, f, X, y):
        """ Find the best split and partition samples for a given feature.
        """

        max_improvement = 0
        feature, threshold, pos = None, None, 0
        s = 0

        # y is not constant (impurity > 0)
        # has been checked by impurity stop criteria in build()

        # Copy Xf=X[samples[start:end],f]
        # note: index and slicing provides a new view on the data, no copy
        Xf = X[self.samples[self.start:self.end], f]

        # Xf is not constant
        Xf_min, Xf_max = Xf.min(), Xf.max()
        if Xf_min + PRECISION < Xf_max:

            # Copy sf=samples[start:end]
            # note: index and slicing provides a new view on the data, no copy
            sf = self.samples[self.start:self.end]
            # Indexing is now from 0 to n_samples
            n_samples = self.end - self.start

            # Initialize number of samples per class histograms for the children of the node
            self.criterion.init_children_histograms()

            if self.max_thresholds == 0:

                # Loop: all thresholds
                # --------------------

                # Sort Xf and sf by Xf using advanced indexing
                # note: advanced indexing provides a copy of the data
                idx = Xf.argsort()
                Xf = Xf[idx]
                sf = sf[idx]
                # samples sf are properly ordered for the partitioning

                # Initialize position of last and next potential split to 0
                p, pn = 0, 0
                # Loop: samples
                while (pn < n_samples):
                    # If remaining Xf values are constant
                    if (Xf[pn] + PRECISION >= Xf[-1]):
                        break  # next feature
                    # Skip constant Xf values
                    while (pn+1 < n_samples and
                            Xf[pn] + PRECISION >= Xf[pn+1]):
                        pn += 1
                    # Set pn to next position
                    pn += 1

                    # if (pn < n_samples): ... p = pn
                    # Not required, because "if (Xf[pn] + PRECISION >= Xf[-1]): break" above
                    # ensures that pn += 1 always points to valid data (pn < n_samples)

                    # Update number of samples per class histograms for the children
                    # of the current node from current position p to the new position np
                    self.criterion.update_children_histograms(y, sf, pn)
                    # Calculate impurity of children
                    impurity_left, impurity_right = self.criterion.calculate_impurity_children()
                    # Calculate impurity improvement for current position
                    improvement = self.criterion.calculate_impurity_improvement()
                    if (improvement > max_improvement):
                        max_improvement = improvement

                        if (feature != f):  # only once per feature
                            s = np.copy(sf)

                        # Best split
                        feature = f
                        threshold = (Xf[p] + Xf[pn]) / 2.0
                        pos = self.start + pn

                    # If impurity of right child is 0.0 (pure)
                    if (impurity_right < PRECISION):
                        break  # next feature

                    p = pn

            else:

                # random threshold
                # ----------------
                # using the Extreme Random Tree formulation

                # Copy Xf and sf
                Xf = Xf[:]
                sf = sf[:]

                # Draw random threshold
                threshold = self.random_state.uniform(Xf_min + PRECISION, Xf_max)  # excludes Xf_min, Xf_max
                # note numpy uniform(low, high), low is inclusive and high is exclusive

                # Partition sf such that X[sf[np-1],f] <= threshold < X[sf[np],f]
                p, pn = 0, n_samples
                while (p < pn):
                    if Xf[p] <= threshold:
                        p += 1
                    else:
                        pn -= 1
                        Xf[p], Xf[pn] = Xf[pn], Xf[p]  # swap
                        sf[p], sf[pn] = sf[pn], sf[p]  # swap

                # Update number of samples per class histograms for the children
                # of the current node from current position p to the new position np
                self.criterion.update_children_histograms(y, sf, pn)
                # Calculate impurity of children
                impurity_left, impurity_right = self.criterion.calculate_impurity_children()
                # Calculate impurity improvement for current position
                max_improvement = self.criterion.calculate_impurity_improvement()

                # Best (and only) split
                s = sf
                pos = self.start + pn

        return threshold, s, pos, max_improvement

    def split_node(self, X, y):
        """ Find the best split and partition samples.

        Find the split (feature, threshold) on samples[start:end] and
        partition samples[start:end] into samples[start:pos] and samples[pos:end]
        according to split.

        Assuming: init_node()
        """

        max_improvement = 0
        feature, threshold, pos = None, None, 0
        s = 0

        # Loop: k random features (k defined by max_features)
        # ---------------------------------------------------

        # When max_features == n_features this is the same as
        # Loop: all features "for f in range(self.n_features):",
        # but in a random order, which is preferable.

        # Features are sampled without replacement using
        # the modern version of the Fischer-Yates shuffle algorithm
        # in an interative way.

        # Alternatively: implement as a generator
        # Alternatively: np.random.shuffle(features) in-place

        features = np.arange(0, self.n_features)  # identity mapping

        i = self.n_features  # i=n instead of n-1 because of randint(0,n)
        while (i > (self.n_features - self.max_features) or  # includes case 0
               # continue until at least one none constant feature was selected
               # or ultimately no more features are left
               (max_improvement == 0 and i > 0)):

            j = self.random_state.randint(0, i) if i > 1 else 0  # covers case 0
            # note randint(low, high), low is inclusive and high is exclusive
            i -= 1  # adjust indexing by i
            features[j], features[i] = features[i], features[j]  # swap
            f = features[i]

            # Split feature
            thresholdf, sf, posf, improvementf = self.split_feature(f, X, y)
            if (improvementf > max_improvement):
                max_improvement = improvementf
                feature = f
                threshold = thresholdf
                pos = posf
                s = sf

        # Replace samples[start:end] with properly ordered samples s
        if (feature is not None):  # only if a split was found
            self.samples[self.start:self.end] = s

        return feature, threshold, pos, max_improvement

# ==============================================================================
# Tree Builder
# ==============================================================================


class Stack:
    """ LIFO data structure.
    """

    def __init__(self):
        self.items = []
        return

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)
        return

    def pop(self):
        # test is_empty() prior to call
        return self.items.pop()

class DepthFirstTreeBuilder:
    """ Build a binary decision tree in depth-first order.
    """

    def __init__(self, n_classes, n_features, n_samples, class_weight, max_depth, max_features, max_thresholds, random_state):
        """ Create and initialize a new depth first tree builder.
        """

        # Create and initialize a new best splitter (and gini criterion)
        self.splitter = BestSplitter(n_classes, n_features, n_samples, class_weight, max_features, max_thresholds, random_state)

        # Hyperparameters
        self.max_depth = max_depth

        return

    def build(self, tree, X, y, n_classes, class_weight):
        """ Build a binary decision tree from the training data.
        """

        # Create an empty stack
        stack = Stack()

        # Push root node information onto the stack
        n_samples = X.shape[0]
        start, end = 0, n_samples
        depth, parent, is_left = 0, None, None
        stack.push((start, end, depth, parent, is_left))
        # Loop: nodes
        while not stack.is_empty():
            # Pop current node information from the stack
            start, end, depth, parent, is_left = stack.pop()
            n_samples = end - start

            # Calculate number of samples per class histogram
            # and impurity for the current node.
            histogram, impurity = self.splitter.init_node(y, start, end)

            # If a stop criterion is met
            is_leaf = (depth >= self.max_depth) or \
                      (impurity <= PRECISION)  # node is leaf node

            # Split node (if node is not a leaf node)
            feature, threshold = None, None
            improvement = 0.0
            if not is_leaf:
                # Find the split on samples[start:end] that maximizes impurity improvement and
                # partition samples[start:end] into samples[start:pos] and samples[pos:end]
                # according to the split
                feature, threshold, pos, improvement = self.splitter.split_node(X, y)
                # If no impurity improvement (no split found)
                is_leaf = True if feature is None else False  # node is leaf node

            # Add node to the decision tree
            node = tree._add_node(
                # references to insert left/right child ID in parent node
                parent, is_left,
                # assign feature, threshold to the node (used to split)
                feature, threshold,
                # assign histogram to the node (used to calculate the prediction)
                histogram,
                # impurity is used for inspection (e.g. graphviz visualization)
                # improvement is used for feature importances
                impurity, improvement)

            # Split node (if not a leaf node)
            if not is_leaf:
                # Push right child node information onto the stack
                stack.push((pos, end, depth+1, node, False))
                # Push left child node information onto the stack
                # LIFO: left depth first order
                stack.push((start, pos, depth+1, node, True))

        return
