# encoding=utf-8
""" Cython binding for C++ implementation.
"""

# Author: AI Werkstatt (TM)
# (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

import numpy as np
cimport cython
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# C++ class interface

cdef extern from "../cpp/random_number_generator.h" namespace "koho" nogil:

    cdef cppclass CppRandomState "koho::RandomState":
        void                CppRandomState() except +
        void                CppRandomState(unsigned long seed) except +
        double              uniform_real(double low, double high)
        long                uniform_int(long low, long high)
        long                MAX_INT

cdef extern from "../cpp/decision_tree.h" namespace "koho" nogil:

    ctypedef double         Features_t
    ctypedef long           Classes_t
    ctypedef double         ClassWeights_t

    ctypedef double         Histogram_t

    ctypedef unsigned long  SamplesIdx_t
    ctypedef unsigned long  FeaturesIdx_t
    ctypedef unsigned long  ClassesIdx_t
    ctypedef unsigned long  NodesIdx_t
    ctypedef unsigned long  TreeDepthIdx_t

    cdef cppclass CppNode "koho::Node":

        NodesIdx_t          left_child
        NodesIdx_t          right_child
        FeaturesIdx_t       feature
        int                 NA
        Features_t          threshold
        vector[Histogram_t] histogram
        double              impurity
        double              improvement

        void                CppNode(NodesIdx_t left_child, NodesIdx_t right_child,
                                    FeaturesIdx_t feature, int NA, Features_t threshold,
                                    const vector[Histogram_t]& histogram,
                                    double impurity, double improvement) except +

    cdef cppclass CppTree "koho::Tree":

        ClassesIdx_t        n_classes
        FeaturesIdx_t       n_features
        TreeDepthIdx_t      max_depth
        NodesIdx_t          node_count
        vector[CppNode]     nodes

        void                CppTree(ClassesIdx_t n_classes, FeaturesIdx_t n_features) except +
        void                predict(double* X, unsigned long n_samples, double* y_prob)
        void                calculate_feature_importances(double* importances)

    cdef cppclass CppDepthFirstTreeBuilder "koho::DepthFirstTreeBuilder":
        void                CppDepthFirstTreeBuilder(ClassesIdx_t      n_classes,
                                                     FeaturesIdx_t     n_features,
                                                     SamplesIdx_t      n_samples,
                                                     ClassWeights_t*   class_weight,
                                                     TreeDepthIdx_t    max_depth,
                                                     FeaturesIdx_t     max_features,
                                                     unsigned long     max_thresholds,
                                                     string            missing_values,
                                                     CppRandomState    random_state) except +
        void                build(CppTree tree, Features_t* X, Classes_t* y, SamplesIdx_t n_samples)

# Cython wrapper class

@cython.boundscheck(False)
@cython.wraparound(False)

cdef class RandomState:
    """ A random number generator.
    """

    cdef unique_ptr[CppRandomState] thisptr

    def __cinit__(self, seed=None):
        if seed is None:
            self.thisptr.reset(new CppRandomState())  # current system time
        else:
            self.thisptr.reset(new CppRandomState(seed))  # seed

    def uniform(self, low, high, size=None):
        """Provide a double random number from a uniform distribution between [low, high).
        """
        cdef long i
        if size is None:
            return deref(self.thisptr).uniform_real(low, high)
        else:
            # [O] uniform random numbers [size]
            rn = np.zeros(size, dtype=np.float64)
            for i in range(size):
                rn[i] = deref(self.thisptr).uniform_real(low, high)
            return rn

    def randint(self, low, high, size=None):
        """Provide a long random number from a uniform distribution between [low, high).
        """
        cdef long i
        if size is None:
            return deref(self.thisptr).uniform_int(low, high)
        else:
            # [O] uniform random numbers [size]
            rn = np.zeros(size, dtype=np.int64)
            for i in range(size):
                rn[i] = deref(self.thisptr).uniform_int(low, high)
            return rn

    @property
    def MAX_INT(self):
        """Upper bound for long random number [..., high).
        """
        return deref(self.thisptr).MAX_INT

cdef class Tree:
    """ Binary tree structure build up of nodes.
    """

    cdef unique_ptr[CppTree] thisptr

    def __cinit__(self, n_classes, n_features):
        self.thisptr.reset(new CppTree(n_classes, n_features))

    def __reduce__(self):
        return (Tree, (deref(self.thisptr).n_classes, deref(self.thisptr).n_features), self.__getstate__())

    def __getstate__(self):
        state = {}
        state['version'] = '1'
        state['n_classes']  = deref(self.thisptr).n_classes
        state['n_features'] = deref(self.thisptr).n_features
        state['max_depth']  = deref(self.thisptr).max_depth
        state['node_count'] = deref(self.thisptr).node_count
        nodes = []
        for idx in range(deref(self.thisptr).node_count): # node id implicit in order
            nodes.append((deref(self.thisptr).nodes[idx].left_child,
                          deref(self.thisptr).nodes[idx].right_child,
                          deref(self.thisptr).nodes[idx].feature,
                          deref(self.thisptr).nodes[idx].NA,
                          deref(self.thisptr).nodes[idx].threshold,
                          deref(self.thisptr).nodes[idx].histogram,
                          deref(self.thisptr).nodes[idx].impurity,
                          deref(self.thisptr).nodes[idx].improvement))
        state['nodes'] = nodes
        return state

    def __setstate__(self, state):
        cdef unique_ptr[CppNode] nodeptr
        if 'version' not in state:
            raise ValueError('Unsupported pickle format!')
        if state['version'] == '1':
            deref(self.thisptr).n_classes  = state['n_classes']
            deref(self.thisptr).n_features = state['n_features']
            deref(self.thisptr).max_depth  = state['max_depth']
            deref(self.thisptr).node_count = state['node_count']
            nodes = state['nodes']
            for node in nodes: # node id implicit in order
                nodeptr.reset(new CppNode(node[0],node[1],node[2],node[3],node[4],node[5],node[6],node[7]))
                deref(self.thisptr).nodes.push_back(deref(nodeptr))
        else:
            raise ValueError('Unsupported pickle format version: %s!' % state['version'])


    def predict(self, double[:, ::1] X_view):
        """ Predict classes probabilities for the test data.
        """
        # [O] class probabilities [n_samples x n_classes]
        cdef unsigned long n_samples = X_view.shape[0]
        cdef unsigned long n_classes = deref(self.thisptr).n_classes
        y_prob = np.zeros((n_samples, n_classes), dtype=np.float64)
        cdef double[:, ::1] y_prob_view = y_prob
        deref(self.thisptr).predict(&X_view[0,0], n_samples, &y_prob_view[0,0])
        return y_prob

    def calculate_feature_importances(self):
        """ Calculate feature importances from the decision tree.
        """
        # [O] feature importances [n_features]
        cdef unsigned long n_features = deref(self.thisptr).n_features
        importances = np.zeros(n_features, dtype=np.float64)
        cdef double[::1] importances_view = importances
        deref(self.thisptr).calculate_feature_importances(&importances_view[0])
        return importances

    def get_n_classes(self):
        return deref(self.thisptr).n_classes
    def get_n_features(self):
        return deref(self.thisptr).n_features
    def get_max_depth(self):
        return deref(self.thisptr).max_depth
    def get_node_count(self):
        return deref(self.thisptr).node_count

    def get_node_left_child(self, idx):
        return deref(self.thisptr).nodes[idx].left_child
    def get_node_right_child(self, idx):
        return deref(self.thisptr).nodes[idx].right_child
    def get_node_feature(self, idx):
        return deref(self.thisptr).nodes[idx].feature
    def get_node_NA(self, idx):
        return deref(self.thisptr).nodes[idx].NA
    def get_node_threshold(self, idx):
        return deref(self.thisptr).nodes[idx].threshold
    def get_node_histogram(self, idx):
        return np.asarray(deref(self.thisptr).nodes[idx].histogram)
    def get_node_impurity(self, idx):
        return deref(self.thisptr).nodes[idx].impurity
    def get_node_improvement(self, idx):
        return deref(self.thisptr).nodes[idx].improvement

cdef class DepthFirstTreeBuilder:
    """ Build a binary decision tree in depth-first order.
    """

    cdef unique_ptr[CppDepthFirstTreeBuilder] thisptr

    def __cinit__(self, n_classes, n_features,n_samples, double[::1] class_weight_view, max_depth, max_features,
                  max_thresholds, missing_values, RandomState random_state):
        self.thisptr.reset(new CppDepthFirstTreeBuilder(n_classes, n_features, n_samples, &class_weight_view[0],
                               max_depth, max_features, max_thresholds, missing_values.encode('UTF-8'), deref(random_state.thisptr)))

    def build(self, Tree tree, double[:, ::1] X_view, long[::1] y_view):
        """ Build a binary decision tree from the training data.
        """
        cdef unsigned long n_samples = X_view.shape[0]
        deref(self.thisptr).build(deref(tree.thisptr), &X_view[0,0], &y_view[0], n_samples)
        return



