.. title:: Implementation : contents

.. _implementation:

==============
Implementation
==============

scikit-learn compatible
=======================

We rolled our own scikit-learn compatible estimator
following the `Rolling your own estimator`_ instructions
and using the provided `project template`_ from `scikit-learn`_.

.. _`Rolling your own estimator`: https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
.. _`project template`: https://github.com/scikit-learn-contrib/project-template
.. _`scikit-learn`: http://scikit-learn.org

We are trying to be consistent with scikit-learn's `decision tree`_ and `ensemble`_ modules.

**Exceptions**

Used ``class_balance`` as hyperparameter name instead of ``class_weight``

    The class_weight hyperparameter name is recognized by check_estimator()
    and the test check_class_weight_classifiers() is performed
    that uses the dict parameter and requires for a decision tree
    the “min_weight_fraction_leaf” hyperparameter to be implemented to pass the test.

.. _`decision tree`: https://scikit-learn.org/stable/modules/tree.html
.. _`ensemble`: https://scikit-learn.org/stable/modules/ensemble.html

We provide and use the same Random Number Generator from our C++ implementation in Python.

Basic Concepts
==============

The basic concepts, including stack, samples LUT with in-place partitioning, incremental histogram updates,
for the implementation of the classifiers are based on:

G. Louppe, `Understanding Random Forests`_, PhD Thesis, 2014

.. _`Understanding Random Forests` : https://arxiv.org/pdf/1407.7502.pdf

Not Missing At Random (NMAR)
----------------------------

The probability of an instance having a missing value for a feature may depend on the value of that feature.

**Training**
The split criterion considers missing values as another category and samples with missing values are passed to either the left or the right child depending on which option provides the best split.

**Testing**
If the split criterion includes missing values, a missing value is dealt with accordingly (passed to left or right child).
If the split criterion does not include missing values, a missing value at a split criterion is dealt with by combining the results from both children proportionally to the number of samples that are passed to the children during training (same as MCAR Missing Completely At Random).

Note that the number of samples that are passed to the children represents the feature's estimated probability distribution for the particular missing value based on the training data.


