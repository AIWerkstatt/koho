.. title:: Implementation : contents

.. _implementation:

==============
Implementation
==============

**Python implementation with Criterion implemented in Cython!**

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

Basic Concepts
==============

The basic concepts for the implementation of the classifiers are based on:

G. Louppe, `Understanding Random Forests`_, PhD Thesis, 2014

.. _`Understanding Random Forests` : https://arxiv.org/pdf/1407.7502.pdf

