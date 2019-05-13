/// Decision Tree module.
/** @file
- Classification
- Numerical (dense) data
- Missing values (Not Missing At Random (NMAR)
- Class balancing
- Multi-Class
- Single-Output
- Build order: depth first
- Impurity criteria: gini
- Split a. features: best over k (incl. all) random features
- Split b. thresholds: 1 random or all thresholds
- Stop criteria: max depth, (pure, no improvement)
- Important Features

Optimized Implementation:
stack, samples LUT with in-place partitioning, incremental histogram updates

C++ implementation.
*/

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

#ifndef KOHO_DECISION_TREE_H
#define KOHO_DECISION_TREE_H

#include <fstream>
#include <stack>
#include <algorithm>

#include <cstring> // memset for C-style data from Cython

#include "random_number_generator.h"

namespace koho {

    typedef double         Features_t; // X provided by Python, needs to be double
    typedef long           Classes_t; // y provided by Python, needs to be long
    typedef double         ClassWeights_t; // class weights provided by Python, needs to be double

    typedef double         Histogram_t; // weighted (number of classes) * number of samples

    typedef unsigned long  SamplesIdx_t; // number of samples
    typedef unsigned long  FeaturesIdx_t; // number of features
    typedef unsigned long  ClassesIdx_t; // number of classes
    typedef unsigned long  NodesIdx_t; // number of nodes
    typedef unsigned long  TreeDepthIdx_t; // maximum tree depth (2 ** 31) - 1

    const double  PRECISION_EQUAL = 1e-7; // float == 0.0 as float <= PRECISION_ZERO

// =============================================================================
// Tree
// =============================================================================

    /// Node of a binary tree.
    class Node {
    public:

        NodesIdx_t                left_child;
        NodesIdx_t                right_child;
        FeaturesIdx_t             feature;
        int                       NA; // NA as part of the split criterion, -1:NA, 0:left, 1:right
        Features_t                threshold;
        std::vector<Histogram_t>  histogram; // weighted number of samples per class
        double                    impurity; // for inspection (e.g. graphviz visualization)
        double                    improvement; // for feature importances

        /// Create a new node.
        Node(NodesIdx_t                       left_child,
             NodesIdx_t                       right_child,
             FeaturesIdx_t                    feature,
             int                              NA,
             Features_t                       threshold,
             const std::vector<Histogram_t>&  histogram,
             double                           impurity,
             double                           improvement);

        /// Serialize
        void  serialize(std::ofstream& fout);
        /// Deserialize
        static Node  deserialize(std::ifstream& fin);
    };

    /// Binary tree structure build up of nodes.
    class Tree {
    public:

        ClassesIdx_t       n_classes;
        FeaturesIdx_t      n_features;
        TreeDepthIdx_t     max_depth;
        // Nodes
        NodesIdx_t         node_count;
        std::vector<Node>  nodes;

        /// Create a new tree without nodes.
        /**
        @param[in]  n_classes   Number of classes.
        @param[in]  n_features  Number of features.
        */
        Tree(ClassesIdx_t   n_classes,
             FeaturesIdx_t  n_features);

        /// Add a new node to the tree.
        /**
        The new node registers itself as the child of its parent.
        */
        NodesIdx_t  add_node(TreeDepthIdx_t                   depth,
                             NodesIdx_t                       parent_id,
                             bool                             is_left,
                             FeaturesIdx_t                    feature,
                             int                              NA,
                             Features_t                       threshold,
                             const std::vector<Histogram_t>&  histogram,
                             double                           impurity,
                             double                           improvement);

        /// Predict classes probabilities for the test data.
        /**
        @param[in]      X          Test input samples [n_samples x n_features].
        @param[in]      n_samples  Number of samples in the test data.
        @param[in,out]  y_prob     Class probabilities corresponding to the test input samples [n_samples x n_classes].
        */
        // Using 1d array addressing for X and y_prob
        // to support efficient Cython bindings to Python using memory views.
        void  predict(Features_t*   X,
                      SamplesIdx_t  n_samples,
                      double*       y_prob);

        /// Calculate feature importances from the decision tree.
        /**
        @param[in,out]  importances  Feature importances corresponding to all features [n_features].
        */
        void  calculate_feature_importances(double*  importances);

        /// Serialize
        void  serialize(std::ofstream& fout);
        /// Deserialize
        void  deserialize(std::ifstream& fin);
    };

// =============================================================================
// Impurity Criterion
// =============================================================================

    /// Gini Index impurity criterion.
    class GiniCriterion {

    protected:
        ClassesIdx_t              n_classes;
        SamplesIdx_t              n_samples;
        ClassWeights_t*           class_weight;
        // Histograms
        // vectors are created in initialization list
        // - all samples
        std::vector<Histogram_t>  node_weighted_histogram;
        Histogram_t               node_weighted_n_samples;
        double                    node_impurity;
        // - samples with missing values
        std::vector<Histogram_t>  node_weighted_histogram_NA;
        Histogram_t               node_weighted_n_samples_NA;
        double                    node_impurity_NA;
        // - samples with values
        std::vector<Histogram_t>  node_weighted_histogram_values;
        Histogram_t               node_weighted_n_samples_values;
        double                    node_impurity_values;
        SamplesIdx_t              node_pos_NA;
        // - samples with values smaller than threshold (assigned to left child)
        std::vector<Histogram_t>  node_weighted_histogram_threshold_left;
        Histogram_t               node_weighted_n_samples_threshold_left;
        double                    node_impurity_threshold_left;
        // -- plus missing values (assigned to left child)
        Histogram_t               node_weighted_n_samples_threshold_left_NA;
        double                    node_impurity_threshold_left_NA;
        // - samples with values greater than threshold (assigned to right child)
        std::vector<Histogram_t>  node_weighted_histogram_threshold_right;
        Histogram_t               node_weighted_n_samples_threshold_right;
        double                    node_impurity_threshold_right;
        // -- plus missing values (assigned to right child)
        Histogram_t               node_weighted_n_samples_threshold_right_NA;
        double                    node_impurity_threshold_right_NA;
        SamplesIdx_t              node_pos_threshold;

    public:
        /// Create and initialize a new gini criterion.
        /**
        Assuming: y is 0, 1, 2, ... (n_classes - 1).
         */
        GiniCriterion(ClassesIdx_t     n_classes, // required: 2 <= n_classes
                      SamplesIdx_t     n_samples, // required: 2 <= n_samples
                      ClassWeights_t*  class_weight);

        /// Calculate weighted class histogram for current node.
        void calculate_node_histogram(Classes_t*                  y,
                                      std::vector<SamplesIdx_t>&  samples,
                                      SamplesIdx_t                start,
                                      SamplesIdx_t                end);

        /// Calculate impurity of a weighted class histogram using the Gini criterion.

        double  calculate_impurity(std::vector<Histogram_t>&  histogram);

        /// Calculate impurity of the current node.
        /**
        Assuming: calculate_node_histogram()
         */
        void calculate_node_impurity();

        /// Calculate class histograms for the samples with missing values and the samples with values.
        /**
        Assuming: number of missing values > 0
         */
        void calculate_NA_histogram(Classes_t*                  y,
                                    std::vector<SamplesIdx_t>&  samples,
                                    SamplesIdx_t                pos);

        /// Calculate impurity of samples with missing values and samples with values.
        /**
        Assuming: number of missing values > 0 <br>
        Assuming: calculate_NA_histogram()
         */
        void calculate_NA_impurity();

        /// Calculate impurity improvement from the current node to its children
        /// assuming a split between missing values and values.
        /**
        Assuming: number of missing values > 0 <br>
        Assuming: calculate_node_impurity(), calculate_NA_impurity()
         */
        double calculate_NA_impurity_improvement();

        /// Initialize class histograms for using a threshold on samples with values,
        /// in the case that all samples have values.
        /**
        Assuming: calculate_node_histogram()
         */
        void init_threshold_histograms();

        /// Initialize class histograms for using a threshold on samples with values,
        /// in the case that there are also samples with missing values.
        /**
        Assuming: number of missing values > 0 <br>
        Assuming: calculate_NA_histogram()
         */
        void init_threshold_values_histograms();

        /// Update class histograms for using a threshold on values,
        /// from current position to the new position (correspond to thresholds).
        /**
        Assuming: new_pos > pos <br>
        Assuming: init_threshold_histograms() or init_threshold_values_histograms()
        */
        void update_threshold_histograms(Classes_t*                  y,
                                         std::vector<SamplesIdx_t>&  samples,
                                         SamplesIdx_t                new_pos);

        /// Calculate impurity of samples with values that are smaller and greater than a threshold.
        /**
        Assuming: update_threshold_histograms()
         */
        void calculate_threshold_impurity();

        /// Calculate the impurity of samples with values that are smaller and greater than a threshold
        /// and passing on the samples with missing values.
        /**
        Assuming: number of missing values > 0 <br>
        Assuming: update_threshold_histograms(), calculate_NA_histograms()
         */
        void calculate_threshold_NA_impurity();

        /// Calculate the impurity improvement from the current node to its children
        /// assuming a split of the samples with values smaller and greater than a threshold
        /// in the case that all samples have values.
        /**
        Assuming: calculate_node_impurity(), calculate_threshold_impurity()
         */
        double calculate_threshold_impurity_improvement();

        /// Calculate the impurity improvement from the current node to its children
        /// assuming a split of the samples with values smaller and greater than a threshold
        /// in the case that there are also samples with missing values.
        /**
        Assuming: calculate_NA_impurity(), calculate_threshold_impurity()
         */
        double calculate_threshold_values_impurity_improvement();

        /// Calculate the impurity improvement from the current node to its children
        /// assuming a split of the samples with values smaller and greater than a threshold
        /// and passing on the samples with missing values to the left child.
        /**
        Assuming: calculate_NA_impurity(), calculate_threshold_impurity(), calculate_threshold_NA_impurity()
         */
        double calculate_threshold_NA_left_impurity_improvement();

        /// Calculate the impurity improvement from the current node to its children
        /// assuming a split of the samples with values smaller and greater than a threshold
        /// and passing on the samples with missing values to the right child.
        /**
        Assuming: calculate_NA_impurity(), calculate_threshold_impurity(), calculate_threshold_NA_impurity()
         */
        double calculate_threshold_NA_right_impurity_improvement();

        std::vector<Histogram_t> get_node_weighted_histogram() {
                                 return GiniCriterion::node_weighted_histogram; }
        double                   get_node_impurity() {
                                 return GiniCriterion::node_impurity; }
        double                   get_node_impurity_NA() {
                                 return GiniCriterion::node_impurity_NA; }
        double                   get_node_impurity_values() {
                                 return GiniCriterion::node_impurity_values; }
        double                   get_node_impurity_threshold_left() {
                                 return GiniCriterion::node_impurity_threshold_left; }
        double                   get_node_impurity_threshold_right() {
                                 return GiniCriterion::node_impurity_threshold_right; }
    };

// =============================================================================
// Node Splitter
// =============================================================================

    /// Splitter to find the best split for a node.
    class BestSplitter {

    protected:
        FeaturesIdx_t              n_features;
        SamplesIdx_t               n_samples;
        FeaturesIdx_t              max_features;
        unsigned long              max_thresholds;
        RandomState                random_state;
        // Samples
        // samples[start, end] is a LUT to the training data X, y
        // to handle the recursive partitioning and
        // the sorting of the data efficiently.
        std::vector<SamplesIdx_t>  samples; // vector created in initialization list
        SamplesIdx_t               start;
        SamplesIdx_t               end;
    public:
        // Gini Criterion
        GiniCriterion              criterion; // nested object created in initialization list

    public:
        /// Create and initialize a new best splitter.
        BestSplitter(ClassesIdx_t     n_classes, // required: 2 <= n_classes
                     FeaturesIdx_t    n_features, // required: 1 <= n_features
                     SamplesIdx_t     n_samples, // required: 2 <= n_samples
                     ClassWeights_t*  class_weight,
                     FeaturesIdx_t    max_features, // required: 0 < max_features <= n_features
                     unsigned long    max_thresholds, // required: 0, 1
                     RandomState const& random_state);

        /// Initialize node and calculate histogram and impurity for the node.
        void init_node(Classes_t*    y,
                       SamplesIdx_t  start,
                       SamplesIdx_t  end);

        /// Find the best split and partition (actually sorted) samples for a given feature.
        void split_feature(Features_t*                 X,
                           Classes_t*                  y,
                           std::vector<SamplesIdx_t>&  s,
                           FeaturesIdx_t               feature,
                           int&                        NA,
                           Features_t&                 threshold,
                           SamplesIdx_t&               pos,
                           double&                     improvement);

        /// Find a split and partition samples for a given feature
        /// using the Extreme Random Tree formulation for the threshold.
        void split_feature_extreme_random(Features_t*                 X,
                                          Classes_t*                  y,
                                          std::vector<SamplesIdx_t>&  s,
                                          FeaturesIdx_t               feature,
                                          int&                        NA,
                                          Features_t&                 threshold,
                                          SamplesIdx_t&               pos,
                                          double&                     improvement);

        /// Find the best split and partition (actually sorted) samples.
        /**
        Find the split (feature, threshold) on samples[start:end] and
        partition samples[start:end] into samples[start:pos] and samples[pos:end]
        according to split.

        Assuming: init_node()
         */
        void split_node(Features_t*     X,
                        Classes_t*      y,
                        FeaturesIdx_t&  feature,
                        int&            NA,
                        Features_t&     threshold,
                        SamplesIdx_t&   pos,
                        double&         improvement);
    };

// =============================================================================
// Tree Builder
// =============================================================================

    /// Build a binary decision tree in depth-first order.
    class DepthFirstTreeBuilder {

    protected:
        TreeDepthIdx_t  max_depth;
        std::string     missing_values;
        // Best Splitter (and Gini Criterion)
        BestSplitter    splitter; // nested object created in initialization list

    public:
        /// Create and initialize a new depth first tree builder.
        /**
        @param[in]  n_classes       Number of classes in the training data, minimum 2.
        @param[in]  n_features      Number of features in the training data, minimum 1.
        @param[in]  n_samples       Number of samples in the training data, minimum 2.
        @param[in]  class_weight    A weight for each class, which should be
        inversely proportional to the class frequencies in the training data for class balancing,
        or 1.0 otherwise.
        @param[in]  max_depth       The depth of the tree is expanded until
        the specified maximum depth of the tree is reached or
        all leaves are pure or no further impurity improvement can be achieved.
        @param[in]  max_features    Number of random features to consider
        when looking for the best split at each node, between 1 and n_features. <br>
        Note: the search for a split does not stop until at least one valid partition of the node samples is found
        up to the point that all features have been considered,
        even if it requires to effectively inspect more than max_features features.
        @param[in]  max_thresholds  Number of random thresholds to consider
        when looking for the best split at each node, 0 or 1. <br>
        If 0, then all thresholds, based on the mid-point of the node samples, are considered. <br>
        If 1, then consider 1 random threshold.
        @param[in]  missing_values      Handling of missing values. <br>
        string "NMAR" or "None", (default="None") <br>
        If "NMAR" (Not Missing At Random), then during training: the split criterion considers missing values
        as another category and samples with missing values are passed to either the left or the right child
        depending on which option provides the best split,
        and then during testing: if the split criterion includes missing values,
        a missing value is dealt with accordingly (passed to left or right child),
        or if the split criterion does not include missing values,
        a missing value at a split criterion is dealt with by combining the results from both children
        proportionally to the number of samples that are passed to the children during training. <br>
        If "None", an error is raised if one of the features has a missing value. <br>
        An option is to use imputation (fill-in) of missing values prior to using the decision tree classifier.
        @param[in]  random_state    Initialized Random Number Generator.

        "Decision Tree": max_features=n_features, max_thresholds=0.

         The following configurations should only be used for "decision forests": <br>
        "Random Tree": max_features<n_features, max_thresholds=0. <br>
        "Extreme Randomized Trees (ET)": max_features=n_features, max_thresholds=1. <br>
        "Totally Randomized Trees": max_features=1, max_thresholds=1, very similar to "Perfect Random Trees (PERT)".
        */
        DepthFirstTreeBuilder(ClassesIdx_t          n_classes,
                              FeaturesIdx_t         n_features,
                              SamplesIdx_t          n_samples,
                              ClassWeights_t*       class_weight,
                              TreeDepthIdx_t        max_depth,
                              FeaturesIdx_t         max_features,
                              unsigned long         max_thresholds,
                              std::string           missing_values,
                              RandomState const&    random_state);

        /// Build a binary decision tree from the training data.
        /**
        @param[in, out]  tree       A binary decision tree.
        @param[in]       X          Training input samples [n_samples x n_features].
        @param[in]       y          Target class labels corresponding to the training input samples [n_samples].
        @param[in]       n_samples  Number of samples, minimum 2.
        */
        // Using 1d array addressing for X and y to support efficient Cython bindings to Python using memory views.
        void build(Tree&         tree,
                   Features_t*   X,
                   Classes_t*    y,
                   SamplesIdx_t  n_samples);
    };

// =============================================================================
// Decision Tree Classifier
// =============================================================================

    /// A decision tree classifier.
    class DecisionTreeClassifier {

    protected:
        std::vector<std::string>  classes;
        ClassesIdx_t              n_classes;
        std::vector<std::string>  features;
        FeaturesIdx_t             n_features;

        // Hyperparameters
        std::string               class_balance;
        TreeDepthIdx_t            max_depth;
        FeaturesIdx_t             max_features;
        unsigned long             max_thresholds;
        std::string               missing_values;

        // Random Number Generator
        RandomState               random_state;

        // Model
        Tree                      tree_; // underlying estimator

    public:
        /// Create and initialize a new decision tree classifier.
        /**
        @param[in]  classes            Class labels.
        @param[in]  n_classes          Number of classes = number of class labels.
        @param[in]  features           Feature names.
        @param[in]  n_features         Number of features = number of feature names.
        @param[in]  class_balance      Weighting of the classes. <br>
        string "balanced" or "None", (default="balanced") <br>
        If "balanced", then the values of y are used to automatically adjust class weights
        inversely proportional to class frequencies in the input data. <br>
        If "None", all classes are supposed to have weight one.
        @param[in]  max_depth          The maximum depth of the tree. <br>
        The depth of the tree is expanded until the specified maximum depth of the tree is reached
        or all leaves are pure or no further impurity improvement can be achieved. <br>
        integer (default=3) <br>
        If 0 the maximum depth of the tree is set to max long (2^31-1).
        @param[in]  max_features       Number of random features to consider
        when looking for the best split at each node, between 1 and n_features. <br>
        Note: the search for a split does not stop until at least one valid partition of the node samples is found
        up to the point that all features have been considered,
        even if it requires to effectively inspect more than max_features features. <br>
        integer (default=0) <br>
        If 0 the number of random features = number of features. <br>
        Note: only to be used by Decision Forest
        @param[in]  max_thresholds     Number of random thresholds to consider
        when looking for the best split at each node. <br>
        integer (default=0) <br>
        If 0, then all thresholds, based on the mid-point of the node samples, are considered. <br>
        If 1, then consider 1 random threshold, based on the `Extreme Randomized Tree` formulation. <br>
        Note: only to be used by Decision Forest
        @param[in]  missing_values      Handling of missing values. <br>
        string "NMAR" or "None", (default="None") <br>
        If "NMAR" (Not Missing At Random), then during training: the split criterion considers missing values
        as another category and samples with missing values are passed to either the left or the right child
        depending on which option provides the best split,
        and then during testing: if the split criterion includes missing values,
        a missing value is dealt with accordingly (passed to left or right child),
        or if the split criterion does not include missing values,
        a missing value at a split criterion is dealt with by combining the results from both children
        proportionally to the number of samples that are passed to the children during training. <br>
        If "None", an error is raised if one of the features has a missing value. <br>
        An option is to use imputation (fill-in) of missing values prior to using the decision tree classifier.
        @param[in]  random_state_seed  Seed used by the random number generator. <br>
        integer (default=0) <br>
        If -1, then the random number generator is seeded with the current system time. <br>
        Note: only to be used by Decision Forest

        "Decision Tree": max_features=n_features, max_thresholds=0.

        The following configurations should only be used for "decision forests": <br>
        "Random Tree": max_features<n_features, max_thresholds=0. <br>
        "Extreme Randomized Trees (ET)": max_features=n_features, max_thresholds=1. <br>
        "Totally Randomized Trees": max_features=1, max_thresholds=1, very similar to "Perfect Random Trees (PERT)".
        */
        DecisionTreeClassifier(std::vector<std::string>  classes,
                               ClassesIdx_t              n_classes,
                               std::vector<std::string>  features,
                               FeaturesIdx_t             n_features,
                               std::string const&        class_balance = "balanced",
                               TreeDepthIdx_t            max_depth = 0,
                               FeaturesIdx_t             max_features = 0,
                               unsigned long             max_thresholds = 0,
                               std::string const&        missing_values = "None",
                               long                      random_state_seed = 0);

        /// Build a decision tree classifier from the training data.
        /**
        @param[in]  X          Training input samples [n_samples x n_features].
        @param[in]  y          Target class labels corresponding to the training input samples [n_samples].
        @param[in]  n_samples  Number of samples, minimum 2.
        */
        void fit(Features_t*   X,
                 Classes_t*    y,
                 SamplesIdx_t  n_samples);

        /// Predict classes probabilities for the test data.
        /**
        @param[in]      X          Test input samples [n_samples x n_features].
        @param[in]      n_samples  Number of samples in the test data.
        @param[in,out]  y_prob     Class probabilities corresponding to the test input samples [n_samples x n_classes].
        */
        void  predict_proba(Features_t*   X,
                            SamplesIdx_t  n_samples,
                            double*       y_prob);

        /// Predict classes for the test data.
        /**
        @param[in]      X          Test input samples [n_samples x n_features].
        @param[in]      n_samples  Number of samples in the test data.
        @param[in,out]  y          Predicted classes for the test input samples [n_samples].
        */
        void  predict(Features_t*   X,
                      SamplesIdx_t  n_samples,
                      Classes_t*    y);

        /// Calculate score for the test data.
        /**
        @param[in]      X          Test input samples [n_samples x n_features].
        @param[in]      y          True classes for the test input samples [n_samples].
        @param[in]      n_samples  Number of samples in the test data.
        @return                    Score.
        */
        double score(Features_t*   X,
                     Classes_t*    y,
                     SamplesIdx_t  n_samples);

        /// Calculate feature importances from the decision tree.
        /**
        @param[in,out]  importances  Feature importances corresponding to all features [n_features]. <br>
        The higher, the more important the feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.
        */

        void  calculate_feature_importances(double*  importances);

        /// Export of a decision tree in GraphViz dot format.
        /**
        @param[in]  filename  Filename of GraphViz dot file, extension .gv added.
        @param[in]  rotate    Rotate display of decision tree. <br>
        boolean (default=false) <br>
        If false, then orient tree top-down. <br>
        If true, then orient tree left-to-right. <br>
        Ubuntu: <br>
        sudo apt-get install graphviz <br>
        sudo apt-get install xdot <br>
        view <br>
        $: xdot filename.gv <br>
        create pdf, png <br>
        $: dot -Tpdf filename.gv -o filename.pdf <br>
        $: dot -Tpng filename.gv -o filename.png <br>
        Windows: <br>
        Install graphviz-2.38.msi from http://www.graphviz.org/Download_windows.php <br>
        START> "Advanced System Settings" <br>
        Click "Environmental Variables ..." <br>
        Click "Browse..." Select "C:/ProgramFiles(x86)/Graphviz2.38/bin" <br>
        view <br>
        START> gvedit
        */
        void  export_graphviz(std::string const& filename, bool rotate=false);

        /// Export of a decision tree in a simple text format.
        std::string  export_text();

        /// Export of a decision tree classifier in binary serialized format.
        /**
        @param[in]  filename  Filename of binary serialized file, extension .dt added.
        */
        void  export_serialize(std::string const& filename);

        /// Import of a decision tree classifier in binary serialized format.
        /**
        @param[in]  filename  Filename of binary serialized file.
        */
        static  DecisionTreeClassifier  import_deserialize(std::string const& filename);

        /// Serialize
        void  serialize(std::ofstream& fout);
        /// Deserialize
        static  DecisionTreeClassifier  deserialize(std::ifstream& fin);
    };

} // namespace koho

#endif
