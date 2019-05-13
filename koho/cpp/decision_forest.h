/// Decision Forest module.
/** @file
- Classification
- n Decision Trees with soft voting
- Important Features

C++ implementation.
*/

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

#ifndef KOHO_DECISION_FOREST_H
#define KOHO_DECISION_FOREST_H

#include "decision_tree.h"

namespace koho {

// =============================================================================
// Decision Forest Classifier
// =============================================================================

    /// A decision forest classifier.
    class DecisionForestClassifier {

    protected:
        std::vector<std::string>  classes;
        ClassesIdx_t              n_classes;
        std::vector<std::string>  features;
        FeaturesIdx_t             n_features;

        // Hyperparameters
        unsigned long             n_estimators;
        bool                      bootstrap;
        bool                      oob_score;
        std::string               class_balance;
        TreeDepthIdx_t            max_depth;
        FeaturesIdx_t             max_features;
        unsigned long             max_thresholds;
        std::string               missing_values;

        // Random Number Generator
        RandomState               random_state;

        // Model
        std::vector<DecisionTreeClassifier> dtc_; // underlying sub-estimators

        // Performance Characteristics
        double                    oob_score_; // Out_Of-Bag estimate

    public:
        /// Create and initialize a new decision forest classifier.
        /**
        @param[in]  classes            Class labels.
        @param[in]  n_classes          Number of classes = number of class labels.
        @param[in]  features           Feature names.
        @param[in]  n_features         Number of features = number of feature names.
        @param[in]  n_estimators       Number of decision trees in the forest. <br>
        If 1, then the decision forest classifier is a decision tree classifier. <br>
        integer (default=10)
        @param[in]  bootstrap          Whether bootstrap samples are used when building trees. <br>
        Out-of-bag samples are used to estimate the generalization accuracy. <br>
        boolean (default=true)
        @param[in]  oob_score          Whether to use out-of-bag samples to estimate the generalization accuracy. <br>
        boolean (default=false)
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

        "Decision Tree": n_estimators=1, max_features=n_features, max_thresholds=0.

        The following configurations should only be used for "decision forests": <br>
        "Random Tree": max_features<n_features, max_thresholds=0. <br>
        "Extreme Randomized Trees (ET)": max_features=n_features, max_thresholds=1. <br>
        "Totally Randomized Trees": max_features=1, max_thresholds=1, very similar to "Perfect Random Trees (PERT)".
        */
        DecisionForestClassifier(std::vector<std::string>   classes,
                                 ClassesIdx_t               n_classes,
                                 std::vector<std::string>   features,
                                 FeaturesIdx_t              n_features,
                                 unsigned long              n_estimators = 100,
                                 bool                       bootstrap = false,
                                 bool                       oob_score = false,
                                 std::string const&         class_balance = "balanced",
                                 TreeDepthIdx_t             max_depth = 3,
                                 FeaturesIdx_t              max_features = 0,
                                 unsigned long              max_thresholds = 0,
                                 std::string const&         missing_values = "None",
                                 long                       random_state_seed = 0);

        /// Build a decision forest classifier from the training data.
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
        void predict_proba(Features_t *X,
                           SamplesIdx_t n_samples,
                           double *y_prob);

        /// Predict classes for the test data.
        /**
        @param[in]      X          Test input samples [n_samples x n_features].
        @param[in]      n_samples  Number of samples in the test data.
        @param[in,out]  y          Predicted classes for the test input samples [n_samples].
        */
        void predict(Features_t *X,
                     SamplesIdx_t n_samples,
                     Classes_t *y);

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

        /// Calculate feature importances from the decision forest.
        /**
        @param[in,out]  importances  Feature importances corresponding to all features [n_features]. <br>
        The higher, the more important the feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.
        */
        void  calculate_feature_importances(double*  importances);

        /// Export of a decision forest as individual decision trees in GraphViz dot format.
        /**
        @param[in]  filename  Common filename of individual decision trees,
        "filename_<0, ... n_estimators-1>" used for the individual decision trees, extension .gv added.
        @param[in]  rotate    Rotate display of decision trees. <br>
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

        /// Export of a decision tree from a decision forest in GraphViz dot format.
        /**
        @param[in]  filename  Common filename of individual decision trees,
        "filename_<0, ... n_estimators-1>" used for the individual decision trees, extension .gv added.
        @param[in]  e         Decision tree index 0, ... n_estimators-1.
        @param[in]  rotate    Rotate display of decision trees. <br>
        boolean (default=false) <br>
        If false, then orient tree top-down. <br>
        If true, then orient tree left-to-right.
        */
        void export_graphviz(std::string const& filename, unsigned long e, bool rotate);

        /// Export of a decision tree from a decision forest in a simple text format.
        /**
        @param[in]  e         Decision tree index 0, ... n_estimators-1.
        */
        std::string  export_text(unsigned long e);

        /// Export of a decision forest classifier in binary serialized format
        /// with separate files for the individual decision trees.
        /**
        @param[in]  filename  Filename of decision forest, extension .dfc added,
        "filename_<0, ... n_estimators-1>" used for the individual decision trees, extension .dtc added.
        */
        void  export_serialize(std::string const& filename);

        /// Import of a decision forest classifier in binary serialized format
        /// with separate files for the individual decision trees.
        /**
        @param[in]  filename  Filename of decision forest, extension .dfc added,
        "filename_<0, ... n_estimators-1>" used for the individual decision trees, extension .dtc added.
        */
        static  DecisionForestClassifier  import_deserialize(std::string const& filename);

        /// Serialize
        void  serialize(std::ofstream& fout);
        /// Deserialize
        static  DecisionForestClassifier  deserialize(std::ifstream& fin);
    };

} // namespace koho

#endif
