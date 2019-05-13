/// Decision Tree module.
/** @file
*/

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

// Basic concepts for the implementation of the classifier are based on
// G. Louppe, “Understanding Random Forests”, PhD Thesis, 2014

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>

#include "utilities.h"
#include "decision_tree.h"

using namespace std;

namespace koho {

// =============================================================================
// Tree
// =============================================================================

    // Create a new node.

    Node::Node(NodesIdx_t                  left_child,
               NodesIdx_t                  right_child,
               FeaturesIdx_t               feature,
               int                         NA,
               Features_t                  threshold,
               const vector<Histogram_t>&  histogram,
               double                      impurity,
               double                      improvement)
    : left_child(left_child),
      right_child(right_child),
      feature(feature),
      NA(NA),
      threshold(threshold),
      histogram(histogram),
      impurity(impurity),
      improvement(improvement) { }

    // Serialize

    void  Node::serialize(ofstream& fout) {

        fout.write((const char *) (&left_child), sizeof(left_child));
        fout.write((const char *) (&right_child), sizeof(right_child));
        fout.write((const char *) (&feature), sizeof(feature));
        fout.write((const char *) (&NA), sizeof(NA));
        fout.write((const char *) (&threshold), sizeof(threshold));

        unsigned long  size = histogram.size();
        fout.write((const char *) (&size), sizeof(size));
        for (unsigned long i=0; i<size; ++i) {
            fout.write((const char *) (&histogram[i]), sizeof(histogram[i]));
        }

        fout.write((const char *) (&impurity), sizeof(impurity));
        fout.write((const char *) (&improvement), sizeof(improvement));

    }

    // Deserialize

    Node  Node::deserialize(ifstream& fin) {

        NodesIdx_t           left_child;
        NodesIdx_t           right_child;
        FeaturesIdx_t        feature;
        int                  NA;
        Features_t           threshold;
        vector<Histogram_t>  histogram;
        double               impurity;
        double               improvement;

        fin.read((char*)(&left_child), sizeof(left_child));
        fin.read((char*)(&right_child), sizeof(right_child));
        fin.read((char*)(&feature), sizeof(feature));
        fin.read((char*)(&NA), sizeof(NA));
        fin.read((char*)(&threshold), sizeof(threshold));

        unsigned long  size;
        fin.read((char*)(&size), sizeof(size));
        for (unsigned long i=0; i<size; ++i) {
            Histogram_t value;
            fin.read((char*)(&value), sizeof(value));
            histogram.emplace_back(value);
        }

        fin.read((char*)(&impurity), sizeof(impurity));
        fin.read((char*)(&improvement), sizeof(improvement));

        return Node(left_child,
                    right_child,
                    feature,
                    NA,
                    threshold,
                    histogram,
                    impurity,
                    improvement);
    }

    // Create a new tree without nodes.

    Tree::Tree(ClassesIdx_t   n_classes,
               FeaturesIdx_t  n_features)
    : n_classes(n_classes),
      n_features(n_features),
      max_depth(0),
      node_count(0) { }

    // Add a new node to the tree.

    NodesIdx_t  Tree::add_node(TreeDepthIdx_t              depth,
                               NodesIdx_t                  parent_id,
                               bool                        is_left,
                               FeaturesIdx_t               feature,
                               int                         NA,
                               Features_t                  threshold,
                               const vector<Histogram_t>&  histogram,
                               double                      impurity,
                               double                      improvement) {

        nodes.emplace_back(0, 0, // children IDs are set when the child nodes are added
                           feature, NA, threshold,
                           histogram, impurity, improvement);

        // Register new node as the child of its parent

        NodesIdx_t  node_id = Tree::node_count++;
        if (depth > 0) { // not root node
            if (is_left) { Tree::nodes[parent_id].left_child  = node_id;
            } else {       Tree::nodes[parent_id].right_child = node_id;
            }
        }

        if (depth > Tree::max_depth) Tree::max_depth = depth;

        return node_id;
    }

    // Predict classes probabilities for the test data.

    void  Tree::predict(Features_t*   X,
                        SamplesIdx_t  n_samples,
                        double*       y_prob) {

         // Index Stack
         struct IdxInfo {

             NodesIdx_t  idx;
             double      weight;

             IdxInfo(NodesIdx_t  idx,
                     double      weight)
             : idx(idx), weight(weight) { };
         };

        // Initialize
        memset(y_prob, 0, n_samples * Tree::n_classes * sizeof(double));

        // Loop: samples
        for(SamplesIdx_t i=0; i<n_samples; i++) {

            // node index stacks to deal with the evaluation of multiple paths
            stack<IdxInfo>  node_idx_stack;
            stack<IdxInfo>  leaf_idx_stack;

            // Go to the root node
            node_idx_stack.emplace(IdxInfo(0, 1.0));
            // Loop: root to leaf node

            while (!node_idx_stack.empty()) { // evaluation of multiple paths
	            IdxInfo node_idx = node_idx_stack.top();
	            node_idx_stack.pop();

                while (Tree::nodes[node_idx.idx].left_child > 0) { // follow path until leaf node
                // leaf nodes do no have any children
                // so we only need to test for one of the children

                    if (isnan(X[i*n_features + Tree::nodes[node_idx.idx].feature])) { // missing value

                        // Split criterion includes missing values
                        // Go to left or right child node depending on split (NA)
                        if (Tree::nodes[node_idx.idx].NA == 0) // left
                            node_idx.idx = Tree::nodes[node_idx.idx].left_child;
                        else if (Tree::nodes[node_idx.idx].NA == 1) // right
                            node_idx.idx = Tree::nodes[node_idx.idx].right_child;

                        else { // Split criterion does NOT includes missing values
                            IdxInfo node_idx2 = node_idx;
			                node_idx2.idx = Tree::nodes[node_idx.idx].right_child;  // right child
			                double n_right = accumulate(Tree::nodes[node_idx2.idx].histogram.begin(),
			                                            Tree::nodes[node_idx2.idx].histogram.end(), 0.0);
			                node_idx.idx = Tree::nodes[node_idx.idx].left_child; // left child
			                double n_left = accumulate(Tree::nodes[node_idx.idx].histogram.begin(),
			                                           Tree::nodes[node_idx.idx].histogram.end(), 0.0);
			                node_idx.weight  *= n_left  / (n_left + n_right); // adjust weights
			                node_idx2.weight *= n_right / (n_left + n_right);
                            node_idx_stack.emplace(node_idx2); // add right child as new path to node index stack
                            // continue left child path until leaf node
                        }
	                } else { // value

                        // Go to left or right child node depending on split (feature, threshold)
                        if (X[i*Tree::n_features + Tree::nodes[node_idx.idx].feature] <=
                            Tree::nodes[node_idx.idx].threshold)
                            node_idx.idx = Tree::nodes[node_idx.idx].left_child;
                        else
                            node_idx.idx = Tree::nodes[node_idx.idx].right_child;
                    }
                }
                leaf_idx_stack.emplace(node_idx); // store leaf nodes
            }

            while (!leaf_idx_stack.empty()) { // combine results from all leaf nodes
	            IdxInfo leaf_idx = leaf_idx_stack.top();
	            leaf_idx_stack.pop();

                // Calculate classes probabilities
                // based on number of samples per class histogram
                double  normalizer = 0.0;
                for (ClassesIdx_t c=0; c<Tree::n_classes; ++c)
                    normalizer += Tree::nodes[leaf_idx.idx].histogram[c];
                if (normalizer > 0.0) {
                    for (ClassesIdx_t c = 0; c < Tree::n_classes; ++c) {
                        y_prob[i * Tree::n_classes + c] +=
                            leaf_idx.weight * Tree::nodes[leaf_idx.idx].histogram[c] / normalizer;
                    }
                }
            }
        }

    }

    // Calculate feature importances from the decision tree.

    void  Tree::calculate_feature_importances(double* importances) {

        // Initialize
        memset(importances, 0, Tree::n_features * sizeof(double));

        if (Tree::node_count == 0) return;

        // Loop: all nodes
        for (NodesIdx_t idx=0; idx<Tree::node_count; ++idx) {

            // Split node
            // leaf nodes do no have any children
            // so we only need to test for one of the children
            if (Tree::nodes[idx].left_child > 0) {
                // Accumulate improvements per feature
                importances[Tree::nodes[idx].feature] += Tree::nodes[idx].improvement;
            }
        }

        // Normalize (to 1)
        double  normalizer = 0.0;
        for (FeaturesIdx_t f=0; f<Tree::n_features; ++f) normalizer += importances[f];
        if (normalizer > 0.0) { // 0 when root is pure
            for (FeaturesIdx_t f=0; f<Tree::n_features; ++f) {
                importances[f] = importances[f] / normalizer;
            }
        }

    }

    // Serialize

    void  Tree::serialize(ofstream& fout) {

        fout.write((const char *)&n_classes, sizeof(n_classes));
        fout.write((const char *)&n_features, sizeof(n_features));
        fout.write((const char *)&max_depth, sizeof(max_depth));
        fout.write((const char *)&node_count, sizeof(node_count));

        // Serialize Nodes
        for (NodesIdx_t i=0; i<node_count; ++i) {
            nodes[i].serialize(fout);
        }

    }

    // Deserialize

    void  Tree::deserialize(ifstream& fin) {

        fin.read((char*)(&n_classes), sizeof(n_classes));
        fin.read((char*)(&n_features), sizeof(n_features));
        fin.read((char*)(&max_depth), sizeof(max_depth));
        fin.read((char*)(&node_count), sizeof(node_count));

        // Deserialize Nodes
        for (NodesIdx_t i=0; i<node_count; ++i) {
            nodes.emplace_back(Node::deserialize(fin));
        }

    }

// =============================================================================
// Impurity Criterion
// =============================================================================

    // Create and initialize a new gini criterion.

    GiniCriterion::GiniCriterion(ClassesIdx_t     n_classes,
                                 SamplesIdx_t     n_samples,
                                 ClassWeights_t*  class_weight)
    : n_classes(n_classes),
      n_samples(n_samples),
      class_weight(class_weight),
      // Create and initialize histograms
      // - all samples
      node_weighted_histogram(n_classes, 0),
      node_weighted_n_samples(0.0),
      node_impurity(0.0),
      // - samples with missing values
      node_weighted_histogram_NA(n_classes, 0),
      node_weighted_n_samples_NA(0.0),
      node_impurity_NA(0.0),
      // - samples with values
      node_weighted_histogram_values(n_classes, 0),
      node_weighted_n_samples_values(0.0),
      node_impurity_values(0.0),
      node_pos_NA(0),
      // - samples with values smaller than threshold (assigned to left child)
      node_weighted_histogram_threshold_left(n_classes, 0),
      node_weighted_n_samples_threshold_left(0.0),
      node_impurity_threshold_left(0.0),
      // -- plus missing values (assigned to left child)
      node_weighted_n_samples_threshold_left_NA(0.0),
      node_impurity_threshold_left_NA(0.0),
      // - samples with values greater than threshold (assigned to right child)
      node_weighted_histogram_threshold_right(n_classes, 0),
      node_weighted_n_samples_threshold_right(0.0),
      node_impurity_threshold_right(0.0),
      // -- plus missing values (assigned to right child)
      node_weighted_n_samples_threshold_right_NA(0.0),
      node_impurity_threshold_right_NA(0.0),
      node_pos_threshold(0) { }

    // Calculate weighted class histogram for current node.

    void  GiniCriterion::calculate_node_histogram(Classes_t*             y,
                                                  vector<SamplesIdx_t>&  samples,
                                                  SamplesIdx_t           start,
                                                  SamplesIdx_t           end) {

        // Calculate class histogram
        vector<Histogram_t>  histogram(GiniCriterion::n_classes, 0);
        for (SamplesIdx_t i=start; i<end; ++i) {
            histogram[y[samples[i]]]++;
        }

        // Apply class weights
        GiniCriterion::node_weighted_n_samples = 0.0;
        Histogram_t  weighted_cnt;
        for (ClassesIdx_t c=0; c<GiniCriterion::n_classes; ++c) {
            weighted_cnt = GiniCriterion::class_weight[c] * histogram[c];
            GiniCriterion::node_weighted_histogram[c] = weighted_cnt;
            GiniCriterion::node_weighted_n_samples += weighted_cnt;
        }

    }

    // Calculate impurity of a weighted class histogram using the Gini criterion.

    double  GiniCriterion::calculate_impurity(vector<Histogram_t>&  histogram) {

        double  cnt;
        double  sum_cnt = 0.0;
        double  sum_cnt_sq = 0.0;

        for (ClassesIdx_t c=0; c<histogram.size(); ++c) {
            cnt = static_cast<double>(histogram[c]);
            sum_cnt += cnt;
            sum_cnt_sq += cnt * cnt;
        }

        return (sum_cnt > 0.0) ? (1.0 - sum_cnt_sq / (sum_cnt*sum_cnt)) : 0.0;
    }

    // Calculate impurity of the current node.

    void  GiniCriterion::calculate_node_impurity() {

        GiniCriterion::node_impurity =
            GiniCriterion::calculate_impurity(GiniCriterion::node_weighted_histogram);

    }

    // Calculate class histograms for the samples with missing values and the samples with values.

    void GiniCriterion::calculate_NA_histogram(Classes_t*                  y,
                                               std::vector<SamplesIdx_t>&  samples,
                                               SamplesIdx_t                pos) {

        // Calculate class histogram for the samples with missing values located in samples[0:pos]
        vector<Histogram_t>  histogram(GiniCriterion::n_classes, 0);
        for (SamplesIdx_t i=0; i<pos; ++i) {
            histogram[y[samples[i]]]++;
        }
        // Apply class weights
        Histogram_t  weighted_cnt;
        GiniCriterion::node_weighted_n_samples_NA = 0.0;
        for (ClassesIdx_t c=0; c<GiniCriterion::n_classes; ++c) {
            weighted_cnt = GiniCriterion::class_weight[c] * histogram[c];
            GiniCriterion::node_weighted_histogram_NA[c] = weighted_cnt;
            GiniCriterion::node_weighted_n_samples_NA += weighted_cnt;
        }

        // Calculate class histogram for samples with values
        for (ClassesIdx_t c=0; c<GiniCriterion::n_classes; ++c) {
            GiniCriterion::node_weighted_histogram_values[c] = GiniCriterion::node_weighted_histogram[c] -
                                                               GiniCriterion::node_weighted_histogram_NA[c];
        }
        GiniCriterion::node_weighted_n_samples_values = GiniCriterion::node_weighted_n_samples -
                                                        GiniCriterion::node_weighted_n_samples_NA;

        // Update position
        GiniCriterion::node_pos_NA = pos;

    }

    // Calculate impurity of samples with missing values and samples with values.

    void  GiniCriterion::calculate_NA_impurity() {

        GiniCriterion::node_impurity_NA =
            GiniCriterion::calculate_impurity(GiniCriterion::node_weighted_histogram_NA);

        GiniCriterion::node_impurity_values =
            GiniCriterion::calculate_impurity(GiniCriterion::node_weighted_histogram_values);

    }

    // Calculate impurity improvement from the current node to its children
    // assuming a split between missing values and values.

    double  GiniCriterion::calculate_NA_impurity_improvement() {

        return (GiniCriterion::node_weighted_n_samples / GiniCriterion::n_samples) *
               (GiniCriterion::node_impurity -
                       (GiniCriterion::node_weighted_n_samples_NA / GiniCriterion::node_weighted_n_samples) *
                        GiniCriterion::node_impurity_NA -
                       (GiniCriterion::node_weighted_n_samples_values / GiniCriterion::node_weighted_n_samples) *
                        GiniCriterion::node_impurity_values);
        // OK to use "n_samples instead of "sum of all weighted samples"
        // given the way the class weights are calculated
    }

    // Initialize class histograms for using a threshold on samples with values,
    // in the case that all samples have values.

    void  GiniCriterion::init_threshold_histograms() {

        // Initialize class histogram for left child to 0
        // Initialize class histogram for right child to
        // class histogram from the current node
        for (ClassesIdx_t c=0; c<GiniCriterion::n_classes; ++c) {
            GiniCriterion::node_weighted_histogram_threshold_left[c] = 0.0;
            GiniCriterion::node_weighted_histogram_threshold_right[c] = GiniCriterion::node_weighted_histogram[c];
        }
        GiniCriterion::node_weighted_n_samples_threshold_left = 0.0;
        GiniCriterion::node_weighted_n_samples_threshold_right = GiniCriterion::node_weighted_n_samples;

        // Update current position
        GiniCriterion::node_pos_threshold = 0;

    }

    // Initialize class histograms for using a threshold on samples with values,
    // in the case that there are also samples with missing values.

    void  GiniCriterion::init_threshold_values_histograms() {

        // Initialize class histogram for left child to 0
        // Initialize class histogram for right child to
        // class histogram of samples with values (vs missing values) from the current node
        for (ClassesIdx_t c=0; c<GiniCriterion::n_classes; ++c) {
            GiniCriterion::node_weighted_histogram_threshold_left[c] = 0.0;
            GiniCriterion::node_weighted_histogram_threshold_right[c] = GiniCriterion::node_weighted_histogram_values[c];
        }
        GiniCriterion::node_weighted_n_samples_threshold_left = 0.0;
        GiniCriterion::node_weighted_n_samples_threshold_right = GiniCriterion::node_weighted_n_samples_values;

        // Update current position
        GiniCriterion::node_pos_threshold = GiniCriterion::node_pos_NA;

    }

    // Update class histograms for using a threshold on values,
    // from current position to the new position (correspond to thresholds).

    void  GiniCriterion::update_threshold_histograms(Classes_t*             y,
                                                     vector<SamplesIdx_t>&  samples,
                                                     SamplesIdx_t           new_pos) {

        // Calculate class histogram for samples[pos:new_pos]
        vector<Histogram_t>  histogram(GiniCriterion::n_classes, 0);
        for (SamplesIdx_t i=GiniCriterion::node_pos_threshold; i<new_pos; ++i) {
            histogram[y[samples[i]]]++;
        }

        // Add class histogram for samples[pos:new_pos]
        // to class histogram for samples[0 or number missing values:pos] with values < threshold (left child)
        // Subtract class histogram for samples[pos:new_pos]
        // from class histogram for samples[pos: number of samples] with values > threshold (right child)
        Histogram_t  weighted_cnt;
        for (ClassesIdx_t c=0; c<GiniCriterion::n_classes; ++c) {
            // Apply class weights
            weighted_cnt = GiniCriterion::class_weight[c] * histogram[c];
            // Left child
            GiniCriterion::node_weighted_histogram_threshold_left[c] += weighted_cnt;
            GiniCriterion::node_weighted_n_samples_threshold_left += weighted_cnt;
            // Right child
            GiniCriterion::node_weighted_histogram_threshold_right[c] -= weighted_cnt;
            GiniCriterion::node_weighted_n_samples_threshold_right -= weighted_cnt;
        }

        // Update current position (correspond to a threshold)
        GiniCriterion::node_pos_threshold = new_pos;

    }

    // Calculate impurity of samples with values that are smaller and greater than a threshold.
    void  GiniCriterion::calculate_threshold_impurity() {

        GiniCriterion::node_impurity_threshold_left =
            GiniCriterion::calculate_impurity(GiniCriterion::node_weighted_histogram_threshold_left);

        GiniCriterion::node_impurity_threshold_right =
            GiniCriterion::calculate_impurity(GiniCriterion::node_weighted_histogram_threshold_right);

    }

    // Calculate the impurity of samples with values that are smaller and greater than a threshold
    // and passing on the samples with missing values.

    void  GiniCriterion::calculate_threshold_NA_impurity() {

        vector<Histogram_t>  histogram(GiniCriterion::node_weighted_histogram_NA.size(), 0.0);

        // Samples with values that are smaller than a threshold and the samples with missing values
        for (ClassesIdx_t c=0; c<GiniCriterion::node_weighted_histogram_NA.size(); ++c) {
            histogram[c] = GiniCriterion::node_weighted_histogram_threshold_left[c] +
                           GiniCriterion::node_weighted_histogram_NA[c];
        }

        GiniCriterion::node_impurity_threshold_left_NA = GiniCriterion::calculate_impurity(histogram);

        GiniCriterion::node_weighted_n_samples_threshold_left_NA =
                GiniCriterion::node_weighted_n_samples_threshold_left + GiniCriterion::node_weighted_n_samples_NA;

        // Samples with values that are greater than a threshold and the samples with missing values
        for (ClassesIdx_t c=0; c<GiniCriterion::node_weighted_histogram_NA.size(); ++c) {
            histogram[c] = GiniCriterion::node_weighted_histogram_threshold_right[c] +
                           GiniCriterion::node_weighted_histogram_NA[c];
        }

        GiniCriterion::node_impurity_threshold_right_NA = GiniCriterion::calculate_impurity(histogram);

        GiniCriterion::node_weighted_n_samples_threshold_right_NA =
                GiniCriterion::node_weighted_n_samples_threshold_right + GiniCriterion::node_weighted_n_samples_NA;

    }

    // Calculate the impurity improvement from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // in the case that all samples have values.

    double  GiniCriterion::calculate_threshold_impurity_improvement() {

        return (GiniCriterion::node_weighted_n_samples / GiniCriterion::n_samples) *
               (GiniCriterion::node_impurity -
                       (GiniCriterion::node_weighted_n_samples_threshold_left /
                        GiniCriterion::node_weighted_n_samples) *
                        GiniCriterion::node_impurity_threshold_left -
                       (GiniCriterion::node_weighted_n_samples_threshold_right /
                        GiniCriterion::node_weighted_n_samples) *
                        GiniCriterion::node_impurity_threshold_right);
        // OK to use "n_samples instead of "sum of all weighted samples"
        // given the way the class weights are calculated
    }

    // Calculate the impurity improvement from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // in the case that there are also samples with missing values.

    double  GiniCriterion::calculate_threshold_values_impurity_improvement() {

        return (GiniCriterion::node_weighted_n_samples_values / GiniCriterion::n_samples) *
               (GiniCriterion::node_impurity_values -
                       (GiniCriterion::node_weighted_n_samples_threshold_left /
                        GiniCriterion::node_weighted_n_samples_values) *
                        GiniCriterion::node_impurity_threshold_left -
                       (GiniCriterion::node_weighted_n_samples_threshold_right /
                        GiniCriterion::node_weighted_n_samples_values) *
                        GiniCriterion::node_impurity_threshold_right);
        // OK to use "n_samples instead of "sum of all weighted samples"
        // given the way the class weights are calculated
    }

    // Calculate the impurity improvement from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // and passing on the samples with missing values to the left child.

    double  GiniCriterion::calculate_threshold_NA_left_impurity_improvement() {

        return (GiniCriterion::node_weighted_n_samples / GiniCriterion::n_samples) *
               (GiniCriterion::node_impurity -
                   (GiniCriterion::node_weighted_n_samples_threshold_left_NA /
                    GiniCriterion::node_weighted_n_samples) *
                    GiniCriterion::node_impurity_threshold_left_NA -
                   (GiniCriterion::node_weighted_n_samples_threshold_right /
                    GiniCriterion::node_weighted_n_samples) *
                    GiniCriterion::node_impurity_threshold_right);
    }

    // Calculate the impurity improvement from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // and passing on the samples with missing values to the right child.

    double  GiniCriterion::calculate_threshold_NA_right_impurity_improvement() {

        return (GiniCriterion::node_weighted_n_samples / GiniCriterion::n_samples) *
               (GiniCriterion::node_impurity -
                   (GiniCriterion::node_weighted_n_samples_threshold_left /
                    GiniCriterion::node_weighted_n_samples) *
                    GiniCriterion::node_impurity_threshold_left -
                   (GiniCriterion::node_weighted_n_samples_threshold_right_NA /
                    GiniCriterion::node_weighted_n_samples) *
                    GiniCriterion::node_impurity_threshold_right_NA);
    }

// =============================================================================
// Node Splitter
// =============================================================================

    // Create and initialize a new best splitter.

    BestSplitter::BestSplitter(ClassesIdx_t         n_classes,
                               FeaturesIdx_t        n_features,
                               SamplesIdx_t         n_samples,
                               ClassWeights_t*      class_weight,
                               FeaturesIdx_t        max_features,
                               unsigned long        max_thresholds,
                               RandomState const&   random_state)
    : n_features(n_features),
      n_samples(n_samples),
      max_features(max_features),
      max_thresholds(max_thresholds),
      random_state(random_state),
      // Create samples
      samples(n_samples),
      start(0),
      end(n_samples),
      // Create amd initialize a gini criterion
      criterion(n_classes, n_samples, class_weight) {

        // Initialize samples[0, n_samples] to the training data X, y
        iota(samples.begin(), samples.end(), 0); // identity mapping
    }

    // Initialize node and calculate weighted histogram and impurity for the node.

    void  BestSplitter::init_node(Classes_t*    y,
                                  SamplesIdx_t  start,
                                  SamplesIdx_t  end) {

        BestSplitter::start = start;
        BestSplitter::end = end;

        BestSplitter::criterion.calculate_node_histogram(y, BestSplitter::samples, start, end);
        BestSplitter::criterion.calculate_node_impurity();

    }

    // Find the best split and partition samples (actually sorted) for a given feature.

    void  BestSplitter::split_feature(Features_t*            X,
                                      Classes_t*             y,
                                      vector<SamplesIdx_t>&  s,
                                      FeaturesIdx_t          feature,
                                      int&                   NA,
                                      Features_t&            threshold,
                                      SamplesIdx_t&          pos,
                                      double&                improvement) {

        NA = -1; // no missing values
        threshold = std::numeric_limits<double>::quiet_NaN(); // no threshold
        pos = 0;
        improvement = 0.0; // leaf node

        // y is not constant (impurity > 0)
        // has been checked by impurity stop criteria in build()
        // moving on we can assume at least 2 samples

        // Copy f_X=X[samples[start:end],f]
        // training data X for the current node.

        SamplesIdx_t  n_samples = BestSplitter::end - BestSplitter::start;
        vector<Features_t>  f_X(n_samples);
        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            f_X[i] = X[s[i]*BestSplitter::n_features + feature];
        }

        // Detect samples with missing values and move them to the beginning of the samples vector
        SamplesIdx_t  pNA = 0;
        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            if (isnan(f_X[i])) {
                swap(f_X[i],f_X[pNA]);
                swap(s[i],s[pNA]);
                pNA++;
            }
        }

        if (pNA == n_samples) return; // Can not split feature when all values are NA
        // moving on, in case that there are missing values, we can assume that there is at least 1 sample with a value
        // and in case that there are no missing values, we can assume that there are at least 2 samples with a value

        // Split just based on missing values
        // ----------------------------------

        if (pNA > 0) { // missing values

            // Calculate class histograms for the samples with missing values and the samples with values
            BestSplitter::criterion.calculate_NA_histogram(y, s, pNA);
            // Calculate impurity of samples with missing values and samples with values
            BestSplitter::criterion.calculate_NA_impurity();
            // Calculate impurity improvement from the current node to its children
            // assuming a split between missing values and values
            improvement = BestSplitter::criterion.calculate_NA_impurity_improvement();
            NA = 0; // pass all samples with missing values to the left child
            // pass all samples with values to the right child
            threshold = std::numeric_limits<double>::quiet_NaN(); // no threshold
            pos = BestSplitter::start + pNA;

            // If impurity of all samples with values is 0.0 (pure) then stop
            // includes special case of having only 1 sample with a value
            if (BestSplitter::criterion.get_node_impurity_values() < PRECISION_EQUAL) return;
        }
        // moving on we can assume that there are at least 2 samples with a value

        // Split based on threshold
        // ------------------------

        // f_X is not constant
        Features_t  f_min, f_max;
        f_min = f_max = f_X[pNA];
        for (SamplesIdx_t i=(pNA+1); i<n_samples; ++i) {
            if (f_X[i] < f_min) f_min = f_X[i]; else if (f_X[i] > f_max) f_max = f_X[i];
        }

        if (f_min + PRECISION_EQUAL < f_max) {

            if (pNA == 0) {
                // Initialize class histograms for using a threshold
                BestSplitter::criterion.init_threshold_histograms();
            } else if (pNA > 0) {
                // Initialize class histograms for using a threshold on samples with values
                BestSplitter::criterion.init_threshold_values_histograms();
            }

            // Loop: all thresholds
            // --------------------

            // Sort f_X and f_s by f_X, leaving missing values at the beginning
            // samples s are now properly ordered for the partitioning
            sort2VectorsByFirstVector(f_X, s, pNA, n_samples);

            // Find threshold with maximum impurity improvement

            // Initialize position of last and next potential split to number of missing values
            SamplesIdx_t   p=pNA, pn=pNA;
            // Loop: samples
            double         max_improvement = 0.0;
            Features_t     max_threshold = std::numeric_limits<double>::quiet_NaN(); // no threshold
            SamplesIdx_t   max_pos = pNA;
            while (pn < n_samples) {
                // If remaining f_X values are constant then stop
                if (f_X[pn] + PRECISION_EQUAL >= f_X[n_samples-1]) break;
                // Skip constant Xf values
                while (pn + 1 < n_samples &&
                       f_X[pn] + PRECISION_EQUAL >= f_X[pn + 1]) ++pn;
                // Set pn to next position
                ++pn;

                // if (pn < n_samples): ... p = pn
                // Not required, because "if (f_X[pn] + PRECISION >= f_X[n_samples-1]) break" above
                // ensures that pn += 1 always points to valid data (pn < n_samples)

                // Update class histograms for using a threshold on values
                // from current position p to the new position np (correspond to thresholds)
                BestSplitter::criterion.update_threshold_histograms(y, s, pn);

                // Calculate impurity of samples with values that are smaller and greater than a threshold
                BestSplitter::criterion.calculate_threshold_impurity();

                // Calculate impurity improvement from the current node to its children
                // assuming a split of the samples with values smaller and greater than a threshold
                double  improvement_threshold = 0.0;
                if (pNA == 0) { // node has samples with values only
                    improvement_threshold =
                        BestSplitter::criterion.calculate_threshold_impurity_improvement();
                } if (pNA > 0) { // node has samples with values and missing values
                    improvement_threshold =
                        BestSplitter::criterion.calculate_threshold_values_impurity_improvement();
                }

                // Identify maximum impurity improvement
                if (improvement_threshold > max_improvement) {
                    max_improvement = improvement_threshold;
                    max_threshold = (f_X[p] + f_X[pn]) / 2.0;
                    max_pos = BestSplitter::start + pn;
                }

                // If impurity of right child is 0.0 (pure) then stop
                if (BestSplitter::criterion.get_node_impurity_threshold_right() < PRECISION_EQUAL) break;

                p = pn;
            }

            if (pNA == 0) { // node has samples with values only

                improvement = max_improvement;
                NA = -1; // no missing values
                threshold = max_threshold;
                pos = max_pos;

            } if (pNA > 0) { // node has samples with values and missing values

                // Add missing values to split (based on threshold)
                // ------------------------------------------------

                // Calculate impurity of samples with values that are smaller and greater than a threshold
                // combined with the samples with missing values
                BestSplitter::criterion.calculate_threshold_NA_impurity();
                // Calculate impurity improvement
                double  improvement_threshold_NA_left =
                        BestSplitter::criterion.calculate_threshold_NA_left_impurity_improvement();
                double  improvement_threshold_NA_right =
                        BestSplitter::criterion.calculate_threshold_NA_right_impurity_improvement();

                if (improvement_threshold_NA_left >= improvement_threshold_NA_right) {
                    // Add missing values to left child
                    if(improvement < improvement_threshold_NA_left) {

                        improvement = improvement_threshold_NA_left;
                        NA = 0; // missing values are passed on to the left child
                        threshold = max_threshold;
                        pos = max_pos;
                    }
                } else { // Add missing values to right child
                    if(improvement < improvement_threshold_NA_right) {

                        improvement = improvement_threshold_NA_right;
                        NA = 1; // missing values are passed on to the right child
                        threshold = max_threshold;

                        // move samples with missing values to the end of the sample vector
                        vector<SamplesIdx_t> s_NA(&s[0], &s[pNA]); // temp for NAs
                        copy(&s[pNA], &s[n_samples], &s[0]);
                        copy(&s_NA[0], &s_NA[pNA], &s[n_samples - pNA]);
                        pos = max_pos - pNA;
                    }
                }
            }
        }

    }

    // Find a split and partition samples for a given feature
    // using the Extreme Random Tree formulation for the threshold.

    void  BestSplitter::split_feature_extreme_random(Features_t*            X,
                                                     Classes_t*             y,
                                                     vector<SamplesIdx_t>&  s,
                                                     FeaturesIdx_t          feature,
                                                     int&                   NA,
                                                     Features_t&            threshold,
                                                     SamplesIdx_t&          pos,
                                                     double&                improvement) {

        NA = -1; // no missing values
        threshold = std::numeric_limits<double>::quiet_NaN(); // no threshold
        pos = 0;
        improvement = 0.0; // leaf node

        // y is not constant (impurity > 0)
        // has been checked by impurity stop criteria in build()
        // moving on we can assume at least 2 samples

        // Copy f_X=X[samples[start:end],f]
        // training data X for the current node.

        SamplesIdx_t  n_samples = BestSplitter::end - BestSplitter::start;
        vector<Features_t>  f_X(n_samples);
        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            f_X[i] = X[s[i]*BestSplitter::n_features + feature];
        }

        // Detect samples with missing values and move them to the beginning of the samples vector
        SamplesIdx_t  pNA = 0;
        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            if (isnan(f_X[i])) {
                swap(f_X[i],f_X[pNA]);
                swap(s[i],s[pNA]);
                pNA++;
            }
        }

        if (pNA == n_samples) return; // Can not split feature when all values are NA
        // moving on, in case that there are missing values, we can assume that there is at least 1 sample with a value
        // and in case that there are no missing values, we can assume that there are at least 2 samples with a value

        // f_X is not constant
        Features_t  f_min, f_max;
        f_min = f_max = f_X[pNA];
        for (SamplesIdx_t i=(pNA+1); i<n_samples; ++i) {
            if (f_X[i] < f_min) f_min = f_X[i]; else if (f_X[i] > f_max) f_max = f_X[i];
        }

        // Split just based on missing values
        // ----------------------------------

        if ((pNA > 0) && // missing values
            ((f_min + PRECISION_EQUAL > f_max) || // f_X is constant or
             (BestSplitter::random_state.uniform_int(0, n_samples) < static_cast<long>((pNA-1))))) {
            // random number proportional to the number of NA values determines if a split is done

            // Calculate class histograms for the samples with missing values and the samples with values
            BestSplitter::criterion.calculate_NA_histogram(y, s, pNA);
            // Calculate impurity of samples with missing values and samples with values
            BestSplitter::criterion.calculate_NA_impurity();
            // Calculate impurity improvement from the current node to its children
            // assuming a split between missing values and values
            improvement = BestSplitter::criterion.calculate_NA_impurity_improvement();
            NA = 0; // pass all samples with missing values to the left child
            // pass all samples with values to the right child
            threshold = std::numeric_limits<double>::quiet_NaN(); // no threshold
            pos = BestSplitter::start + pNA;

        // Split based on threshold
        // ------------------------

        } else {

            if (f_min + PRECISION_EQUAL < f_max) { // f_X is not constant

                // random threshold
                // ----------------
                // using the Extreme Random Tree formulation

                // Draw random threshold
                threshold = BestSplitter::random_state.uniform_real(f_min + PRECISION_EQUAL, f_max);
                // excludes f_min, f_max, with uniform_real(low, high), low is inclusive and high is exclusive

                // Partition s such that f_X[s[np-1]] <= threshold < f_X[s[np]]
                SamplesIdx_t  p = pNA, pn = n_samples;
                while (p < pn) {
                    if (f_X[p] <= threshold) ++p;
                    else {
                        --pn;
                        swap(f_X[p],f_X[pn]);
                        swap(s[p],s[pn]);
                    }
                }

                if (pNA == 0) { // node has samples with values only

                    // Initialize class histograms for using a threshold
                    BestSplitter::criterion.init_threshold_histograms();
                    // Update class histograms for the children
                    // of the current node from position 0 to the position np
                    BestSplitter::criterion.update_threshold_histograms(y, s, pn);
                    // Calculate impurity of children
                    BestSplitter::criterion.calculate_threshold_impurity();
                    // Calculate impurity improvement
                    double  improvement_threshold = BestSplitter::criterion.calculate_threshold_impurity_improvement();

                    improvement = improvement_threshold;
                    NA = -1; // no missing values
                    pos = BestSplitter::start + pn;

                } else if (pNA > 0) { // node has samples with values and missing values

                    // Add missing values to split (based on threshold)
                    // ------------------------------------------------
                    // impurity improvement determines how the split is done

                    // Calculate class histograms for the samples with missing values and the samples with values
                    BestSplitter::criterion.calculate_NA_histogram(y, s, pNA);
                    // Initialize class histograms for using a threshold on samples with values
                    BestSplitter::criterion.init_threshold_values_histograms();
                    // Update class histograms for the children
                    // of the current node from position 0 to the position np
                    BestSplitter::criterion.update_threshold_histograms(y, s, pn);
                    // Calculate impurity of children
                    BestSplitter::criterion.calculate_threshold_impurity();
                    // Calculate impurity improvements
                    double  improvement_threshold_NA_left =
                            BestSplitter::criterion.calculate_threshold_NA_left_impurity_improvement();
                    double  improvement_threshold_NA_right =
                            BestSplitter::criterion.calculate_threshold_NA_right_impurity_improvement();

                    if (improvement_threshold_NA_left >= improvement_threshold_NA_right) {
                        // Add missing values to left child
                        improvement = improvement_threshold_NA_left;
                        NA = 0; // missing values are passed on to the left child
                        pos = BestSplitter::start + pn;
                    } else {
                        // Add missing values to right child
                        improvement = improvement_threshold_NA_right;
                        NA = 1; // missing values are passed on to the right child

                        // move samples with missing values to the end of the sample vector
                        vector<SamplesIdx_t> s_NA(&s[0], &s[pNA]); // temp for NAs
                        copy(&s[pNA], &s[n_samples], &s[0]);
                        copy(&s_NA[0], &s_NA[pNA], &s[n_samples - pNA]);
                        pos = BestSplitter::start + pn - pNA;
                    }
                }
            }
        }

    }

    // Find the best split and partition (actually sorted) samples.

    void  BestSplitter::split_node(Features_t*     X,
                                   Classes_t*      y,
                                   FeaturesIdx_t&  feature,
                                   int&            NA,
                                   Features_t&     threshold,
                                   SamplesIdx_t&   pos,
                                   double&         improvement) {

        feature = 0;
        NA = -1; // NA
        threshold = 0.0;
        pos = 0;

        // Copy s = samples[start:end]
        // LUT to the training data X, y for the current node.
        vector<SamplesIdx_t> s(BestSplitter::end - BestSplitter::start);
        copy(&BestSplitter::samples[BestSplitter::start], &BestSplitter::samples[BestSplitter::end], s.begin());

        // Loop: k random features (k defined by max_features)
        // ---------------------------------------------------

        // When max_features == n_features this is the same as
        // Loop: all features "for (f=0; f<n_features; ++f)",
        // but in a random order, which is preferable.

        // Features are sampled without replacement using
        // the modern version of the Fischer-Yates shuffle algorithm
        // in an iterative way.

        vector<FeaturesIdx_t>  features(BestSplitter::n_features);
        iota(features.begin(), features.end(), 0); // identity mapping

        improvement = 0.0;
        auto i = BestSplitter::n_features; // i=n instead of n-1 because of uniform_int(0,n)
        while ((i > (BestSplitter::n_features - BestSplitter::max_features)) || // includes case 0
                    (improvement < PRECISION_EQUAL && i > 0)) {
            // continue until at least one none constant feature was selected
            // or ultimately no more features are left

            unsigned long  j = 0;
            if (i>1) j = static_cast<unsigned long>(BestSplitter::random_state.uniform_int(0, i)); // covers case 0
            // uniform_int(low, high), low is inclusive and high is exclusive
            --i; // adjust indexing by i
            swap(features[i],features[static_cast<FeaturesIdx_t>(j)]);
            FeaturesIdx_t  f = features[i];

            // Split feature
            int           f_NA = 0;
            Features_t    f_threshold = 0;
            SamplesIdx_t  f_pos = 0;
            double        f_improvement = improvement;

            if (BestSplitter::max_thresholds == 0) {
                BestSplitter::split_feature(X, y, s, f, f_NA, f_threshold, f_pos, f_improvement);
            } else if (BestSplitter::max_thresholds == 1) { // Extreme Random Tree
                BestSplitter::split_feature_extreme_random(X, y, s, f, f_NA, f_threshold, f_pos, f_improvement);
            }

            // keeping sorted samples s from feature run
            if (f_improvement > improvement) { // keep results for maximum improvement
                improvement = f_improvement;
                feature = f;
                NA = f_NA;
                threshold = f_threshold;
                pos = f_pos; // position f_pos corresponds to s samples
                // Replace samples[start:end] with properly ordered samples s
                copy(s.begin(), s.end(), &BestSplitter::samples[BestSplitter::start]);
            }
        }

    }

// =============================================================================
// Tree Builder
// =============================================================================

    // Create and initialize a new depth first tree builder.

    DepthFirstTreeBuilder::DepthFirstTreeBuilder(ClassesIdx_t     n_classes,
                                                 FeaturesIdx_t    n_features,
                                                 SamplesIdx_t     n_samples,
                                                 ClassWeights_t*  class_weight,
                                                 TreeDepthIdx_t   max_depth,
                                                 FeaturesIdx_t    max_features,
                                                 unsigned long    max_thresholds,
                                                 string           missing_values,
                                                 RandomState const& random_state)
    : max_depth(max_depth), missing_values(std::move(missing_values)),
      // Create and initialize a new best splitter (and gini criterion)
      splitter(n_classes,
               n_features,
               n_samples,
               class_weight,
               max_features,
               max_thresholds,
               random_state) { }

    // Build a binary decision tree from the training data.

    void  DepthFirstTreeBuilder::build(Tree&         tree,
                                       Features_t*   X,
                                       Classes_t*    y,
                                       SamplesIdx_t  n_samples) {

        // Create an empty node information stack
        struct NodeInfo {

            SamplesIdx_t    start;
            SamplesIdx_t    end;
            TreeDepthIdx_t  depth;
            NodesIdx_t      parent_id;
            bool            is_left;

            NodeInfo(SamplesIdx_t    start,
                     SamplesIdx_t    end,
                     TreeDepthIdx_t  depth,
                     NodesIdx_t      parent_id,
                     bool            is_left)
            : start(start),
              end(end),
              depth(depth),
              parent_id(parent_id),
              is_left(is_left) { };
        };
        stack<NodeInfo>  node_info_stack;

        tree.nodes.reserve((1u << (DepthFirstTreeBuilder::max_depth+1))-1);

        // Push root node information onto the stack
        node_info_stack.emplace(NodeInfo(0, n_samples, 0, 0, false));
        // Loop: nodes
        while (!node_info_stack.empty()) {
            // Pop current node information from the stack
            NodeInfo  cn = node_info_stack.top();
            node_info_stack.pop();

            // Calculate number of samples per class histogram
            // and impurity for the current node
            splitter.init_node(y, cn.start, cn.end);
            vector<Histogram_t>  histogram = DepthFirstTreeBuilder::splitter.criterion.get_node_weighted_histogram();
            double  impurity = DepthFirstTreeBuilder::splitter.criterion.get_node_impurity();
            SamplesIdx_t  pos = 0;

            // If a stop criterion is met node becomes a leaf node
            bool  is_leaf = (cn.depth >= DepthFirstTreeBuilder::max_depth) ||
                            (impurity <= PRECISION_EQUAL);

            // Split node (if node is not a leaf node)
            FeaturesIdx_t  feature = 0;
            int            NA = -1; // NA
            Features_t     threshold = std::numeric_limits<double>::quiet_NaN(); // no threshold
            double         improvement = 0.0;
            if (!is_leaf) {
                // Find the split on samples[start:end] that maximizes impurity improvement and
                // partition samples[start:end] into samples[start:pos] and samples[pos:end]
                // according to the split
                DepthFirstTreeBuilder::splitter.split_node(X, y, feature, NA, threshold, pos, improvement);
                // If no impurity improvement (no split found) then node is a leaf node
                if (improvement <= PRECISION_EQUAL) is_leaf = true;
            }

            // Add node to the decision tree
            NodesIdx_t node_id = tree.add_node(cn.depth, cn.parent_id, cn.is_left,
                                               feature, NA, threshold,
                                               histogram, impurity, improvement);

            // Split node (if not a leaf node)
            if (!is_leaf) {
                // Push right child node information onto the stack
                node_info_stack.emplace(NodeInfo(pos, cn.end, cn.depth+1, node_id, false));
                // Push left child node information onto the stack
                // LIFO: left depth first order
                node_info_stack.emplace(NodeInfo(cn.start, pos, cn.depth+1, node_id, true));
            }
        }

        tree.nodes.shrink_to_fit();

    }

// =============================================================================
// Decision Tree Classifier
// =============================================================================

    // Create and initialize a new decision tree classifier.

    DecisionTreeClassifier::DecisionTreeClassifier(vector<std::string>  classes,
                                                   ClassesIdx_t         n_classes,
                                                   vector<std::string>  features,
                                                   FeaturesIdx_t        n_features,
                                                   string const&        class_balance,
                                                   TreeDepthIdx_t       max_depth,
                                                   FeaturesIdx_t        max_features,
                                                   unsigned long        max_thresholds,
                                                   string const&        missing_values,
                                                   long                 random_state_seed)
    : tree_(static_cast<ClassesIdx_t>(n_classes),
            static_cast<FeaturesIdx_t>(n_features)) {

        DecisionTreeClassifier::classes = std::move(classes);
        DecisionTreeClassifier::n_classes = n_classes;
        DecisionTreeClassifier::features = std::move(features);
        DecisionTreeClassifier::n_features = n_features;

        // Check hyperparameters

        // class balance
        if (class_balance == "balanced" || class_balance == "None")
            DecisionTreeClassifier::class_balance = class_balance;
        else
            DecisionTreeClassifier::class_balance = "balanced"; // default

        // max depth
        const TreeDepthIdx_t  MAX_DEPTH = 2147483647; // max long: (2^31)-1
        if ((0 < max_depth) && (max_depth <= MAX_DEPTH))
            DecisionTreeClassifier::max_depth = max_depth;
        else
            DecisionTreeClassifier::max_depth = MAX_DEPTH;

        // max features
        if ((0 < max_features) && (max_features <= DecisionTreeClassifier::n_features))
            DecisionTreeClassifier::max_features = max_features;
        else
            DecisionTreeClassifier::max_features = DecisionTreeClassifier::n_features;

        // max thresholds
        if ((max_thresholds == 0) || (max_thresholds == 1))
            DecisionTreeClassifier::max_thresholds = max_thresholds;
        else
            DecisionTreeClassifier::max_thresholds = 0; // default

        // missing values
        if (missing_values == "MCAR" || missing_values == "NMAR" || missing_values == "None")
            DecisionTreeClassifier::missing_values = missing_values;
        else
            DecisionTreeClassifier::missing_values = "None"; // default

        // Random Number Generator

        if (random_state_seed == -1)
            DecisionTreeClassifier::random_state = RandomState();
        else
            DecisionTreeClassifier::random_state = RandomState(static_cast<unsigned long>(random_state_seed));

    }

    // Build a decision tree classifier from the training data.

    void DecisionTreeClassifier::fit(Features_t*   X,
                                     Classes_t*    y,
                                     SamplesIdx_t  n_samples) {

        // Calculate class weights
        // so that n_samples == sum of all weighted samples

        std::vector<double>  class_weight(n_samples, 1.0);
        if (DecisionTreeClassifier::class_balance == "balanced") {
            std::vector<long>  bincount(n_classes, 0);
            for (unsigned long i=0; i<n_samples; ++i) {
                bincount[y[i]]++;
            }
            for (unsigned long i=0; i<n_classes; ++i) {
                class_weight[i] = (static_cast<double>(n_samples) / bincount[i]) / DecisionTreeClassifier::n_classes;
            }
        }

        //  Build decision tree

        DepthFirstTreeBuilder builder(DecisionTreeClassifier::n_classes,
                                      DecisionTreeClassifier::n_features,
                                      n_samples,
                                      &class_weight[0],
                                      DecisionTreeClassifier::max_depth,
                                      DecisionTreeClassifier::max_features,
                                      DecisionTreeClassifier::max_thresholds,
                                      DecisionTreeClassifier::missing_values,
                                      DecisionTreeClassifier::random_state);

        builder.build(DecisionTreeClassifier::tree_, X, y, n_samples);

    };

    // Predict classes probabilities for the test data.

    void  DecisionTreeClassifier::predict_proba(Features_t*   X,
                                                SamplesIdx_t  n_samples,
                                                double*       y_prob) {

        DecisionTreeClassifier::tree_.predict(X, n_samples, y_prob);

    }

    // Predict classes for the test data.

    void  DecisionTreeClassifier::predict(Features_t*   X,
                                          SamplesIdx_t  n_samples,
                                          Classes_t*    y) {

        std::vector<double>  y_prob(n_samples*DecisionTreeClassifier::n_classes, 0.0);
        DecisionTreeClassifier::predict_proba(X, n_samples, &y_prob[0]);

        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            y[i] = maxIndex(&y_prob[i*DecisionTreeClassifier::n_classes], DecisionTreeClassifier::n_classes);
        }

    }

    // Calculate score for the test data.

    double DecisionTreeClassifier::score(Features_t*   X,
                                         Classes_t*    y,
                                         SamplesIdx_t  n_samples) {

        std::vector<long>    y_predict(n_samples, 0);
        DecisionTreeClassifier::predict(X, n_samples, &y_predict[0]);

        unsigned long n_true = 0;
        for (unsigned long i = 0; i < n_samples; ++i) {
            if (y_predict[i] == y[i]) n_true++;
        }
        return static_cast<double>(n_true) / n_samples;

    }

    // Calculate feature importances from the decision tree.

    void  DecisionTreeClassifier::calculate_feature_importances(double*  importances) {

        DecisionTreeClassifier::tree_.calculate_feature_importances(importances);

    }

    // Create a rgb color look up table (LUT) for all classes.

    vector<vector<int>>  create_rgb_LUT(ClassesIdx_t  n_classes) {

        // Define rgb colors for the different classes
        // with (somewhat) max differences in hue between nearby classes

        // Number of iterations over the grouping of 2x 3 colors
        n_classes = max(n_classes, static_cast<ClassesIdx_t>(1)); // input check > 0
        auto n = static_cast<unsigned long>(floor((n_classes - 1) / 6) + 1); // > 0

        // Create a list of offsets for the grouping of 2x 3 colors
        // that (somewhat) max differences in hue between nearby classes
        vector<int>  offset_list;
        offset_list.emplace_back(0); // creates pure R G B - Y C M colors
        int  d = 128;
        int  n_offset_levels = 1; // log(0) not defined
        if (n > 1) n_offset_levels = static_cast<int>(log2(n-1)+1);
        n_offset_levels = min(n_offset_levels, 4);  // limit number of colors to 96
        for (int i=0; i<n_offset_levels; ++i) {
            // Create in between R G B Y C M colors
            // in a divide by 2 pattern per level
            // i=0: + 128,
            // i=1: +  64, 192,
            // i=2: +  32, 160, 96, 224,
            // i=3: +  16, 144, 80, 208, 48, 176, 112, 240
            // abs max i=7 with + 1 ...
            vector<int>  offset_list_tmp;
            for (auto offset : offset_list) {
                offset_list_tmp.emplace_back(offset + d);
            }
            offset_list.insert(offset_list.end(),
                               make_move_iterator(offset_list_tmp.begin()),
                               make_move_iterator(offset_list_tmp.end()));
            d = d / 2;
        }

        // If there are more classes than colors
        // then the offset_list is duplicated,
        // which assigns the same colors to different classes
        // but at least to the most distance classes

        unsigned long  l = offset_list.size();
        if (n > l) {
            vector<int>  offset_list_tmp(offset_list);
            int n_copies = static_cast<int>(1 + ceil((n - l) / l));
            for (int i=0; i<n_copies; ++i) {
                offset_list.insert(offset_list.end(),
                                   offset_list_tmp.begin(),
                                   offset_list_tmp.end());
            }
        }

        vector<vector<int>>  rgb_LUT(n*6, vector<int>(3, 0));
        for (unsigned long i=0; i<n; ++i) {
            // Calculate grouping of 2x 3 rgb colors R G B - Y C M
            // that (somewhat) max differences in hue between nearby classes
            // and makes it easy to define other in between colors
            // using a simple linear offset
            // Based on HSI to RGB calculation with I = 1 and S = 1
            int  offset = offset_list[i];
            rgb_LUT[i*6] =   { 255, offset, 0 };  // 0 <= h < 60 RED ...
            rgb_LUT[i*6+1] = { 0, 255, offset };  // 120 <= h < 180 GREEN ...
            rgb_LUT[i*6+2] = { offset, 0, 255 };  // 240 <= h < 300 BLUE ...
            rgb_LUT[i*6+3] = { 255 - offset, 255, 0 };  // 60 <= h < 120 YELLOW ...
            rgb_LUT[i*6+4] = { 0, 255 - offset, 255 };  // 180 <= h < 240 CYAN ...
            rgb_LUT[i*6+5] = { 255, 0, 255 - offset };  // 300 <= h < 360 MAGENTA ...
        }

        return rgb_LUT;
    }

    // Process tree recursively node by node and provide GraphViz dot format for node.

    void  process_tree_recursively_graphviz(const Tree&                 tree,
                                            NodesIdx_t                  node_id,
                                            const vector<vector<int>>&  rgb_LUT,
                                            const vector<string>&       classes,
                                            const vector<string>&       features,
                                            bool                        rotate,
                                            ofstream&                   fout) {

        // Current node
        NodesIdx_t           left_child = tree.nodes[node_id].left_child;
        NodesIdx_t           right_child = tree.nodes[node_id].right_child;
        FeaturesIdx_t        feature = tree.nodes[node_id].feature;
        int                  NA = tree.nodes[node_id].NA;
        Features_t           threshold = tree.nodes[node_id].threshold;
        vector<Histogram_t>  histogram = tree.nodes[node_id].histogram;
        double               impurity = tree.nodes[node_id].impurity;

        // Prediction
        ClassesIdx_t  c = maxIndex(&histogram[0],histogram.size());
        stringstream  p_c;
        Histogram_t   n = accumulate(histogram.begin(), histogram.end(), 0.0);
        p_c << "[" << setprecision(2) << setw(2) << (histogram[0] / n);
        for (ClassesIdx_t cc=1; cc<histogram.size(); ++cc) {
            p_c << "," << setw(2) << (histogram[cc] / n);
        }
        p_c << "]";

        // Node color and intensity based on classification and impurity
        int  r = rgb_LUT[c][0];
        int  g = rgb_LUT[c][1];
        int  b = rgb_LUT[c][2];
        double  max_impurity = 1.0 - (1.0 / tree.n_classes);
        int     alpha = static_cast<int>(255 * (max_impurity - impurity) / max_impurity);
        stringstream  color; // #RRGGBBAA hex format
        color << '#' << hex << setw(2) << setfill('0') << r << setw(2) << g << setw(2) << b << setw(2) << alpha << dec;

        // Leaf node
        if (left_child == 0) {
            // leaf nodes do no have any children
            // so we only need to test for one of the children

            // Node
            stringstream  node;
            node << node_id
                 << " [label=\""
                 << p_c.str() << "\\n" << classes[c] << "\""
                 << ", fillcolor=\"" << color.str() << "\"] ;"
                 << endl;
            fout << node.str();

        } else { // Split node

            // Order children nodes by predicted classes (and their probabilities)
            // Switch left_child with right_child and
            // modify test feature <= threshold (default) vs feature > threshold accordingly

            bool           order = true;
            unsigned long  test_type = 0; // 0: feature <= threshold (default)
                                          // 1: feature >  threshold, when left and right children are switched

            bool  change = false;
            if (order) {
                // Left Child Prediction
                vector<Histogram_t> lc_histogram = tree.nodes[left_child].histogram;
                ClassesIdx_t        lc_c = maxIndex(&lc_histogram[0], lc_histogram.size());
                Histogram_t         lc_n = accumulate(lc_histogram.begin(), lc_histogram.end(), 0.0);
                double              lc_p_c = lc_histogram[lc_c] / lc_n;
                // Right Child Prediction
                vector<Histogram_t> rc_histogram = tree.nodes[right_child].histogram;
                ClassesIdx_t        rc_c = maxIndex(&rc_histogram[0], rc_histogram.size());
                Histogram_t         rc_n = accumulate(rc_histogram.begin(), rc_histogram.end(), 0.0);
                double              rc_p_c = rc_histogram[rc_c] / rc_n;
                // Determine if left_child and right_child should be switched based on predictions
                if (lc_c > rc_c) { // assign left child to lower class index
                    change = true;
                } else if (lc_c == rc_c) {     // if class indices are the same for left and right children
                    if (lc_c == 0) {           // for the first class index = 0
                        if (lc_p_c < rc_p_c) { // assign left child to higher class probability
                            change = true;
                        }
                    } else {                   // for all other class indices > 0
                        if (lc_p_c > rc_p_c) { // assign left child to lower class probability
                            change = true;
                        }
                    }
                }
                if (change) {
                    test_type = 1;
                    NodesIdx_t idx = left_child;
                    left_child = right_child;
                    right_child = idx;
                }
            }

            // Edge width based on (weighted) number of samples used for training
            vector<Histogram_t>  root_histogram = tree.nodes[0].histogram;
            vector<Histogram_t>  left_child_histogram = tree.nodes[left_child].histogram;
            vector<Histogram_t>  right_child_histogram = tree.nodes[right_child].histogram;
            // total number of samples used for training
            Histogram_t   n_root = accumulate(root_histogram.begin(), root_histogram.end(), 0.0);
            Histogram_t   n_left_child = accumulate(left_child_histogram.begin(), left_child_histogram.end(), 0.0)
                                         / n_root; // normalized
            Histogram_t   n_right_child = accumulate(right_child_histogram.begin(), right_child_histogram.end(), 0.0)
                                          / n_root; // normalized

            const double  MAX_WIDTH = 10;

            // Node
            stringstream  node;
            node << node_id << setprecision(4) << setw(4) << " [label=\"";
            // - feature
            node << features[feature];
            // - threshold
            if (!(isnan(threshold))) {
                if (test_type == 0) {
                    node << " <= " << threshold;
                } else { // test_type == 1
                    node << " > " << threshold;
                }
            }
            // - NA
            if (NA != -1) {
                if (!change) {
                    if (NA == 0) node << " NA"; // left
                    if (NA == 1) node << " not NA"; // right
                } else {
                    if (NA == 0) node << " not NA"; // right
                    if (NA == 1) node << " NA"; // left
                }
            }
            // - histogram
            if (node_id == 0) { // Root node with legend
                node << "\\np(class) = " << p_c.str() << "\\nclass, n = " << static_cast<unsigned long>(n) << "\"";
            } else {
                node << "\\n" << p_c.str() << "\\n" << "\"";
            }
            node << ", fillcolor=\"" << color.str() << "\"] ;" << endl;
            fout << node.str();

            // Edges
            stringstream  edges;
            // - left child
            edges << node_id << " -> " << left_child << " [";
            edges << "penwidth=" << max(MAX_WIDTH/100.0, MAX_WIDTH*n_left_child);
            if (node_id == 0) { // Root node with legend
                if (rotate) edges << " headlabel=\"true\", labeldistance=2.5, labelangle=-45";
                else        edges << " headlabel=\"true\", labeldistance=2.5, labelangle=45";
            }
            edges << "] ;" << endl;
            // - right child
            edges << node_id << " -> " << right_child << " [";
            edges << "penwidth=" << max(MAX_WIDTH/100.0, MAX_WIDTH*n_right_child);
            // layout problems with legend true and false depending on tree size
            // no need to define false when true is defined
            edges << "] ;" << endl;
            fout << edges.str();

            // Process the tree recursively
            process_tree_recursively_graphviz(tree, left_child,  rgb_LUT, classes, features, rotate, fout);
            process_tree_recursively_graphviz(tree, right_child, rgb_LUT, classes, features, rotate, fout);
        }

    }

    /// Export of a decision tree in GraphViz dot format.

    void  DecisionTreeClassifier::export_graphviz(std::string const& filename,
                                                  bool               rotate) {

        string fn = filename + ".gv";

        ofstream  fout(fn);
        if (fout.is_open()) {

            fout << "digraph Tree {" << endl;
            fout << "node [shape=box, style=\"rounded, filled\", color=\"black\", fontname=helvetica, fontsize=14] ;" << endl;
            fout << "edge [fontname=helvetica, fontsize=12] ;" << endl;

            // Rotate (default: top-down)
            if (rotate) fout << "rankdir=LR ;" << endl; // left-right orientation

            // Define rgb colors for the different classes
            vector<vector<int>>  rgb_LUT = create_rgb_LUT(n_classes);

            // Process the tree recursively
            process_tree_recursively_graphviz(tree_, 0, rgb_LUT, classes, features, rotate, fout);  // root node = 0

            fout << "}" << endl;

            fout.close();
        }

    }

    // Process tree recursively node by node and provide GraphViz dot format for node.

    void  process_tree_recursively_text(const Tree&                 tree,
                                        NodesIdx_t                  node_id,
                                        ostringstream&              sout) {

        // Current node
        NodesIdx_t           left_child = tree.nodes[node_id].left_child;
        NodesIdx_t           right_child = tree.nodes[node_id].right_child;
        FeaturesIdx_t        feature = tree.nodes[node_id].feature;
        int                  NA = tree.nodes[node_id].NA;
        Features_t           threshold = tree.nodes[node_id].threshold;
        vector<Histogram_t>  histogram = tree.nodes[node_id].histogram;

        // Histogram formatting as string
        stringstream         histogram_string;
        histogram_string << "[" << histogram[0];
        for (ClassesIdx_t c=1; c<histogram.size(); ++c) {
            histogram_string << ", " << histogram[c];
        }
        histogram_string << "]";

        // Leaf node
        if (left_child == 0) {
            // leaf nodes do no have any children
            // so we only need to test for one of the children

            sout << node_id;
            sout << " " << histogram_string.str() << "; ";

        } else { // Split node

            sout << node_id;
            sout << " X[" << feature << "]";
            if (!(isnan(threshold))) sout << "<=" << threshold;
            if (NA == 0) sout << " NA"; // left
            if (NA == 1) sout << " not NA"; // right
            sout << " " << histogram_string.str() << "; ";

            sout << node_id << "->" << left_child << "; ";
            sout << node_id << "->" << right_child << "; ";

            // Process the tree recursively
            process_tree_recursively_text(tree, left_child,  sout);
            process_tree_recursively_text(tree, right_child, sout);
        }

    }

    // Export of a decision tree in a simple text format.

    string  DecisionTreeClassifier::export_text() {

        ostringstream sout;

        // Process the tree recursively
        process_tree_recursively_text(tree_, 0, sout);  // root node = 0

        return sout.str();
    }

    // Serialize

    void  DecisionTreeClassifier::serialize(ofstream& fout) {

        // Classes
        fout.write((const char*)(&n_classes), sizeof(n_classes));
        for (unsigned long c=0; c<n_classes; ++c) {
            unsigned long  size = classes[c].size();
            fout.write((const char*)&size, sizeof(size));
            fout.write((const char*)&classes[c][0], size);
        }

        // Features
        fout.write((const char*)(&n_features), sizeof(n_features));
        for (unsigned long f=0; f<n_features; ++f) {
            unsigned long  size = features[f].size();
            fout.write((const char*)&size, sizeof(size));
            fout.write((const char*)&features[f][0], size);
        }

        // Hyperparameters
        unsigned long  size = class_balance.size();
        fout.write((const char*)&size, sizeof(size));
        fout.write((const char*)&class_balance[0], size);
        fout.write((const char*)&max_depth, sizeof(max_depth));
        fout.write((const char*)&max_features, sizeof(max_features));
        fout.write((const char*)&max_thresholds, sizeof(max_thresholds));
        size = missing_values.size();
        fout.write((const char*)&size, sizeof(size));
        fout.write((const char*)&missing_values[0], size);

        // Random Number Generator
        fout.write((const char*)&random_state, sizeof(random_state));

        // Model
        // Serialize Decision Tree
        tree_.serialize(fout);

    }

    // Export of a decision tree classifier in binary serialized format.

    void  DecisionTreeClassifier::export_serialize(std::string const& filename) {

        string fn = filename + ".dtc";

        ofstream  fout(fn, ios_base::binary);
        if (fout.is_open()) {

            const int  version = 1; // file version number
            fout.write((const char*)&version, sizeof(version));

            // Serialize Decision Tree Classifier
            serialize(fout);

            fout.close();

        } else {
            throw runtime_error("Unable to open file.");
        }
    }

    // Deserialize

    DecisionTreeClassifier  DecisionTreeClassifier::deserialize(ifstream& fin) {

        // Classes
        ClassesIdx_t    n_classes;
        vector<string>  classes;
        fin.read((char*)(&n_classes), sizeof(n_classes));
        for (unsigned long c=0; c<n_classes; ++c) {
            string str;
            unsigned long  size;
            fin.read((char*)(&size), sizeof(size));
            str.resize(size);
            fin.read((char*)(&str[0]), size);
            classes.emplace_back(str);
        }

        // Features
        FeaturesIdx_t   n_features;
        vector<string>  features;
        fin.read((char*)(&n_features), sizeof(n_features));
        for (unsigned long f=0; f<n_features; ++f) {
            string str;
            unsigned long  size;
            fin.read((char*)(&size), sizeof(size));
            str.resize(size);
            fin.read((char*)(&str[0]), size);
            features.emplace_back(str);
        }

        // Hyperparameters
        string          class_balance;
        TreeDepthIdx_t  max_depth;
        FeaturesIdx_t   max_features;
        unsigned long   max_thresholds;
        string          missing_values;

        unsigned long  size;
        fin.read((char*)(&size), sizeof(size));
        class_balance.resize(size);
        fin.read((char*)(&class_balance[0]), size);
        fin.read((char*)(&max_depth), sizeof(max_depth));
        fin.read((char*)(&max_features), sizeof(max_features));
        fin.read((char*)(&max_thresholds), sizeof(max_thresholds));
        fin.read((char*)(&size), sizeof(size));
        fin.read((char*)(&missing_values[0]), size);

        // Random Number Generator
        long    random_state_seed = 0;

        DecisionTreeClassifier  dtc(classes, n_classes,
                                    features, n_features,
                                    class_balance, max_depth,
                                    max_features, max_thresholds,
                                    missing_values,
                                    random_state_seed);

        // Random Number Generator - overwrite random state
        fin.read((char*)(&dtc.random_state), sizeof(dtc.random_state));

        // Model
        // Deserialize Decision Tree
        dtc.tree_.deserialize(fin);

        return dtc;
    }

    // Import of a decision tree classifier in binary serialized format.

    DecisionTreeClassifier  DecisionTreeClassifier::import_deserialize(std::string const& filename) {

        string fn = filename + ".dtc";

        ifstream  fin(fn, ios_base::binary);
        if (fin.is_open()) {

            int  version;
            fin.read((char*)(&version), sizeof(version));

            if (version == 1) { // file version number

                // Deserialize Decision Tree Classifier
                DecisionTreeClassifier  dtc = DecisionTreeClassifier::deserialize(fin);

                fin.close();

                return dtc;

            } else {
                fin.close();
                throw runtime_error("Unsupported file version number.");
            }
        } else {
            throw runtime_error("Unable to open file.");
        }
    }

} // namespace koho


