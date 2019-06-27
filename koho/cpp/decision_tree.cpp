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
#include <set>

#include "utilities.h"
#include "decision_tree.h"

using namespace std;

namespace koho {

// =============================================================================
// Tree
// =============================================================================

    // Create a new node.

    Node::Node(NodesIdx_t                           left_child,
               NodesIdx_t                           right_child,
               FeaturesIdx_t                        feature,
               int                                  NA,
               Features_t                           threshold,
               const vector<vector<Histogram_t>>&   histogram,
               double                               impurity,
               double                               improvement)
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

        unsigned long  n_outputs = histogram.size();
        fout.write((const char *) (&n_outputs), sizeof(n_outputs));
        for (unsigned long o=0; o<n_outputs; ++o) {
            unsigned long  n_classes = histogram[o].size();
            fout.write((const char *) (&n_classes), sizeof(n_classes));
            for (unsigned long c=0; c<n_classes; ++c) {
                fout.write((const char *) (&histogram[o][c]), sizeof(histogram[o][c]));
            }
        }

        fout.write((const char *) (&impurity), sizeof(impurity));
        fout.write((const char *) (&improvement), sizeof(improvement));
    }

    // Deserialize

    Node  Node::deserialize(ifstream& fin) {

        NodesIdx_t                      left_child;
        NodesIdx_t                      right_child;
        FeaturesIdx_t                   feature;
        int                             NA;
        Features_t                      threshold;
        vector<vector<Histogram_t>>     histogram;
        double                          impurity;
        double                          improvement;

        fin.read((char*)(&left_child), sizeof(left_child));
        fin.read((char*)(&right_child), sizeof(right_child));
        fin.read((char*)(&feature), sizeof(feature));
        fin.read((char*)(&NA), sizeof(NA));
        fin.read((char*)(&threshold), sizeof(threshold));

        unsigned long  n_outputs;
        fin.read((char*)(&n_outputs), sizeof(n_outputs));
        for (OutputsIdx_t o=0; o<n_outputs; ++o) {
            vector<Histogram_t> o_histogram;
            unsigned long  n_classes;
            fin.read((char*)(&n_classes), sizeof(n_classes));
            for (unsigned long c=0; c<n_classes; ++c) {
                Histogram_t value;
                fin.read((char*)(&value), sizeof(value));
                o_histogram.emplace_back(value);
            }
            histogram.emplace_back(o_histogram);
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

    Tree::Tree(OutputsIdx_t           n_outputs,
               vector<ClassesIdx_t>   n_classes,
               FeaturesIdx_t          n_features)

    : n_outputs(n_outputs),
      n_classes(move(n_classes)),
      n_features(n_features),
      max_depth(0),
      node_count(0) {

        // for convenience

        Tree::n_classes_max = *max_element(begin(Tree::n_classes), end(Tree::n_classes));
    }


    // Create a new tree without nodes for Python binding.

    Tree::Tree(OutputsIdx_t           n_outputs,
               ClassesIdx_t*          n_classes_ptr,
               FeaturesIdx_t          n_features)

            : n_outputs(n_outputs),
              n_classes(n_outputs),
              n_features(n_features),
              max_depth(0),
              node_count(0) {

        // use pointer to array from Python binding
        // copy data to std vector, which is much nicer to handle for serialization and deserialization
        memcpy(&Tree::n_classes[0], n_classes_ptr, n_outputs*sizeof(ClassesIdx_t));

        // for convenience

        Tree::n_classes_max = *max_element(begin(Tree::n_classes), end(Tree::n_classes));
    }

    // Add a new node to the tree.

    NodesIdx_t  Tree::add_node(TreeDepthIdx_t                       depth,
                               NodesIdx_t                           parent_id,
                               bool                                 is_left,
                               FeaturesIdx_t                        feature,
                               int                                  NA,
                               Features_t                           threshold,
                               const vector<vector<Histogram_t>>&   histogram,
                               double                               impurity,
                               double                               improvement) {

        nodes.emplace_back(0, 0, // children IDs are set when the child nodes are added
                           feature, NA, threshold,
                           histogram, impurity, improvement);

        // Register new node as the child of its parent

        NodesIdx_t  node_id = node_count++;
        if (depth > 0) { // not root node
            if (is_left) { nodes[parent_id].left_child  = node_id;
            } else {       nodes[parent_id].right_child = node_id;
            }
        }

        if (depth > max_depth) max_depth = depth;

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
        // We use n_classes_max to create a nice 3D array to hold the predicted values x samples x classes
        // as the number of classes can be different for different outputs
        memset(y_prob, 0, n_samples * n_outputs * n_classes_max * sizeof(double));

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

                while (nodes[node_idx.idx].left_child > 0) { // follow path until leaf node
                // leaf nodes do no have any children
                // so we only need to test for one of the children

                    if (isnan(X[i*n_features + nodes[node_idx.idx].feature])) { // missing value

                        // Split criterion includes missing values
                        // Go to left or right child node depending on split (NA)
                        if (nodes[node_idx.idx].NA == 0) // left
                            node_idx.idx = nodes[node_idx.idx].left_child;
                        else if (nodes[node_idx.idx].NA == 1) // right
                            node_idx.idx = nodes[node_idx.idx].right_child;

                        else { // Split criterion does NOT includes missing values
                            IdxInfo node_idx2 = node_idx;
			                node_idx2.idx = nodes[node_idx.idx].right_child;  // right child
                            // Histograms for all outputs have the same number of elements n
                            // there only the histogram for the first output is used
			                double n_right = accumulate(nodes[node_idx2.idx].histogram[0].begin(),
			                                            nodes[node_idx2.idx].histogram[0].end(), 0.0);
			                node_idx.idx = nodes[node_idx.idx].left_child; // left child
			                double n_left = accumulate(nodes[node_idx.idx].histogram[0].begin(),
			                                           nodes[node_idx.idx].histogram[0].end(), 0.0);
			                node_idx.weight  *= n_left  / (n_left + n_right); // adjust weights
			                node_idx2.weight *= n_right / (n_left + n_right);
                            node_idx_stack.emplace(node_idx2); // add right child as new path to node index stack
                            // continue left child path until leaf node
                        }
	                } else { // value

                        // Go to left or right child node depending on split (feature, threshold)
                        if (X[i*n_features + nodes[node_idx.idx].feature] <=
                            nodes[node_idx.idx].threshold)
                            node_idx.idx = nodes[node_idx.idx].left_child;
                        else
                            node_idx.idx = nodes[node_idx.idx].right_child;
                    }
                }
                leaf_idx_stack.emplace(node_idx); // store leaf nodes
            }

            while (!leaf_idx_stack.empty()) { // combine results from all leaf nodes
	            IdxInfo leaf_idx = leaf_idx_stack.top();
	            leaf_idx_stack.pop();

                // Calculate classes probabilities for each output
                // based on number of samples per class histogram
                for (OutputsIdx_t o=0; o<n_outputs; ++o) {
                    double normalizer = 0.0;
                    for (ClassesIdx_t c = 0; c < n_classes[o]; ++c)
                        normalizer += nodes[leaf_idx.idx].histogram[o][c];
                    if (normalizer > 0.0) {
                        for (ClassesIdx_t c = 0; c < n_classes[o]; ++c) {
                            y_prob[i * n_outputs * n_classes_max +
                                   o * n_classes_max + c] +=
                                    leaf_idx.weight * nodes[leaf_idx.idx].histogram[o][c] / normalizer;
                        }
                    }
                }
            }
        }
    }

    // Calculate feature importances from the decision tree.

    void  Tree::calculate_feature_importances(double* importances) {

        // Initialize
        memset(importances, 0, n_features * sizeof(double));

        if (node_count == 0) return;

        // Loop: all nodes
        for (NodesIdx_t idx=0; idx<node_count; ++idx) {

            // Split node
            // leaf nodes do no have any children
            // so we only need to test for one of the children
            if (nodes[idx].left_child > 0) {
                // Accumulate improvements per feature
                importances[nodes[idx].feature] += nodes[idx].improvement;
            }
        }

        // Normalize (to 1)
        double  normalizer = 0.0;
        for (FeaturesIdx_t f=0; f<n_features; ++f) normalizer += importances[f];
        if (normalizer > 0.0) { // 0 when root is pure
            for (FeaturesIdx_t f=0; f<n_features; ++f) {
                importances[f] = importances[f] / normalizer;
            }
        }

    }

    // Serialize

    void  Tree::serialize(ofstream& fout) {

        fout.write((const char *)&n_outputs, sizeof(n_outputs));

        // Serialize Number of Classes (multi-output)
        for (OutputsIdx_t o=0; o<n_outputs; ++o) {
            fout.write((const char *) (&n_classes[o]), sizeof(n_classes[o]));
        }

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

        fin.read((char*)(&n_outputs), sizeof(n_outputs));

        // Deserialize Number of Classes (multi-output)
        for (OutputsIdx_t o=0; o<n_outputs; ++o) {
            ClassesIdx_t value;
            fin.read((char*)(&value), sizeof(value));
            n_classes.emplace_back(value);
        }

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

    GiniCriterion::GiniCriterion(OutputsIdx_t     n_outputs,
                                 ClassesIdx_t*    n_classes,
                                 ClassesIdx_t     n_classes_max,
                                 SamplesIdx_t     n_samples,
                                 ClassWeights_t*  class_weight)
    : n_outputs(n_outputs),
      n_classes(n_classes),
      n_classes_max(n_classes_max),
      n_samples(n_samples),
      class_weight(class_weight),
      // Create and initialize histograms
      // - all samples
      node_weighted_histogram(n_outputs, vector<Histogram_t> (n_classes_max, 0)),
      node_weighted_n_samples(n_outputs, 0.0),
      node_impurity(n_outputs, 0.0),
      // - samples with missing values
      node_weighted_histogram_NA(n_outputs, vector<Histogram_t> (n_classes_max, 0)),
      node_weighted_n_samples_NA(n_outputs, 0.0),
      node_impurity_NA(n_outputs, 0.0),
      // - samples with values
      node_weighted_histogram_values(n_outputs, vector<Histogram_t> (n_classes_max, 0)),
      node_weighted_n_samples_values(n_outputs, 0.0),
      node_impurity_values(n_outputs, 0.0),
      node_pos_NA(0),
      // - samples with values smaller than threshold (assigned to left child)
      node_weighted_histogram_threshold_left(n_outputs, vector<Histogram_t> (n_classes_max, 0)),
      node_weighted_n_samples_threshold_left(n_outputs, 0.0),
      node_impurity_threshold_left(n_outputs, 0.0),
      // -- plus missing values (assigned to left child)
      node_weighted_n_samples_threshold_left_NA(n_outputs, 0.0),
      node_impurity_threshold_left_NA(n_outputs, 0.0),
      // - samples with values greater than threshold (assigned to right child)
      node_weighted_histogram_threshold_right(n_outputs, vector<Histogram_t> (n_classes_max, 0)),
      node_weighted_n_samples_threshold_right(n_outputs, 0.0),
      node_impurity_threshold_right(n_outputs, 0.0),
      // -- plus missing values (assigned to right child)
      node_weighted_n_samples_threshold_right_NA(n_outputs, 0.0),
      node_impurity_threshold_right_NA(n_outputs, 0.0),
      node_pos_threshold(0) {  }

    // Calculate weighted class histograms for all outputs for current node.

    void  GiniCriterion::calculate_node_histogram(Classes_t*             y,
                                                  vector<SamplesIdx_t>&  samples,
                                                  SamplesIdx_t           start,
                                                  SamplesIdx_t           end) {

        for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently

            // Calculate class histogram
            // we use n_classes_max to create a nice 2D array to hold the class histograms
            // as the number of classes can be different for different outputs
            vector<Histogram_t>     histogram(n_classes_max, 0);

            for (SamplesIdx_t i = start; i < end; ++i) {
                histogram[y[samples[i] * n_outputs + o]]++;
            }

            // Apply class weights
            node_weighted_n_samples[o] = 0.0;
            Histogram_t weighted_cnt;
            for (ClassesIdx_t c = 0; c < n_classes[o]; ++c) {
                weighted_cnt = class_weight[o * n_classes_max + c] * histogram[c];
                node_weighted_histogram[o][c] = weighted_cnt;
                node_weighted_n_samples[o] += weighted_cnt;
            }
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

    // Calculate impurity for all outputs of the current node.

    void  GiniCriterion::calculate_node_impurity() {

        for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently

            node_impurity[o] =
                    calculate_impurity(node_weighted_histogram[o]);
        }
    }

    // Calculate class histograms for all outputs for the samples with missing values and the samples with values.

    void GiniCriterion::calculate_NA_histogram(Classes_t*               y,
                                               vector<SamplesIdx_t>&    samples,
                                               SamplesIdx_t             pos) {

        for (unsigned long o = 0; o < n_outputs; ++o) { // process each output independently

            // Calculate class histogram for the samples with missing values located in samples[0:pos]
            // we use n_classes_max to create a nice 2D array to hold the class histograms
            // as the number of classes can be different for different outputs
            vector<Histogram_t> histogram(n_classes_max, 0);

            for (SamplesIdx_t i = 0; i < pos; ++i) {
                histogram[y[samples[i] * n_outputs + o]]++;
            }

            // Apply class weights
            node_weighted_n_samples_NA[o] = 0.0;
            Histogram_t weighted_cnt;
            for (ClassesIdx_t c = 0; c < n_classes[o]; ++c) {
                weighted_cnt = class_weight[o*n_classes_max+c] * histogram[c];
                node_weighted_histogram_NA[o][c] = weighted_cnt;
                node_weighted_n_samples_NA[o] += weighted_cnt;
            }

            // Calculate class histogram for samples with values
            for (ClassesIdx_t c = 0; c < n_classes[o]; ++c) {

                node_weighted_histogram_values[o][c] = node_weighted_histogram[o][c] -
                                                       node_weighted_histogram_NA[o][c];
            }
            node_weighted_n_samples_values[o] = node_weighted_n_samples[o] -
                                                node_weighted_n_samples_NA[o];
        }

        // Update position
        node_pos_NA = pos;
    }

    // Calculate impurity for all outputs of samples with missing values and samples with values.

    void  GiniCriterion::calculate_NA_impurity() {

        for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently

            node_impurity_NA[o] =
                    calculate_impurity(node_weighted_histogram_NA[o]);

            node_impurity_values[o] =
                    calculate_impurity(node_weighted_histogram_values[o]);
        }
    }

    // Calculate impurity improvement from the current node to its children
    // assuming a split between missing values and values.

    double  GiniCriterion::calculate_NA_impurity_improvement() {

        vector<double> impurity_improvement(n_outputs, 0.0);

        for (unsigned long o=0; o<n_outputs; ++o) { // over all outputs

            impurity_improvement[o] +=
                   (node_weighted_n_samples[o] / n_samples) *
                   (node_impurity[o] -
                    (node_weighted_n_samples_NA[o] /
                     node_weighted_n_samples[o]) *
                    node_impurity_NA[o] -
                    (node_weighted_n_samples_values[o] /
                     node_weighted_n_samples[o]) *
                    node_impurity_values[o]);
            // OK to use "n_samples instead of "sum of all weighted samples"
            // given the way the class weights are calculated
        }

        // average
        return accumulate(impurity_improvement.begin(),impurity_improvement.end(), 0.0) / impurity_improvement.size();
    }

    // Initialize class histograms for all outputs for using a threshold on samples with values,
    // in the case that all samples have values.

    void  GiniCriterion::init_threshold_histograms() {

        for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently

            // Initialize class histogram for left child to 0
            // Initialize class histogram for right child to
            // class histogram from the current node
            for (ClassesIdx_t c = 0; c < n_classes[o]; ++c) {
                node_weighted_histogram_threshold_left[o][c] = 0.0;
                node_weighted_histogram_threshold_right[o][c] = node_weighted_histogram[o][c];
            }
            node_weighted_n_samples_threshold_left[o] = 0.0;
            node_weighted_n_samples_threshold_right[o] = node_weighted_n_samples[o];
        }

        // Update current position
        node_pos_threshold = 0;
    }

    // Initialize class histograms for all outputs for using a threshold on samples with values,
    // in the case that there are also samples with missing values.

    void  GiniCriterion::init_threshold_values_histograms() {

        for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently

            // Initialize class histogram for left child to 0
            // Initialize class histogram for right child to
            // class histogram of samples with values (vs missing values) from the current node
            for (ClassesIdx_t c = 0; c < n_classes[o]; ++c) {
                node_weighted_histogram_threshold_left[o][c] = 0.0;
                node_weighted_histogram_threshold_right[o][c] = node_weighted_histogram_values[o][c];
            }
            node_weighted_n_samples_threshold_left[o] = 0.0;
            node_weighted_n_samples_threshold_right[o] = node_weighted_n_samples_values[o];
        }

        // Update current position
        node_pos_threshold = node_pos_NA;

    }

    // Update class histograms for all outputs for using a threshold on values,
    // from current position to the new position (correspond to thresholds).

    void  GiniCriterion::update_threshold_histograms(Classes_t*             y,
                                                     vector<SamplesIdx_t>&  samples,
                                                     SamplesIdx_t           new_pos) {

        for (unsigned long o = 0; o < n_outputs; ++o) { // process each output independently

            // Calculate class histogram for samples[pos:new_pos]
            vector<Histogram_t> histogram(n_classes_max, 0);
            for (SamplesIdx_t i = node_pos_threshold; i < new_pos; ++i) {
                histogram[y[samples[i] * n_outputs + o]]++;
            }

            // Add class histogram for samples[pos:new_pos]
            // to class histogram for samples[0 or number missing values:pos] with values < threshold (left child)
            // Subtract class histogram for samples[pos:new_pos]
            // from class histogram for samples[pos: number of samples] with values > threshold (right child)
            Histogram_t weighted_cnt;
            for (ClassesIdx_t c = 0; c < n_classes[o]; ++c) {
                // Apply class weights
                weighted_cnt = class_weight[o * n_classes_max + c] * histogram[c];
                // Left child
                node_weighted_histogram_threshold_left[o][c] += weighted_cnt;
                node_weighted_n_samples_threshold_left[o] += weighted_cnt;
                // Right child
                node_weighted_histogram_threshold_right[o][c] -= weighted_cnt;
                node_weighted_n_samples_threshold_right[o] -= weighted_cnt;
            }
        }

        // Update current position (correspond to a threshold)
        node_pos_threshold = new_pos;
    }

    // Calculate impurity for all outputs of samples with values that are smaller and greater than a threshold.
    void  GiniCriterion::calculate_threshold_impurity() {

        for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently

            node_impurity_threshold_left[o] =
                    calculate_impurity(node_weighted_histogram_threshold_left[o]);

            node_impurity_threshold_right[o] =
                    calculate_impurity(node_weighted_histogram_threshold_right[o]);
        }
    }

    // Calculate the impurity for all outputs of samples with values that are smaller and greater than a threshold
    // and passing on the samples with missing values.

    void  GiniCriterion::calculate_threshold_NA_impurity() {

        for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently

            vector<Histogram_t> histogram(node_weighted_histogram_NA[o].size(), 0.0);

            // Samples with values that are smaller than a threshold and the samples with missing values
            for (ClassesIdx_t c = 0; c < node_weighted_histogram_NA[o].size(); ++c) {
                histogram[c] = node_weighted_histogram_threshold_left[o][c] +
                               node_weighted_histogram_NA[o][c];
            }

            node_impurity_threshold_left_NA[o] = calculate_impurity(histogram);

            node_weighted_n_samples_threshold_left_NA[o] =
                    node_weighted_n_samples_threshold_left[o] +
                    node_weighted_n_samples_NA[o];

            // Samples with values that are greater than a threshold and the samples with missing values
            for (ClassesIdx_t c = 0; c < node_weighted_histogram_NA[o].size(); ++c) {
                histogram[c] = node_weighted_histogram_threshold_right[o][c] +
                               node_weighted_histogram_NA[o][c];
            }

            node_impurity_threshold_right_NA[o] = calculate_impurity(histogram);

            node_weighted_n_samples_threshold_right_NA[o] =
                    node_weighted_n_samples_threshold_right[o] +
                    node_weighted_n_samples_NA[o];
        }
    }

    // Calculate the impurity improvement over all outputs from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // in the case that all samples have values.

    double  GiniCriterion::calculate_threshold_impurity_improvement() {

        vector<double> impurity_improvement(n_outputs, 0.0);

        for (unsigned long o=0; o<n_outputs; ++o) { // over all outputs

            impurity_improvement[o] +=
                   (node_weighted_n_samples[o] / n_samples) *
                   (node_impurity[o] -
                    (node_weighted_n_samples_threshold_left[o] /
                     node_weighted_n_samples[o]) *
                    node_impurity_threshold_left[o] -
                    (node_weighted_n_samples_threshold_right[o] /
                     node_weighted_n_samples[o]) *
                    node_impurity_threshold_right[o]);
            // OK to use "n_samples instead of "sum of all weighted samples"
            // given the way the class weights are calculated
        }

        // average
        return accumulate(impurity_improvement.begin(),impurity_improvement.end(), 0.0) / impurity_improvement.size();
    }

    // Calculate the impurity improvement over all outputs from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // in the case that there are also samples with missing values.

    double  GiniCriterion::calculate_threshold_values_impurity_improvement() {

        vector<double> impurity_improvement(n_outputs, 0.0);

        for (unsigned long o=0; o<n_outputs; ++o) { // over all outputs

            impurity_improvement[o] +=
                      (node_weighted_n_samples_values[o] / n_samples) *
                      (node_impurity_values[o] -
                       (node_weighted_n_samples_threshold_left[o] /
                        node_weighted_n_samples_values[o]) *
                        node_impurity_threshold_left[o] -
                       (node_weighted_n_samples_threshold_right[o] /
                        node_weighted_n_samples_values[o]) *
                        node_impurity_threshold_right[o]);
        // OK to use "n_samples instead of "sum of all weighted samples"
        // given the way the class weights are calculated
        }

        // average
        return accumulate(impurity_improvement.begin(),impurity_improvement.end(), 0.0) / impurity_improvement.size();
    }

    // Calculate the impurity improvement over all outputs from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // and passing on the samples with missing values to the left child.

    double  GiniCriterion::calculate_threshold_NA_left_impurity_improvement() {

        vector<double> impurity_improvement(n_outputs, 0.0);

        for (unsigned long o=0; o<n_outputs; ++o) { // over all outputs

            impurity_improvement[o] +=
               (node_weighted_n_samples[o] / n_samples) *
               (node_impurity[o] -
                   (node_weighted_n_samples_threshold_left_NA[o] /
                    node_weighted_n_samples[o]) *
                    node_impurity_threshold_left_NA[o] -
                   (node_weighted_n_samples_threshold_right[o] /
                    node_weighted_n_samples[o]) *
                    node_impurity_threshold_right[o]);
        }

        // average
        return accumulate(impurity_improvement.begin(),impurity_improvement.end(), 0.0) / impurity_improvement.size();
    }

    // Calculate the impurity improvement over all outputs from the current node to its children
    // assuming a split of the samples with values smaller and greater than a threshold
    // and passing on the samples with missing values to the right child.

    double  GiniCriterion::calculate_threshold_NA_right_impurity_improvement() {

        vector<double> impurity_improvement(n_outputs, 0.0);

        for (unsigned long o=0; o<n_outputs; ++o) { // over all outputs

            impurity_improvement[o] +=
               (node_weighted_n_samples[o] / n_samples) *
               (node_impurity[o] -
                   (node_weighted_n_samples_threshold_left[o] /
                    node_weighted_n_samples[o]) *
                    node_impurity_threshold_left[o] -
                   (node_weighted_n_samples_threshold_right_NA[o] /
                    node_weighted_n_samples[o]) *
                    node_impurity_threshold_right_NA[o]);
        }

        // average
        return accumulate(impurity_improvement.begin(),impurity_improvement.end(), 0.0) / impurity_improvement.size();
    }

// =============================================================================
// Node Splitter
// =============================================================================

    // Create and initialize a new best splitter.

    BestSplitter::BestSplitter(OutputsIdx_t         n_outputs,
                               ClassesIdx_t*        n_classes,
                               ClassesIdx_t         n_classes_max,
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
      criterion(n_outputs, n_classes, n_classes_max, n_samples, class_weight) {

        // Initialize samples[0, n_samples] to the training data X, y
        iota(samples.begin(), samples.end(), 0); // identity mapping
    }

    // Initialize node and calculate weighted histograms for all outputs and impurity for the node.

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
        threshold = numeric_limits<double>::quiet_NaN(); // no threshold
        pos = 0;
        improvement = 0.0; // leaf node

        // y is not constant (impurity > 0)
        // has been checked by impurity stop criteria in build()
        // moving on we can assume at least 2 samples

        // Copy f_X=X[samples[start:end],f]
        // training data X for the current node.

        SamplesIdx_t  n_samples = end - start;
        vector<Features_t>  f_X(n_samples);
        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            f_X[i] = X[s[i]*n_features + feature];
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

            // Calculate class histograms for all outputs for the samples with missing values and the samples with values
            criterion.calculate_NA_histogram(y, s, pNA);
            // Calculate impurity for all outputs of samples with missing values and samples with values
            criterion.calculate_NA_impurity();
            // Calculate impurity improvement over all outputs from the current node to its children
            // assuming a split between missing values and values
            improvement = criterion.calculate_NA_impurity_improvement();
            NA = 0; // pass all samples with missing values to the left child
            // pass all samples with values to the right child
            threshold = numeric_limits<double>::quiet_NaN(); // no threshold
            pos = start + pNA;

            // If impurity of all samples with values is 0.0 (pure) then stop
            // includes special case of having only 1 sample with a value
            if (criterion.get_node_impurity_values() < PRECISION_EQUAL) return;
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
                // Initialize class histograms for all outputs for using a threshold
                criterion.init_threshold_histograms();
            } else if (pNA > 0) {
                // Initialize class histograms for all outputs for using a threshold on samples with values
                criterion.init_threshold_values_histograms();
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
            Features_t     max_threshold = numeric_limits<double>::quiet_NaN(); // no threshold
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

                // Update class histograms for all outputs for using a threshold on values
                // from current position p to the new position np (correspond to thresholds)
                criterion.update_threshold_histograms(y, s, pn);

                // Calculate impurity for all outputs of samples with values that are smaller and greater than a threshold
                criterion.calculate_threshold_impurity();

                // Calculate impurity improvement over all outputs from the current node to its children
                // assuming a split of the samples with values smaller and greater than a threshold
                double  improvement_threshold = 0.0;
                if (pNA == 0) { // node has samples with values only
                    improvement_threshold =
                        criterion.calculate_threshold_impurity_improvement();
                } if (pNA > 0) { // node has samples with values and missing values
                    improvement_threshold =
                        criterion.calculate_threshold_values_impurity_improvement();
                }

                // Identify maximum impurity improvement
                if (improvement_threshold > max_improvement) {
                    max_improvement = improvement_threshold;
                    max_threshold = (f_X[p] + f_X[pn]) / 2.0;
                    max_pos = start + pn;
                }

                // If impurity of right child is 0.0 (pure) then stop
                if (criterion.get_node_impurity_threshold_right() < PRECISION_EQUAL) break;

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

                // Calculate impurity for all outputs of samples with values that are smaller and greater than a threshold
                // combined with the samples with missing values
                criterion.calculate_threshold_NA_impurity();
                // Calculate impurity improvement over all outputs
                double  improvement_threshold_NA_left =
                        criterion.calculate_threshold_NA_left_impurity_improvement();
                double  improvement_threshold_NA_right =
                        criterion.calculate_threshold_NA_right_impurity_improvement();

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
        threshold = numeric_limits<double>::quiet_NaN(); // no threshold
        pos = 0;
        improvement = 0.0; // leaf node

        // y is not constant (impurity > 0)
        // has been checked by impurity stop criteria in build()
        // moving on we can assume at least 2 samples

        // Copy f_X=X[samples[start:end],f]
        // training data X for the current node.

        SamplesIdx_t  n_samples = end - start;
        vector<Features_t>  f_X(n_samples);
        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            f_X[i] = X[s[i]*n_features + feature];
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
             (random_state.uniform_int(0, n_samples) < static_cast<long>((pNA-1))))) {
            // random number proportional to the number of NA values determines if a split is done

            // Calculate class histograms for all outputs for the samples with missing values and the samples with values
            criterion.calculate_NA_histogram(y, s, pNA);
            // Calculate impurity for all outputs of samples with missing values and samples with values
            criterion.calculate_NA_impurity();
            // Calculate impurity improvement over all outputs from the current node to its children
            // assuming a split between missing values and values
            improvement = criterion.calculate_NA_impurity_improvement();
            NA = 0; // pass all samples with missing values to the left child
            // pass all samples with values to the right child
            threshold = numeric_limits<double>::quiet_NaN(); // no threshold
            pos = start + pNA;

        // Split based on threshold
        // ------------------------

        } else {

            if (f_min + PRECISION_EQUAL < f_max) { // f_X is not constant

                // random threshold
                // ----------------
                // using the Extreme Random Tree formulation

                // Draw random threshold
                threshold = random_state.uniform_real(f_min + PRECISION_EQUAL, f_max);
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

                    // Initialize class histograms for all outputs for using a threshold
                    criterion.init_threshold_histograms();
                    // Update class histograms for all outputs for the children
                    // of the current node from position 0 to the position np
                    criterion.update_threshold_histograms(y, s, pn);
                    // Calculate impurity for all outputs of children
                    criterion.calculate_threshold_impurity();
                    // Calculate impurity improvement over all outputs
                    double  improvement_threshold = criterion.calculate_threshold_impurity_improvement();

                    improvement = improvement_threshold;
                    NA = -1; // no missing values
                    pos = start + pn;

                } else if (pNA > 0) { // node has samples with values and missing values

                    // Add missing values to split (based on threshold)
                    // ------------------------------------------------
                    // impurity improvement determines how the split is done

                    // Calculate class histograms for all outputs for the samples with missing values and the samples with values
                    criterion.calculate_NA_histogram(y, s, pNA);
                    // Initialize class histograms for all outputs for using a threshold on samples with values
                    criterion.init_threshold_values_histograms();
                    // Update class histograms for all outputs for the children
                    // of the current node from position 0 to the position np
                    criterion.update_threshold_histograms(y, s, pn);
                    // Calculate impurity for all outputs of children
                    criterion.calculate_threshold_impurity();
                    // Calculate impurity improvements over all outputs
                    double  improvement_threshold_NA_left =
                            criterion.calculate_threshold_NA_left_impurity_improvement();
                    double  improvement_threshold_NA_right =
                            criterion.calculate_threshold_NA_right_impurity_improvement();

                    if (improvement_threshold_NA_left >= improvement_threshold_NA_right) {
                        // Add missing values to left child
                        improvement = improvement_threshold_NA_left;
                        NA = 0; // missing values are passed on to the left child
                        pos = start + pn;
                    } else {
                        // Add missing values to right child
                        improvement = improvement_threshold_NA_right;
                        NA = 1; // missing values are passed on to the right child

                        // move samples with missing values to the end of the sample vector
                        vector<SamplesIdx_t> s_NA(&s[0], &s[pNA]); // temp for NAs
                        copy(&s[pNA], &s[n_samples], &s[0]);
                        copy(&s_NA[0], &s_NA[pNA], &s[n_samples - pNA]);
                        pos = start + pn - pNA;
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
        vector<SamplesIdx_t> s(end - start);
        copy(&samples[start], &samples[end], s.begin());

        // Loop: k random features (k defined by max_features)
        // ---------------------------------------------------

        // When max_features == n_features this is the same as
        // Loop: all features "for (f=0; f<n_features; ++f)",
        // but in a random order, which is preferable.

        // Features are sampled without replacement using
        // the modern version of the Fischer-Yates shuffle algorithm
        // in an iterative way.

        vector<FeaturesIdx_t>  features(n_features);
        iota(features.begin(), features.end(), 0); // identity mapping

        improvement = 0.0;
        auto i = n_features; // i=n instead of n-1 because of uniform_int(0,n)
        while ((i > (n_features - max_features)) || // includes case 0
                    (improvement < PRECISION_EQUAL && i > 0)) {
            // continue until at least one none constant feature was selected
            // or ultimately no more features are left

            unsigned long  j = 0;
            if (i>1) j = static_cast<unsigned long>(random_state.uniform_int(0, i)); // covers case 0
            // uniform_int(low, high), low is inclusive and high is exclusive
            --i; // adjust indexing by i
            swap(features[i],features[static_cast<FeaturesIdx_t>(j)]);
            FeaturesIdx_t  f = features[i];

            // Split feature
            int           f_NA = 0;
            Features_t    f_threshold = 0;
            SamplesIdx_t  f_pos = 0;
            double        f_improvement = improvement;

            if (max_thresholds == 0) {
                split_feature(X, y, s, f, f_NA, f_threshold, f_pos, f_improvement);
            } else if (max_thresholds == 1) { // Extreme Random Tree
                split_feature_extreme_random(X, y, s, f, f_NA, f_threshold, f_pos, f_improvement);
            }

            // keeping sorted samples s from feature run
            if (f_improvement > improvement) { // keep results for maximum improvement
                improvement = f_improvement;
                feature = f;
                NA = f_NA;
                threshold = f_threshold;
                pos = f_pos; // position f_pos corresponds to s samples
                // Replace samples[start:end] with properly ordered samples s
                copy(s.begin(), s.end(), &samples[start]);
            }
        }
    }

// =============================================================================
// Tree Builder
// =============================================================================

    // Create and initialize a new depth first tree builder.

    DepthFirstTreeBuilder::DepthFirstTreeBuilder(OutputsIdx_t           n_outputs,
                                                 ClassesIdx_t*          n_classes,
                                                 ClassesIdx_t           n_classes_max,
                                                 FeaturesIdx_t          n_features,
                                                 SamplesIdx_t           n_samples,
                                                 ClassWeights_t*        class_weight,
                                                 TreeDepthIdx_t         max_depth,
                                                 FeaturesIdx_t          max_features,
                                                 unsigned long          max_thresholds,
                                                 string                 missing_values,
                                                 RandomState const&     random_state)
    :   max_depth(max_depth),
        missing_values(move(missing_values)),
        // Create and initialize a new best splitter (and gini criterion)
        splitter(n_outputs,
                 n_classes,
                 n_classes_max,
                 n_features,
                 n_samples,
                 class_weight,
                 max_features,
                 max_thresholds,
                 random_state) {  }

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

               tree.nodes.reserve((1u << (max_depth+1))-1);

               // Push root node information onto the stack
               node_info_stack.emplace(NodeInfo(0, n_samples, 0, 0, false));
               // Loop: nodes
               while (!node_info_stack.empty()) {
                   // Pop current node information from the stack
                   NodeInfo  cn = node_info_stack.top();
                   node_info_stack.pop();

                   // Calculate number of samples per class histogram for all outputs
                   // and impurity for the current node
                   splitter.init_node(y, cn.start, cn.end);
                   vector<vector<Histogram_t>>  histogram =
                           splitter.criterion.get_node_weighted_histogram();
                   double  impurity = splitter.criterion.get_node_impurity();
                   SamplesIdx_t  pos = 0;

                   // If a stop criterion is met node becomes a leaf node
                   bool  is_leaf = (cn.depth >= max_depth) ||
                                   (impurity <= PRECISION_EQUAL);

                   // Split node (if node is not a leaf node)
                   FeaturesIdx_t  feature = 0;
                   int            NA = -1; // NA
                   Features_t     threshold = numeric_limits<double>::quiet_NaN(); // no threshold
                   double         improvement = 0.0;
                   if (!is_leaf) {
                       // Find the split on samples[start:end] that maximizes impurity improvement and
                       // partition samples[start:end] into samples[start:pos] and samples[pos:end]
                       // according to the split
                       splitter.split_node(X, y, feature, NA, threshold, pos, improvement);
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

    auto calculate_n_classes = [](const vector<vector<string>>& classes) {
        vector<ClassesIdx_t> n_classes(classes.size(), 0);
        for (OutputsIdx_t o=0; o<classes.size(); o++) {
            n_classes[o] = classes[o].size();
        }
        return n_classes;
    };

    DecisionTreeClassifier::DecisionTreeClassifier(vector<vector<string>> const&   classes,
                                                   vector<string> const&           features,
                                                   string const&                   class_balance,
                                                   TreeDepthIdx_t                  max_depth,
                                                   FeaturesIdx_t                   max_features,
                                                   unsigned long                   max_thresholds,
                                                   string const&                   missing_values,
                                                   long                            random_state_seed)
            :   n_outputs(classes.size()),
                classes(classes),
                n_classes(calculate_n_classes(classes)),
                features(features),
                n_features(features.size()),
                tree_(n_outputs, n_classes, n_features) {

        // for convenience

        DecisionTreeClassifier::n_classes_max = *max_element(begin(DecisionTreeClassifier::n_classes),
                                                             end(DecisionTreeClassifier::n_classes));

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
        if ( missing_values == "NMAR" || missing_values == "None")
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

    void DecisionTreeClassifier::fit(vector<Features_t>&   X,
                                     vector<Classes_t>&    y) {

        // number of samples
        SamplesIdx_t    n_samples = y.size() / n_outputs;
        if (n_samples != X.size() / n_features) {
            throw runtime_error("Mismatch: n_outputs, n_features and n_samples.");
        }

        // check that training data includes all classes across all outputs
        for (unsigned long o=0; o<n_outputs; ++o) {
            set<long> classesSet;
            for (unsigned long c = 0; c < n_classes[o]; ++c) { classesSet.insert(c); }
            for (unsigned long i = 0; i < n_samples; ++i) {
                classesSet.erase(y[i * n_outputs + o]);
                if (classesSet.empty()) continue;
            }
            if (!classesSet.empty()) {
                throw runtime_error("Training data does not include all classes.");
            }
        }

        // Calculate class weights for each output separately
        // so that n_samples == sum of all weighted samples

        // we use n_classes_max to create a nice 2D array to hold the class weights
        // as the number of classes can be different for different outputs
        vector<double>  class_weight(n_outputs * n_classes_max, 1.0);
        if (class_balance == "balanced") {
            for (unsigned long o=0; o<n_outputs; ++o) { // process each output independently
                vector<long> bincount(n_classes[o], 0);
                for (unsigned long i = 0; i < n_samples; ++i) {
                    bincount[y[i*n_outputs+o]]++;
                }
                for (unsigned long c = 0; c < n_classes[o]; ++c) {
                    class_weight[o*n_classes_max + c] =
                            (static_cast<double>(n_samples) / bincount[c]) / n_classes[o];
                }
            }
        }

        //  Build decision tree

        DepthFirstTreeBuilder builder(n_outputs,
                                      &n_classes[0],
                                      n_classes_max,
                                      n_features,
                                      n_samples,
                                      &class_weight[0],
                                      max_depth,
                                      max_features,
                                      max_thresholds,
                                      missing_values,
                                      random_state);

        builder.build(tree_, &X[0], &y[0], n_samples);

    }

    // Predict classes probabilities for the test data.

    void  DecisionTreeClassifier::predict_proba(Features_t*   X,
                                                SamplesIdx_t  n_samples,
                                                double*       y_prob) {

        tree_.predict(X, n_samples, y_prob);

    }

    // Predict classes for the test data.

    void  DecisionTreeClassifier::predict(Features_t*   X,
                                          SamplesIdx_t  n_samples,
                                          Classes_t*    y) {

        // We use n_classes_max to create a nice 3D array to hold the predicted values x samples x classes
        // as the number of classes can be different for different outputs

        vector<double>  y_prob(n_samples * n_outputs * n_classes_max, 0.0);
        predict_proba(X, n_samples, &y_prob[0]);


        for (SamplesIdx_t i=0; i<n_samples; ++i) {
            for (OutputsIdx_t o=0; o<n_outputs; ++o) {
                y[i*n_outputs + o] =
                        maxIndex(&y_prob[i * n_outputs * n_classes_max +
                                         o * n_classes_max],
                                 n_classes[o]);
            }
        }
    }

    // Calculate score for the test data.

    double DecisionTreeClassifier::score(Features_t*   X,
                                         Classes_t*    y,
                                         SamplesIdx_t  n_samples) {

        vector<long>    y_predict(n_samples*n_outputs, 0);
        predict(X, n_samples, &y_predict[0]);

        unsigned long n_true = 0;
        for (SamplesIdx_t i = 0; i < n_samples; ++i) {
            for (OutputsIdx_t o = 0; o < n_outputs; ++o) {
                if (y_predict[i*n_outputs + o] ==
                    y[i*n_outputs + o])
                    n_true++;
            }
        }
        return static_cast<double>(n_true) / (n_samples*n_outputs);

    }

    // Calculate feature importances from the decision tree.

    void  DecisionTreeClassifier::calculate_feature_importances(double*  importances) {

        tree_.calculate_feature_importances(importances);

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

    void  process_tree_recursively_graphviz(const Tree&                     tree,
                                            NodesIdx_t                      node_id,
                                            const vector<vector<int>>&      rgb_LUT,
                                            const vector<vector<string>>&   classes,
                                            const vector<string>&           features,
                                            bool                            rotate,
                                            ofstream&                       fout) {

        // Current node
        NodesIdx_t                      left_child = tree.nodes[node_id].left_child;
        NodesIdx_t                      right_child = tree.nodes[node_id].right_child;
        FeaturesIdx_t                   feature = tree.nodes[node_id].feature;
        int                             NA = tree.nodes[node_id].NA;
        Features_t                      threshold = tree.nodes[node_id].threshold;
        vector<vector<Histogram_t>>     histogram = tree.nodes[node_id].histogram;
        double                          impurity = tree.nodes[node_id].impurity;

        // Predictions
        vector<ClassesIdx_t>  c(tree.n_outputs, 0);
        for (unsigned long o=0; o<tree.n_outputs; ++o) {
            c[o] = maxIndex(&histogram[o][0], tree.n_classes[o]);
        }

        // Histograms
        stringstream  p_c;
        Histogram_t   n = accumulate(histogram[0].begin(), histogram[0].end(), 0.0); // use histogram from 1st output, all the same
        for (unsigned long o=0; o<tree.n_outputs; ++o) {
            p_c << "[" << setprecision(2) << setw(2) << (histogram[o][0] / n);
            for (ClassesIdx_t cc = 1; cc < tree.n_classes[o]; ++cc) {
                p_c << "," << setw(2) << (histogram[o][cc] / n);
            }
            p_c << "]";
            if (o < tree.n_outputs-1) { p_c << "\\n"; }
        }

        // Node color and intensity based on classification and impurity
        unsigned long classes_combination = c[0];
        for (unsigned long o=1; o<tree.n_outputs; ++o) {
            classes_combination += tree.n_classes[o-1]*c[o];
        }
        int  r = rgb_LUT[classes_combination][0];
        int  g = rgb_LUT[classes_combination][1];
        int  b = rgb_LUT[classes_combination][2];
        vector<double>  max_impurity(tree.n_outputs, 0.0);
        for (unsigned long o=0; o<tree.n_outputs; ++o) {
            max_impurity[o] = 1.0 - (1.0 / tree.n_classes[o]);
        }
        double  max_impurity_avrg = accumulate(max_impurity.begin(), max_impurity.end(), 0.0) / tree.n_outputs;
        int     alpha = static_cast<int>(255 * (max_impurity_avrg - impurity) / max_impurity_avrg);
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
                 << p_c.str() << "\\n";
            for (unsigned long o=0; o<tree.n_outputs; ++o) {
                node << classes[o][c[o]];
                if (o < tree.n_outputs-1) { node << " "; }
            }
            node << "\"" << ", fillcolor=\"" << color.str() << "\"] ;"
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
                // Order children based on prediction from first output
                // Left Child Prediction
                vector<Histogram_t> lc_histogram = tree.nodes[left_child].histogram[0];
                ClassesIdx_t        lc_c = maxIndex(&lc_histogram[0], lc_histogram.size());
                Histogram_t         lc_n = accumulate(lc_histogram.begin(), lc_histogram.end(), 0.0);
                double              lc_p_c = lc_histogram[lc_c] / lc_n;
                // Right Child Prediction
                vector<Histogram_t> rc_histogram = tree.nodes[right_child].histogram[0];
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
            vector<Histogram_t>  root_histogram = tree.nodes[0].histogram[0]; // use histogram from 1st output, all the same
            vector<Histogram_t>  left_child_histogram = tree.nodes[left_child].histogram[0];
            vector<Histogram_t>  right_child_histogram = tree.nodes[right_child].histogram[0];
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

    void  DecisionTreeClassifier::export_graphviz(string const& filename,
                                                  bool          rotate) {

        string fn = filename + ".gv";

        ofstream  fout(fn);
        if (fout.is_open()) {

            fout << "digraph Tree {" << endl;
            fout << "node [shape=box, style=\"rounded, filled\", color=\"black\", fontname=helvetica, fontsize=14] ;" << endl;
            fout << "edge [fontname=helvetica, fontsize=12] ;" << endl;

            // Rotate (default: top-down)
            if (rotate) fout << "rankdir=LR ;" << endl; // left-right orientation

            // Define rgb colors for the different classes over all outputs
            unsigned long n_classes_combinations = 1;
            for (unsigned long o=0; o<tree_.n_outputs; ++o) {
                n_classes_combinations *= tree_.n_classes[o];
            }
            vector<vector<int>>  rgb_LUT = create_rgb_LUT(n_classes_combinations);

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
        NodesIdx_t                      left_child = tree.nodes[node_id].left_child;
        NodesIdx_t                      right_child = tree.nodes[node_id].right_child;
        FeaturesIdx_t                   feature = tree.nodes[node_id].feature;
        int                             NA = tree.nodes[node_id].NA;
        Features_t                      threshold = tree.nodes[node_id].threshold;
        vector<vector<Histogram_t>>     histogram = tree.nodes[node_id].histogram;

        // Histogram formatting as string
        stringstream         histogram_string;
        for (unsigned long o=0; o<tree.n_outputs; ++o) {
            histogram_string << "[" << histogram[o][0];
            for (ClassesIdx_t c = 1; c < tree.n_classes[o]; ++c) {
                histogram_string << ", " << histogram[o][c];
            }
            histogram_string << "]";
        }

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

        // Number of Outputs
        fout.write((const char*)(&n_outputs), sizeof(n_outputs));

        // Classes
        for (OutputsIdx_t o=0; o<n_outputs; ++o) {
            fout.write((const char *) (&n_classes[o]), sizeof(n_classes[o]));
        }
        for (OutputsIdx_t o=0; o<n_outputs; ++o) {
            for (unsigned long c=0; c<n_classes[o]; ++c) {
                unsigned long size = classes[o][c].size();
                fout.write((const char *) &size, sizeof(size));
                fout.write((const char *) &classes[o][c][0], size);
            }
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

    void  DecisionTreeClassifier::export_serialize(string const& filename) {

        string fn = filename + ".dtc";

        ofstream  fout(fn, ios_base::binary);
        if (fout.is_open()) {

            const int  version = 2; // file version number
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

        // Number of Outputs
        OutputsIdx_t                n_outputs;
        fin.read((char*)(&n_outputs), sizeof(n_outputs));

        // Classes
        vector<ClassesIdx_t>        n_classes;
        for (OutputsIdx_t o=0; o<n_outputs; ++o) {
            ClassesIdx_t o_n_classes;
            fin.read((char *) (&o_n_classes), sizeof(o_n_classes));
            n_classes.emplace_back(o_n_classes);
        }
        vector<vector<string>>      classes;
        for (OutputsIdx_t o=0; o<n_outputs; ++o) {
            vector<string>          o_classes;
            for (unsigned long c=0; c<n_classes[o]; ++c) {
                string str;
                unsigned long  size;
                fin.read((char*)(&size), sizeof(size));
                str.resize(size);
                fin.read((char*)(&str[0]), size);
                o_classes.emplace_back(str);
            }
            classes.emplace_back(o_classes);
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

        DecisionTreeClassifier  dtc(classes, features,
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

    DecisionTreeClassifier  DecisionTreeClassifier::import_deserialize(string const& filename) {

        string fn = filename + ".dtc";

        ifstream  fin(fn, ios_base::binary);
        if (fin.is_open()) {

            int  version;
            fin.read((char*)(&version), sizeof(version));

            if (version == 2) { // file version number

                // Deserialize Decision Tree Classifier
                DecisionTreeClassifier dtc = deserialize(fin);

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


