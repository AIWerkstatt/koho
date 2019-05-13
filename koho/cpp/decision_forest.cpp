/// Decision Forest module.
/** @file
*/

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

// Basic concepts for the implementation of the classifier are based on
// G. Louppe, “Understanding Random Forests”, PhD Thesis, 2014

#include <iostream>
#include <set>

#include "utilities.h"
#include "decision_tree.h"
#include "decision_forest.h"

using namespace std;

namespace koho {

// =============================================================================
// Decision Forest Classifier
// =============================================================================

    // Create and initialize a new decision forest classifier.

    DecisionForestClassifier::DecisionForestClassifier(vector<string>  classes,
                                                       ClassesIdx_t    n_classes,
                                                       vector<string>  features,
                                                       FeaturesIdx_t   n_features,
                                                       unsigned long   n_estimators,
                                                       bool            bootstrap,
                                                       bool            oob_score,
                                                       string const&   class_balance,
                                                       TreeDepthIdx_t  max_depth,
                                                       FeaturesIdx_t   max_features,
                                                       unsigned long   max_thresholds,
                                                       string const&   missing_values,
                                                       long            random_state_seed) {

        DecisionForestClassifier::classes = std::move(classes);
        DecisionForestClassifier::n_classes = n_classes;
        DecisionForestClassifier::features = std::move(features);
        DecisionForestClassifier::n_features = n_features;

        // Check hyperparameters

        // n estimators
        if ((0 < n_estimators))
            DecisionForestClassifier::n_estimators = n_estimators;
        else
            DecisionForestClassifier::n_estimators = 100; // default

        // bootstrap
        DecisionForestClassifier::bootstrap = bootstrap;

        // oob_score
        if (bootstrap)
            DecisionForestClassifier::oob_score = oob_score;
        else
            DecisionForestClassifier::oob_score = false;

        // class balance
        if (class_balance == "balanced" || class_balance == "None")
            DecisionForestClassifier::class_balance = class_balance;
        else
            DecisionForestClassifier::class_balance = "balanced"; // default

        // max depth
        const TreeDepthIdx_t MAX_DEPTH = 2147483647; // max long: (2^31)-1
        if ((0 < max_depth) && (max_depth <= MAX_DEPTH))
            DecisionForestClassifier::max_depth = max_depth;
        else
            DecisionForestClassifier::max_depth = MAX_DEPTH;

        // max features
        if ((0 < max_features) && (max_features <= DecisionForestClassifier::n_features))
            DecisionForestClassifier::max_features = max_features;
        else
            DecisionForestClassifier::max_features = DecisionForestClassifier::n_features;

        // max thresholds
        if ((max_thresholds == 0) || (max_thresholds == 1))
            DecisionForestClassifier::max_thresholds = max_thresholds;
        else
            DecisionForestClassifier::max_thresholds = 0;

        // missing values
        if (missing_values == "NMAR" || missing_values == "None")
            DecisionForestClassifier::missing_values = missing_values;
        else
            DecisionForestClassifier::missing_values = "None"; // default

        // Random Number Generator

        if (random_state_seed == -1)
            DecisionForestClassifier::random_state = RandomState();
        else
            DecisionForestClassifier::random_state = RandomState(static_cast<unsigned long>(random_state_seed));

    }

    // Build a decision forest classifier from the training data.

    void DecisionForestClassifier::fit(Features_t*   X,
                                       Classes_t*    y,
                                       SamplesIdx_t  n_samples) {

        // Create explicitly different seeds for the decision trees
        // to avoid building the same tree over and over again for the entire decision forest
        // when decision trees are build in parallel.
        vector<long> algo_seeds(n_estimators);
        for (unsigned long e = 0; e < n_estimators; ++e) {
            algo_seeds[e] = random_state.uniform_int(0, random_state.MAX_INT);
        }

        // Instantiate decision trees
        for (unsigned long e = 0; e < n_estimators; ++e) {
            dtc_.emplace_back(DecisionTreeClassifier(classes, n_classes,
                                                     features, n_features,
                                                     class_balance, max_depth,
                                                     max_features, max_thresholds,
                                                     missing_values,
                                                     algo_seeds[e]));
        }
        oob_score_ = 0.0;

        // Build decision trees from training data
        if (!bootstrap) {

            // >>> mapping embarrassing parallelism
            for (unsigned long e = 0; e < n_estimators; ++e) {
                dtc_[e].fit(&X[0], &y[0], n_samples); // decision trees
            }

        } else { // Bagging & Out_Of-Bag estimate

            // Different seeds for algorithm and data (bagging)
            // to avoid building the same trees multiple times
            // when the same seed comes up again.
            vector<long> data_seeds(n_estimators);
            for (unsigned long e = 0; e < n_estimators; ++e) {
                data_seeds[e] = random_state.uniform_int(0, random_state.MAX_INT);
            }

            // >>> mapping embarrassing parallelism
            vector<vector<double>> ps;
            ClassesIdx_t n_classes = set<long>(&y[0], &y[0]+n_samples).size();
            for (unsigned long e = 0; e < n_estimators; ++e) {

                // Build a decision tree from the bootstrapped training data
                // drawing random samples with replacement

                vector<SamplesIdx_t> idx(n_samples);
                RandomState random_state = RandomState(static_cast<unsigned long>(data_seeds[e]));
                for (SamplesIdx_t s = 0; s < n_samples; ++s) {
                    idx[s] = static_cast<SamplesIdx_t>(random_state.uniform_int(0, n_samples));
                }

                vector<double> X_train;
                vector<long>   y_train;
                for (SamplesIdx_t s = 0; s < n_samples; ++s) { // samples
                    for (FeaturesIdx_t f = 0; f < n_features; ++f) { // features
                        X_train.emplace_back(X[idx[s] * n_features + f]);
                    }
                    y_train.emplace_back(y[idx[s]]);
                }
                unsigned long n_samples_train = n_samples;

                // make sure training data includes all classes

                while (set<long>(&y_train[0], &y_train[0]+n_samples_train).size() < n_classes) {

                    X_train.clear();
                    y_train.clear();

                    for (SamplesIdx_t s = 0; s < n_samples; ++s) {
                        idx[s] = static_cast<SamplesIdx_t>(random_state.uniform_int(0, n_samples));
                    }

                    for (SamplesIdx_t s = 0; s < n_samples; ++s) { // samples
                        for (FeaturesIdx_t f = 0; f < n_features; ++f) { // features
                            X_train.emplace_back(X[idx[s] * n_features + f]);
                        }
                        y_train.emplace_back(y[idx[s]]);
                    }
                }

                dtc_[e].fit(&X_train[0], &y_train[0], n_samples_train); // decision trees

                // Compute Out-Of-Bag estimates
                // as average error for all samples when not included in bootstrap

                vector<double> p(n_samples * n_classes, 0.0);
                if (oob_score) {
                    vector<bool> unsampled_idx(n_samples, true);
                    for (SamplesIdx_t s = 0; s < n_samples; ++s) unsampled_idx[idx[s]] = false;

                    vector<double> X_test;
                    unsigned long  n_samples_test = 0;
                    for (SamplesIdx_t s = 0; s < n_samples; ++s) { // samples
                        if (unsampled_idx[s] == 0) { // unsampled
                            for (FeaturesIdx_t f = 0; f < n_features; ++f) { // features
                                X_test.emplace_back(X[idx[s] * n_features + f]);
                            }
                            n_samples_test++;
                        }
                    }
                    vector<double> y_prob(n_samples_test * n_classes, 0.0);
                    dtc_[e].predict_proba(&X_test[0], n_samples_test, &y_prob[0]);

                    unsigned long i = 0;
                    for (SamplesIdx_t s = 0; s < n_samples; ++s) { // samples
                        if (unsampled_idx[s] == 0) { // unsampled
                            p[s] = y_prob[i++];
                        }
                    }
                }
                ps.emplace_back(p);
            }
            if (oob_score) {

                // Predict classes probabilities for the decision forest
                // as average of the class probabilities from all decision trees
                // >>> reduce
                vector<double> class_probabilities(n_samples * n_classes, 0.0);
                vector<bool> valid_idx(n_samples, false);
                unsigned long n_valid_idx = 0;
                for (SamplesIdx_t s = 0; s < n_samples; ++s) {
                    for (ClassesIdx_t c = 0; c < n_classes; ++c) {
                        double sum = 0.0;
                        for (unsigned long e = 0; e < n_estimators; ++e) {
                            sum += ps[e][s * n_classes + c];
                        }
                        class_probabilities[s * n_classes + c] = sum; // no normalization needed when using maxIndex( )
                        // Identify samples with oob score
                        if (sum > 0.0) {
                            valid_idx[s] = true;
                            n_valid_idx++;
                        }
                    }
                }

                if (n_valid_idx == n_samples) { // oob score for all samples

                    // Predict classes
                    vector<long> predictions(n_samples, 0);
                    for (SamplesIdx_t s = 0; s < n_samples; ++s) {
                        predictions[s] = maxIndex(&class_probabilities[s * n_classes], n_classes);
                    }

                    // Score
                    unsigned long n_true = 0;
                    for (SamplesIdx_t s = 0; s < n_samples; ++s) {
                        if (valid_idx[s]) {
                            if (y[s] == predictions[s]) n_true++;
                        }
                    }
                    oob_score_ = static_cast<double>(n_true) / n_valid_idx;

                } else {
                    oob_score_ = 0.0;
                    cout << "Only " << n_valid_idx << " out of " << n_samples
                         << "have an out-of-bag estimate. "
                         << "This probably means too few estimators were used "
                         << "to compute any reliable oob estimates." << endl;
                }
            }
        }

    };

    // Predict classes probabilities for the test data.

    void DecisionForestClassifier::predict_proba(Features_t*   X,
                                                 SamplesIdx_t  n_samples,
                                                 double*       y_prob) {

        // Predict class probabilities for all decision trees

        // >>> mapping embarrassing parallelism
        vector<vector<double>> ps;
        for (unsigned long e = 0; e < n_estimators; ++e) {
            vector<double> p(n_samples * n_classes, 0.0);
            DecisionForestClassifier::dtc_[e].predict_proba(&X[0], n_samples, &p[0]);
            ps.emplace_back(p);
        }

        // Predict classes probabilities for the decision forest
        // as average of the class probabilities from all decision trees

        // >>> reduce
        for (SamplesIdx_t s = 0; s < n_samples; ++s) {
            for (ClassesIdx_t c = 0; c < n_classes; ++c) {
                double sum = 0.0;
                for (unsigned long e = 0; e < n_estimators; ++e) {
                    sum += ps[e][s*n_classes + c];
                }
                y_prob[s*n_classes + c] = sum / n_estimators;
            }
        }

    }

    // Predict classes for the test data.

    void DecisionForestClassifier::predict(Features_t*   X,
                                           SamplesIdx_t  n_samples,
                                           Classes_t*    y) {

        vector<double>  y_prob(n_samples*n_classes, 0.0);
        predict_proba(X, n_samples, &y_prob[0]);

        for (SamplesIdx_t s=0; s<n_samples; ++s) {
            y[s] = maxIndex(&y_prob[s*n_classes], n_classes);
        }

    }

    // Calculate score for the test data.

    double DecisionForestClassifier::score(Features_t*   X,
                                           Classes_t*    y,
                                           SamplesIdx_t  n_samples) {

        std::vector<long>    y_predict(n_samples, 0);
        DecisionForestClassifier::predict(X, n_samples, &y_predict[0]);

        unsigned long n_true = 0;
        for (unsigned long i = 0; i < n_samples; ++i) {
            if (y_predict[i] == y[i]) n_true++;
        }
        return static_cast<double>(n_true) / n_samples;

    }

    // Calculate feature importances from the decision forest.

    void DecisionForestClassifier::calculate_feature_importances(double *importances) {

        // Calculate feature importances for all decision trees
        // >>> mapping embarrassing parallelism
        vector<vector<double>> dtc_importances(n_estimators, vector<double>(n_features, 0.0));
        for (unsigned long e = 0; e < n_estimators; ++e) {
            dtc_[e].calculate_feature_importances(&dtc_importances[e][0]);
        }

        // Calculate feature importances for the decision forest
        // as average of feature importances from all decision trees
        // >>> reduce
        for (FeaturesIdx_t f = 0; f < n_features; ++f) {
            double sum = 0.0;
            for (unsigned long e = 0; e < n_estimators; ++e) {
                sum += dtc_importances[e][f];
            }
            importances[f] = sum / n_estimators;
        }

    }

    // Export of a decision forest as individual decision trees in GraphViz dot format.

    void DecisionForestClassifier::export_graphviz(std::string const& filename, bool rotate) {

        for (unsigned long e = 0; e < n_estimators; ++e) {
            dtc_[e].export_graphviz(filename + "_" + to_string(e), rotate);
        }

    }

    // Export of a decision tree from a decision forest in GraphViz dot format.

    void DecisionForestClassifier::export_graphviz(std::string const& filename, unsigned long e, bool rotate) {

        dtc_[e].export_graphviz(filename + "_" + to_string(e), rotate);

    }

    // Export of a decision tree from a decision forest in a simple text format.

    std::string  DecisionForestClassifier::export_text(unsigned long e) {

        return dtc_[e].export_text();

    }

    // Serialize

    void  DecisionForestClassifier::serialize(std::ofstream& fout) {

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
        fout.write((const char*)&n_estimators, sizeof(n_estimators));
        fout.write((const char*)&bootstrap, sizeof(bootstrap));
        fout.write((const char*)&oob_score, sizeof(oob_score));
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
        // Serialize Decision Trees done separately
        fout.write((const char*)&oob_score_, sizeof(oob_score_));
    }

    // Export of a decision forest classifier in binary serialized format
    // with separate files for the individual decision trees.

    void DecisionForestClassifier::export_serialize(std::string const& filename) {

        string fn = filename + ".dfc";

        ofstream  fout(fn, ios_base::binary);
        if (fout.is_open()) {

            const int  version = 1; // file version number
            fout.write((const char*)&version, sizeof(version));

            // Serialize Decision Forest Classifier
            serialize(fout);

            fout.close();

            // Export of decision tree classifiers in binary serialized format
            for (unsigned long e = 0; e < n_estimators; ++e) {
                dtc_[e].export_serialize(filename + "_" + to_string(e));
            }

            return;

        } else {
            throw runtime_error("Unable to open file.");
        }
    }

    // Deserialize

    DecisionForestClassifier  DecisionForestClassifier::deserialize(std::ifstream& fin) {

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
        unsigned long   n_estimators;
        bool            bootstrap;
        bool            oob_score;
        string          class_balance;
        TreeDepthIdx_t  max_depth;
        FeaturesIdx_t   max_features;
        unsigned long   max_thresholds;
        string          missing_values;

        fin.read((char*)(&n_estimators), sizeof(n_estimators));
        fin.read((char*)(&bootstrap), sizeof(bootstrap));
        fin.read((char*)(&oob_score), sizeof(oob_score));
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

        DecisionForestClassifier  dfc(classes, n_classes,
                                      features, n_features,
                                      n_estimators, bootstrap, oob_score,
                                      class_balance, max_depth,
                                      max_features, max_thresholds,
                                      missing_values,
                                      random_state_seed);

        // Random Number Generator - overwrite random state
        fin.read((char*)(&dfc.random_state), sizeof(dfc.random_state));

        // Model
        // Deserialize Decision Trees separately done
        fin.read((char*)(&dfc.oob_score_), sizeof(dfc.oob_score_));

        return dfc;
    }

    // Import of a decision forest classifier in binary serialized format
    // with separate files for the individual decision trees.

    DecisionForestClassifier DecisionForestClassifier::import_deserialize(std::string const& filename) {

        string fn = filename + ".dfc";

        ifstream  fin(fn, ios_base::binary);
        if (fin.is_open()) {

            int  version;
            fin.read((char*)(&version), sizeof(version));

            if (version == 1) { // file version number

                // Deserialize Decision Forest Classifier
                DecisionForestClassifier  dfc = DecisionForestClassifier::deserialize(fin);

                fin.close();

                // Import of decision tree classifiers in binary serialized format
                for (unsigned long e = 0; e < dfc.n_estimators; ++e) {
                    DecisionTreeClassifier dtc = DecisionTreeClassifier::import_deserialize(filename + "_" + to_string(e));
                    dfc.dtc_.emplace_back(dtc);
                }

                return dfc;

            } else {
                fin.close();
                throw runtime_error("Unsupported file version number.");
            }
        } else {
            throw runtime_error("Unable to open file.");
        }
    }

} // namespace koho


