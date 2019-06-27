#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>
#include <exception>

#include "../decision_tree.h"
#include "../utilities.h"

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

using namespace std;
using namespace koho;

template <class T>
bool compareVectors(std::vector<T> a, std::vector<T> b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != Approx(b[i])) {
            cout << a[i] << " should == " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// simple example (User's Guide C++)
// =================================

TEST_CASE("simple example") {

    // using multi-output with a single output

    vector<vector<string>> classes = {{"0", "1", "2", "3", "4", "5", "6", "7"}};
    vector<string> features = {"2^2", "2^1", "2^0"};

    // Hyperparameters
    string class_balance = "balanced";
    long max_depth = 3;
    long max_features = 0;
    long max_thresholds = 0;
    string missing_values = "None";
    // Random Number Generator
    long random_state = 0;

    vector<double> X = {0, 0, 0,
                        0, 0, 1,
                        0, 1, 0,
                        0, 1, 1,
                        1, 0, 0,
                        1, 0, 1,
                        1, 1, 0,
                        1, 1, 1};
    vector<long> y = {0,
                      1,
                      2,
                      3,
                      4,
                      5,
                      6,
                      7};

    try {

        DecisionTreeClassifier dtc(classes, features,
                                   class_balance, max_depth,
                                   max_features, max_thresholds,
                                   missing_values,
                                   random_state);

        dtc.fit(X, y);

        SECTION("Training") {
            //cout << dtc.export_text() << endl;
            REQUIRE(dtc.export_text().compare(
                    "0 X[1]<=0.5 [1, 1, 1, 1, 1, 1, 1, 1]; 0->1; 0->8; 1 X[2]<=0.5 [1, 1, 0, 0, 1, 1, 0, 0]; 1->2; 1->5; 2 X[0]<=0.5 [1, 0, 0, 0, 1, 0, 0, 0]; 2->3; 2->4; 3 [1, 0, 0, 0, 0, 0, 0, 0]; 4 [0, 0, 0, 0, 1, 0, 0, 0]; 5 X[0]<=0.5 [0, 1, 0, 0, 0, 1, 0, 0]; 5->6; 5->7; 6 [0, 1, 0, 0, 0, 0, 0, 0]; 7 [0, 0, 0, 0, 0, 1, 0, 0]; 8 X[2]<=0.5 [0, 0, 1, 1, 0, 0, 1, 1]; 8->9; 8->12; 9 X[0]<=0.5 [0, 0, 1, 0, 0, 0, 1, 0]; 9->10; 9->11; 10 [0, 0, 1, 0, 0, 0, 0, 0]; 11 [0, 0, 0, 0, 0, 0, 1, 0]; 12 X[0]<=0.5 [0, 0, 0, 1, 0, 0, 0, 1]; 12->13; 12->14; 13 [0, 0, 0, 1, 0, 0, 0, 0]; 14 [0, 0, 0, 0, 0, 0, 0, 1]; ") == 0);
        }

        SECTION("Feature Importances") {
            vector<double> importances(features.size());
            dtc.calculate_feature_importances(&importances[0]);
            //for (auto i: importances) cout << i << ' ';
            REQUIRE(compareVectors(importances, {0.571429, 0.142857, 0.285714}));
        }

        // Visualize Trees
        std::string filename = "simple_example_dtc";
        dtc.export_graphviz(filename, true);
        // $: xdot simple_example.gv
        // $: dot -Tpdf simple_example.gv -o simple_example.pdf

        dtc.export_serialize(filename);
        DecisionTreeClassifier dtc2 = DecisionTreeClassifier::import_deserialize(filename);

        SECTION("Persistence") {
            //cout << dtc2.export_text() << endl;
            REQUIRE(dtc2.export_text().compare(dtc.export_text()) == 0);
        }

        unsigned long   n_samples = y.size();

        SECTION("Classification") {
            vector<long> c(n_samples, 0);
            dtc2.predict(&X[0], n_samples, &c[0]);
            //for (auto i: c) cout << i << ' ';
            REQUIRE(compareVectors(c, y));
        }

        SECTION("Testing") {
            double score = dtc2.score(&X[0], &y[0], n_samples);
            REQUIRE(score == Approx(1.0));
        }

    } catch (exception &e) {
        cout << e.what() << '\n';
    }
}

// simple example multi-output
// ===========================

TEST_CASE("simple example multi-output") {

    // using multi-output with a single output

    vector<vector<string>> classes = {{"0", "1", "2", "3", "4", "5", "6", "7"},
                                      {"0", "4"},
                                      {"0", "2"},
                                      {"0", "1"}};
    vector<string> features = {"2^2", "2^1", "2^0"};

    // Hyperparameters
    string class_balance = "balanced";
    long max_depth = 0;
    long max_features = 0;
    long max_thresholds = 0;
    string missing_values = "None";
    // Random Number Generator
    long random_state = 0;

    vector<double> X = {0, 0, 0,
                        0, 0, 1,
                        0, 1, 0,
                        0, 1, 1,
                        1, 0, 0,
                        1, 0, 1,
                        1, 1, 0,
                        1, 1, 1};
    vector<long> y = {0, 0, 0, 0,
                      1, 0, 0, 1,
                      2, 0, 1, 0,
                      3, 0, 1, 1,
                      4, 1, 0, 0,
                      5, 1, 0, 1,
                      6, 1, 1, 0,
                      7, 1, 1, 1};

    try {

        DecisionTreeClassifier dtc(classes, features,
                                   class_balance, max_depth,
                                   max_features, max_thresholds,
                                   missing_values,
                                   random_state);

        dtc.fit(X, y);

        SECTION("Training") {
            //cout << dtc.export_text() << endl;
            REQUIRE(dtc.export_text().compare(
            "0 X[1]<=0.5 [1, 1, 1, 1, 1, 1, 1, 1][4, 4][4, 4][4, 4]; 0->1; 0->8; 1 X[2]<=0.5 [1, 1, 0, 0, 1, 1, 0, 0][2, 2][4, 0][2, 2]; 1->2; 1->5; 2 X[0]<=0.5 [1, 0, 0, 0, 1, 0, 0, 0][1, 1][2, 0][2, 0]; 2->3; 2->4; 3 [1, 0, 0, 0, 0, 0, 0, 0][1, 0][1, 0][1, 0]; 4 [0, 0, 0, 0, 1, 0, 0, 0][0, 1][1, 0][1, 0]; 5 X[0]<=0.5 [0, 1, 0, 0, 0, 1, 0, 0][1, 1][2, 0][0, 2]; 5->6; 5->7; 6 [0, 1, 0, 0, 0, 0, 0, 0][1, 0][1, 0][0, 1]; 7 [0, 0, 0, 0, 0, 1, 0, 0][0, 1][1, 0][0, 1]; 8 X[2]<=0.5 [0, 0, 1, 1, 0, 0, 1, 1][2, 2][0, 4][2, 2]; 8->9; 8->12; 9 X[0]<=0.5 [0, 0, 1, 0, 0, 0, 1, 0][1, 1][0, 2][2, 0]; 9->10; 9->11; 10 [0, 0, 1, 0, 0, 0, 0, 0][1, 0][0, 1][1, 0]; 11 [0, 0, 0, 0, 0, 0, 1, 0][0, 1][0, 1][1, 0]; 12 X[0]<=0.5 [0, 0, 0, 1, 0, 0, 0, 1][1, 1][0, 2][0, 2]; 12->13; 12->14; 13 [0, 0, 0, 1, 0, 0, 0, 0][1, 0][0, 1][0, 1]; 14 [0, 0, 0, 0, 0, 0, 0, 1][0, 1][0, 1][0, 1]; ") == 0);
        }

        SECTION("Feature Importances") {
            vector<double> importances(features.size());
            dtc.calculate_feature_importances(&importances[0]);
            //for (auto i: importances) cout << i << ' ';
            REQUIRE(compareVectors(importances, {0.421053, 0.263158, 0.315789}));
        }

        // Visualize Trees
        std::string filename = "simple_example_multi_output_dtc";
        dtc.export_graphviz(filename, true);
        // $: xdot simple_example.gv
        // $: dot -Tpdf simple_example.gv -o simple_example.pdf

        dtc.export_serialize(filename);
        DecisionTreeClassifier dtc2 = DecisionTreeClassifier::import_deserialize(filename);

        SECTION("Persistence") {
            //cout << dtc2.export_text() << endl;
            REQUIRE(dtc2.export_text().compare(dtc.export_text()) == 0);
        }

        unsigned long   n_outputs = classes.size();
        unsigned long   n_samples = y.size() / n_outputs;

        SECTION("Classification") {
            vector<long> c(n_samples*n_outputs, 0);
            dtc2.predict(&X[0], n_samples, &c[0]);
            //for (unsigned long i = 0; i < n_samples; ++i) {
            //    for (unsigned long o = 0; o < n_outputs; ++o) {
            //        cout << c[i * n_outputs + o] << ' ';
            //     }
            //    cout << endl;
            //}
            REQUIRE(compareVectors(c, y));
        }

        SECTION("Testing") {
            double score = dtc2.score(&X[0], &y[0], n_samples);
            REQUIRE(score == Approx(1.0));
        }

    } catch (exception &e) {
        cout << e.what() << '\n';
    }
}