#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>
#include <exception>

#include "../decision_forest.h"
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

// simple example
// ==============

TEST_CASE("simple example"){

    // using multi-output with a single output

    vector<vector<string>> classes = {{"0", "1", "2", "3", "4", "5", "6", "7"}};
    vector<string> features = {"2^2", "2^1", "2^0"};

    // Hyperparameters
    long            n_estimators = 10;
    bool            bootstrap = false;
    bool            oob_score = false;
    string          class_balance  = "balanced";
    long            max_depth = 3;
    long            max_features = 1;
    long            max_thresholds = 1;
    string          missing_values = "None";
    // Random Number Generator
    long            random_state = 0;

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

        DecisionForestClassifier dfc(classes, features,
                                     n_estimators, bootstrap, oob_score,
                                     class_balance, max_depth,
                                     max_features, max_thresholds,
                                     missing_values,
                                     random_state);

        dfc.fit(X, y);

        SECTION("Training") {
            // cout << dfc.export_text(0) << endl;
            REQUIRE(dfc.export_text(0).compare(
                    "0 X[0]<=0.873756 [1, 1, 1, 1, 1, 1, 1, 1]; 0->1; 0->8; 1 X[2]<=0.753259 [1, 1, 1, 1, 0, 0, 0, 0]; 1->2; 1->5; 2 X[1]<=0.36797 [1, 0, 1, 0, 0, 0, 0, 0]; 2->3; 2->4; 3 [1, 0, 0, 0, 0, 0, 0, 0]; 4 [0, 0, 1, 0, 0, 0, 0, 0]; 5 X[1]<=0.309585 [0, 1, 0, 1, 0, 0, 0, 0]; 5->6; 5->7; 6 [0, 1, 0, 0, 0, 0, 0, 0]; 7 [0, 0, 0, 1, 0, 0, 0, 0]; 8 X[1]<=0.24681 [0, 0, 0, 0, 1, 1, 1, 1]; 8->9; 8->12; 9 X[2]<=0.316374 [0, 0, 0, 0, 1, 1, 0, 0]; 9->10; 9->11; 10 [0, 0, 0, 0, 1, 0, 0, 0]; 11 [0, 0, 0, 0, 0, 1, 0, 0]; 12 X[2]<=0.515696 [0, 0, 0, 0, 0, 0, 1, 1]; 12->13; 12->14; 13 [0, 0, 0, 0, 0, 0, 1, 0]; 14 [0, 0, 0, 0, 0, 0, 0, 1]; ") == 0);
        }

        SECTION("Feature Importances") {
            vector<double> importances(features.size());
            dfc.calculate_feature_importances(&importances[0]);
            // for (auto i: importances) cout << i << ' ';
            REQUIRE(compareVectors(importances, {0.328571, 0.328571, 0.342857}));
        }

        // Visualize Trees
        std::string filename = "simple_example_dfc";
        // dfc.export_graphviz(filename, true);
        std::string filename2 = "simple_example_dfc_dt";
        dfc.export_graphviz(filename2, 0, true);
        // $: xdot simple_example_0.gv
        // $: dot -Tpdf simple_example_0.gv -o simple_example_0.pdf

        dfc.export_serialize(filename);
        DecisionForestClassifier dfc2 = DecisionForestClassifier::import_deserialize(filename);

        SECTION("Persistence") {
            // cout << dfc.export_text(0) << endl;
            REQUIRE(dfc2.export_text(0).compare(dfc.export_text(0)) == 0);
        }

        unsigned long   n_samples = y.size();
        
        SECTION("Classification") {
            vector<long> c(n_samples, 0);
            dfc2.predict(&X[0], n_samples, &c[0]);
            // for (auto i: c) cout << i << ' ';
            REQUIRE(compareVectors(c, y));
        }

        SECTION("Testing") {
            double score = dfc2.score(&X[0], &y[0], n_samples);
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
    long            n_estimators = 10;
    bool            bootstrap = false;
    bool            oob_score = false;
    string          class_balance  = "balanced";
    long            max_depth = 3;
    long            max_features = 1;
    long            max_thresholds = 1;
    string          missing_values = "None";
    // Random Number Generator
    long            random_state = 0;

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

        DecisionForestClassifier dfc(classes, features,
                                     n_estimators, bootstrap, oob_score,
                                     class_balance, max_depth,
                                     max_features, max_thresholds,
                                     missing_values,
                                     random_state);

        dfc.fit(X, y);

        SECTION("Training") {
            // cout << dfc.export_text(0) << endl;
            REQUIRE(dfc.export_text(0).compare(
                    "0 X[0]<=0.873756 [1, 1, 1, 1, 1, 1, 1, 1][4, 4][4, 4][4, 4]; 0->1; 0->8; 1 X[2]<=0.753259 [1, 1, 1, 1, 0, 0, 0, 0][4, 0][2, 2][2, 2]; 1->2; 1->5; 2 X[1]<=0.36797 [1, 0, 1, 0, 0, 0, 0, 0][2, 0][1, 1][2, 0]; 2->3; 2->4; 3 [1, 0, 0, 0, 0, 0, 0, 0][1, 0][1, 0][1, 0]; 4 [0, 0, 1, 0, 0, 0, 0, 0][1, 0][0, 1][1, 0]; 5 X[1]<=0.309585 [0, 1, 0, 1, 0, 0, 0, 0][2, 0][1, 1][0, 2]; 5->6; 5->7; 6 [0, 1, 0, 0, 0, 0, 0, 0][1, 0][1, 0][0, 1]; 7 [0, 0, 0, 1, 0, 0, 0, 0][1, 0][0, 1][0, 1]; 8 X[1]<=0.24681 [0, 0, 0, 0, 1, 1, 1, 1][0, 4][2, 2][2, 2]; 8->9; 8->12; 9 X[2]<=0.316374 [0, 0, 0, 0, 1, 1, 0, 0][0, 2][2, 0][1, 1]; 9->10; 9->11; 10 [0, 0, 0, 0, 1, 0, 0, 0][0, 1][1, 0][1, 0]; 11 [0, 0, 0, 0, 0, 1, 0, 0][0, 1][1, 0][0, 1]; 12 X[2]<=0.515696 [0, 0, 0, 0, 0, 0, 1, 1][0, 2][0, 2][1, 1]; 12->13; 12->14; 13 [0, 0, 0, 0, 0, 0, 1, 0][0, 1][0, 1][1, 0]; 14 [0, 0, 0, 0, 0, 0, 0, 1][0, 1][0, 1][0, 1]; ") == 0);
        }

        SECTION("Feature Importances") {
            vector<double> importances(features.size());
            dfc.calculate_feature_importances(&importances[0]);
            // for (auto i: importances) cout << i << ' ';
            REQUIRE(compareVectors(importances, {0.331579, 0.331579, 0.336842}));
        }

        // Visualize Trees
        std::string filename = "simple_example_multi_output_dfc";
        //dfc.export_graphviz(filename, true);
        std::string filename2 = "simple_example_multi_output_dfc_dt";
        dfc.export_graphviz(filename2, 0, true);
        // $: xdot simple_example_0.gv
        // $: dot -Tpdf simple_example_0.gv -o simple_example_0.pdf

        dfc.export_serialize(filename);
        DecisionForestClassifier dfc2 = DecisionForestClassifier::import_deserialize(filename);

        SECTION("Persistence") {
            //cout << dfc.export_text(0) << endl;
            REQUIRE(dfc2.export_text(0).compare(dfc.export_text(0)) == 0);
        }

        unsigned long   n_outputs = classes.size();
        unsigned long   n_samples = y.size() / n_outputs;

        SECTION("Classification") {
            vector<long> c(n_samples*n_outputs, 0);
            dfc2.predict(&X[0], n_samples, &c[0]);
            //for (unsigned long i = 0; i < n_samples; ++i) {
            //    for (unsigned long o = 0; o < n_outputs; ++o) {
            //        cout << c[i * n_outputs + o] << ' ';
            //     }
            //    cout << endl;
            //}
            REQUIRE(compareVectors(c, y));
        }

        SECTION("Testing") {
            double score = dfc2.score(&X[0], &y[0], n_samples);
            REQUIRE(score == Approx(1.0));
        }

    } catch (exception &e) {
        cout << e.what() << '\n';
    }
}