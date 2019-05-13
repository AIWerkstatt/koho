#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>

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

TEST_CASE("simple example"){

    vector<string>  classes = {"A", "B"};
    long            n_classes = classes.size();
    vector<string>  features = {"a", "b", "c"};
    long            n_features = features.size();

    vector<double>  X = {0, 0, 0,
                         0, 0, 1,
                         0, 1, 0,
                         0, 1, 1,
                         0, 1, 1,
                         1, 0, 0,
                         1, 0, 0,
                         1, 0, 0,
                         1, 0, 0,
                         1, 1, 1};
    vector<long>    y = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    unsigned long   n_samples = y.size();

    // Hyperparameters
    string          class_balance  = "balanced";
    long            max_depth      = 3;
    long            max_features   = 0;
    long            max_thresholds = 0;
    string          missing_values = "None";
    // Random Number Generator
    long            random_state = 0;

    DecisionTreeClassifier dtc(classes, n_classes,
                               features, n_features,
                               class_balance, max_depth,
                               max_features, max_thresholds,
                               missing_values,
                               random_state);

    dtc.fit(&X[0], &y[0], n_samples);

    SECTION("Training") {
        //cout << dtc.export_text() << endl;
        REQUIRE(dtc.export_text().compare("0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.875]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.875]; 4 [0, 3.125]; ") == 0);
    }

    SECTION("Feature Importances") {
        vector<double> importances(n_features);
        dtc.calculate_feature_importances(&importances[0]);
        //for (auto i: importances) cout << i << ' ';
        REQUIRE(compareVectors(importances,{0.454545, 0.545455, 0}));
    }

    // Visualize Trees
    std::string filename = "simple_example_dtc";
    dtc.export_graphviz(filename, true);
    // $: xdot simple_example.gv
    // $: dot -Tpdf simple_example.gv -o simple_example.pdf

    dtc.export_serialize(filename);
    DecisionTreeClassifier dtc2 = DecisionTreeClassifier::import_deserialize(filename);

    SECTION("Persistence") {
        //cout << dtc.export_text() << endl;
        REQUIRE(dtc2.export_text().compare(dtc.export_text()) == 0);
    }

    SECTION("Classification") {
        vector<long> c(n_samples, 0);
        dtc2.predict(&X[0], n_samples, &c[0]);
        //for (auto i: c) cout << i << ' ';
        REQUIRE(compareVectors(c, {0, 0, 1, 1, 1, 1, 1, 1, 1, 1}));
    }

    SECTION("Testing") {
        double score = dtc2.score(&X[0], &y[0], n_samples);
        REQUIRE(score == Approx(1.0));
    }
}


