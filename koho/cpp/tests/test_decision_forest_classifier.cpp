#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>

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
    long            n_estimators = 10;
    bool            bootstrap = false;
    bool            oob_score = false;
    string          class_balance  = "balanced";
    long            max_depth = 3;
    long            max_features = 2;
    long            max_thresholds = 0;
    string          missing_values = "None";
    // Random Number Generator
    long            random_state = 0;
    DecisionForestClassifier dfc(classes, n_classes,
                                 features, n_features,
                                 n_estimators, bootstrap, oob_score,
                                 class_balance, max_depth,
                                 max_features, max_thresholds,
                                 missing_values,
                                 random_state);

    dfc.fit(&X[0], &y[0], n_samples);

    SECTION("Training") {
        //cout << dfc.export_text(0) << endl;
        REQUIRE(dfc.export_text(0).compare("0 X[0]<=0.5 [5, 5]; 0->1; 0->4; 1 X[1]<=0.5 [5, 1.875]; 1->2; 1->3; 2 [5, 0]; 3 [0, 1.875]; 4 [0, 3.125]; ") == 0);
    }

    SECTION("Feature Importances") {
        vector<double> importances(n_features);
        dfc.calculate_feature_importances(&importances[0]);
        //for (auto i: importances) cout << i << ' ';
        REQUIRE(compareVectors(importances,{0.484848, 0.479394, 0.0357576}));
    }

    // Visualize Trees
    std::string filename = "simple_example_dfc";
    dfc.export_graphviz(filename, true);
    std::string filename2 = "simple_example_dfc_dt";
    dfc.export_graphviz(filename2, 0, true);
    // $: xdot simple_example_0.gv
    // $: dot -Tpdf simple_example_0.gv -o simple_example_0.pdf

    dfc.export_serialize(filename);
    DecisionForestClassifier dfc2 = DecisionForestClassifier::import_deserialize(filename);

    SECTION("Persistence") {
        //cout << dtc.export_text() << endl;
        REQUIRE(dfc2.export_text(0).compare(dfc.export_text(0)) == 0);
    }

    SECTION("Classification") {
        vector<long> c(n_samples, 0);
        dfc2.predict(&X[0], n_samples, &c[0]);
        //for (auto i: c) cout << i << ' ';
        REQUIRE(compareVectors(c, {0, 0, 1, 1, 1, 1, 1, 1, 1, 1}));
    }

    SECTION("Testing") {
        double score = dfc2.score(&X[0], &y[0], n_samples);
        REQUIRE(score == Approx(1.0));
    }
}


