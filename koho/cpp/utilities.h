/// Utilities module.
/** @file

C++ implementation.
*/

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

#ifndef KOHO_UTILITIES_H
#define KOHO_UTILITIES_H

#include <string>
#include <vector>
#include <utility> // pair
#include <algorithm> // sort

using namespace std;

namespace koho {

    // Provide the index of the maximum in an array.

    template <class X>
    unsigned long  maxIndex(X*             x,
                            unsigned long  n) {

        unsigned long  max_index = 0;
        X max_value = x[max_index];

        for (unsigned long i=1; i<n; ++i) {
            if (max_value < x[i]) {
                max_index = i;
                max_value = x[max_index];
            }
        }

        return max_index;
    }

    // Sort 2 vectors by the first vector.

    template <class X, class S>
    void  sort2VectorsByFirstVector(std::vector<X>& x, std::vector<S>& s,
                                    long start, long end,
                                    bool increase=true) {

        // Combine vector x and vector s into vector pair<x,s>
        std::vector<std::pair<X, S>> pxs(x.size());
        auto x_itr = x.begin();
        auto s_itr = s.begin();
        for (auto &p : pxs) {
            p.first = *(x_itr++);
            p.second = *(s_itr++);
        }

        // Sort vector pair<x,s> by x
        if (increase) {
            sort(pxs.begin()+start, pxs.end()-(pxs.size()-end),
                 [](const std::pair<X, S> &a, const std::pair<X, S> &b) -> bool { return a.first < b.first; });
        } else { // decrease
            sort(pxs.begin()+start, pxs.end()-(pxs.size()-end),
                 [](const std::pair<X, S> &a, const std::pair<X, S> &b) -> bool { return a.first > b.first; });
        }

        // Copy sorted vector pair<x,s> back into vector x and vector s
        x_itr = x.begin();
        s_itr = s.begin();
        for (auto& p : pxs) {
            *(x_itr++) = p.first;
            *(s_itr++) = p.second;
        }
    }

} // namespace koho

#endif
