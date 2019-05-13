/// Random Number Generator module.
/** @file
*/

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

#include "random_number_generator.h"

using namespace std;

namespace koho {

// =============================================================================
// Random Number Generator
// =============================================================================

    // Create and initialize random number generator with the current system time

    RandomState::RandomState() : eng(static_cast<unsigned long>(time(nullptr))) {}

    // Create and initialize random number generator with a seed

    RandomState::RandomState(unsigned long  seed) : eng(seed) {}

    // Provide a double random number from a uniform distribution between [low, high).

    double  RandomState::uniform_real(double  low,
                                      double  high) {

        uniform_real_distribution<double> dist(low, high);
        return dist(RandomState::eng);
    }

    // Provide a long random number from a uniform distribution between [low, high).

    long  RandomState::uniform_int(long  low,
                                   long  high) {

        std::uniform_int_distribution<long> dist(low, high - 1);
        return dist(RandomState::eng);
    }

} // namespace koho
