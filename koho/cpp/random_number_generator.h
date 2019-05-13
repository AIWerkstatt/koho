/// Random Number Generator module.
/** @file
*/

// Author: AI Werkstatt (TM)
// (C) Copyright 2019, AI Werkstatt (TM) www.aiwerkstatt.com. All rights reserved.

#ifndef KOHO_RANDOM_NUMBER_GENERATOR_H
#define KOHO_RANDOM_NUMBER_GENERATOR_H

#include <random>

namespace koho {

// =============================================================================
// Random Number Generator
// =============================================================================

    /// A random number generator.
    class RandomState {

    protected:
        /// Mersenne twister
        std::mt19937  eng;

    public:
        /// Create and initialize random number generator with the current system time
        RandomState();

        /// Create and initialize random number generator with a seed
        /**
        @param[in] seed
        */
        RandomState(unsigned long  seed);

        /// Provide a double random number from a uniform distribution between [low, high).
        /**
        @param[in] low      included lower bound for random number.
        @param[in] high     excluded upper bound for random number.
        @return             random number.
        */
        double  uniform_real(double  low,
                             double  high);

        /// Provide a long random number from a uniform distribution between [low, high).
        /**
        @param[in] low      included lower bound for random number.
        @param[in] high     excluded upper bound for random number.
        @return             random number.
        */
        long  uniform_int(long  low,
                          long  high);

        /// Upper bound for long random numbers [..., high).
        static const long  MAX_INT = std::numeric_limits<long>::max();
    };

} // namespace koho

#endif
