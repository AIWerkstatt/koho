cmake_minimum_required(VERSION 3.10)
project(koho)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_RELEASE "-s -O3 -mavx -msse2 -mfma -mbmi -Wno-unused-variable")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
set(CMAKE_CXX_FLAGS_SANITIZE "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
set(CMAKE_CXX_FLAGS_SANITIZEREL "${CMAKE_CXX_FLAGS_RELEASE} -g -fsanitize=address")

# library

add_library(koho STATIC
            utilities.h
            random_number_generator.h random_number_generator.cpp
            decision_tree.h decision_tree.cpp
            decision_forest.h decision_forest.cpp)

# tests

add_executable(test_decision_tree_classifier tests/test_decision_tree_classifier.cpp )
target_link_libraries(test_decision_tree_classifier koho)
add_executable(test_decision_forest_classifier tests/test_decision_forest_classifier.cpp )
target_link_libraries(test_decision_forest_classifier koho)
