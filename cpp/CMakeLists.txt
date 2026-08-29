cmake_minimum_required(VERSION 3.16)
project(regime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

# Library ---------------------------------------------------------------
add_library(regime
    src/hmm.cpp
    src/strategies.cpp
    src/backtest.cpp
    src/pipeline.cpp
)
target_include_directories(regime PUBLIC include)

# Bundled data directory (compiled in; overridable at runtime by CLI arg).
set(REGIME_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")

# Demo ------------------------------------------------------------------
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE regime)
target_compile_definitions(demo PRIVATE REGIME_DATA_DIR="${REGIME_DATA_DIR}")

# Tests -----------------------------------------------------------------
enable_testing()
find_package(GTest REQUIRED)
add_executable(regime_tests
    tests/test_hmm.cpp
    tests/test_strategies.cpp
    tests/test_backtest.cpp
    tests/test_golden.cpp
)
target_link_libraries(regime_tests PRIVATE regime GTest::gtest GTest::gtest_main)
target_compile_definitions(regime_tests PRIVATE REGIME_DATA_DIR="${REGIME_DATA_DIR}")
add_test(NAME regime_tests COMMAND regime_tests)
