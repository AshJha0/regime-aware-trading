#!/bin/sh
# Build the regime C++ library, demo, and tests (Release): bash build.sh from cpp/.
# Requires CMake >= 3.20, a C++17 compiler and GoogleTest (find_package(GTest)).
set -e
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
