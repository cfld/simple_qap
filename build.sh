#!/bin/bash

# build.sh

# git clone https://github.com/pybind/pybind11

# --
# Build project

rm -rf build
mkdir build
cd build
cmake ..
make -j12
cd ..
