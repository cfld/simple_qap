#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n qap_env python=3.7
conda activate qap_env
pip install scipy==1.6.1

# --
# Get problems

mkdir -p data/qaplib
cd data/qaplib
wget http://coral.ise.lehigh.edu/wp-content/uploads/2014/07/qapdata.tar.gz
tar -xzvf qapdata.tar.gz
rm qapdata.tar.gz
cd ../..

# --
# Build project

git clone https://github.com/pybind/pybind11
rm -rf build
mkdir build
cd build
cmake ..
make -j12
cd ..

# --
# Run test

python test.py