cmake_minimum_required(VERSION 3.6)
project(simple_qap)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(OpenMP)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -mtune=native -march=native -Wall -Wunused-result -O3 -DNDEBUG -g -ffast-math")

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./pybind11/include
)

add_library(simple_qap SHARED simple_qap.cpp)

target_link_libraries(simple_qap ${PYTHON_LIBRARIES} OpenMP::OpenMP_CXX)
set_target_properties(simple_qap PROPERTIES PREFIX "")