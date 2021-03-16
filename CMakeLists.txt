cmake_minimum_required(VERSION 3.6)
project(simple_qap)

find_package(OpenMP)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./pybind11/include
)

add_library(simple_qap SHARED simple_qap.cpp)

target_link_libraries(simple_qap ${PYTHON_LIBRARIES} OpenMP::OpenMP_CXX)
set_target_properties(simple_qap PROPERTIES PREFIX "")