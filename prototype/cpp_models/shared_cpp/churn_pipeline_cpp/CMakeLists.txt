cmake_minimum_required(VERSION 3.14)
project(churn_cascade_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/churn_pipeline_cpp/include
)

# Define the Python extension module
pybind11_add_module(churn_cascade_cpp
    churn_pipeline_cpp/bindings/python_bindings.cpp
    churn_pipeline_cpp/src/churn_cascade.cpp
)