cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkExperiment VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

file(GLOB_RECURSE SOURCES
  CONFIGURE_DEPENDS
  src/*.cpp
)

include_directories("${PROJECT_BINARY_DIR}")

add_executable(NeuralNetworkExperiment ${SOURCES})
