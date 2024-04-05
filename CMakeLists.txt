cmake_minimum_required(VERSION 3.27)
project(mlir_example)

set(CMAKE_CXX_STANDARD 23)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddMLIR)
include(AddLLVM)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(lib)

add_executable(mlir_example main.cpp)
target_link_libraries(mlir_example MLIRLambda MLIR)