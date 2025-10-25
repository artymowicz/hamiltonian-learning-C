cmake_minimum_required(VERSION 3.15)
project(HamiltonianLearningBenchmarks VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
    # Suppress warnings from external libraries (iTensor)
    add_compile_options(-Wno-unused-parameter -Wno-sign-compare -Wno-unused-function -Wno-ignored-qualifiers)
endif()

# ============================================================================
# Find Dependencies
# ============================================================================

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

# ITensor paths
set(ITENSOR_DIR "$ENV{HOME}/itensor" CACHE PATH "ITensor installation directory")
set(ITENSOR_INCLUDE_DIR "${ITENSOR_DIR}" CACHE PATH "ITensor include directory")
set(ITENSOR_LIB_DIR "${ITENSOR_DIR}/lib" CACHE PATH "ITensor library directory")

# MOSEK paths (required by hamiltonian_learning headers)
set(MOSEK_ROOT "$ENV{HOME}/mosek/10.2/tools/platform/linux64x86" CACHE PATH "MOSEK installation directory")
set(MOSEK_INCLUDE_DIR "${MOSEK_ROOT}/h" CACHE PATH "MOSEK include directory")
set(MOSEK_LIB_DIR "${MOSEK_ROOT}/bin" CACHE PATH "MOSEK library directory")

# Include external libraries as SYSTEM to suppress their warnings
include_directories(SYSTEM
    ${EIGEN3_INCLUDE_DIR}
    ${ITENSOR_INCLUDE_DIR}
    ${MOSEK_INCLUDE_DIR}
)

link_directories(
    ${ITENSOR_LIB_DIR}
    ${MOSEK_LIB_DIR}
)

# ============================================================================
# Link to Core Libraries
# ============================================================================

# Link to parent build directory
set(PARENT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build" CACHE PATH "Parent project build directory")

# Add parent include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Find parent libraries
find_library(SPARSE_TENSOR_LIB sparse_tensor PATHS ${PARENT_BUILD_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(HAMILTONIAN_LEARNING_LIB hamiltonian_learning PATHS ${PARENT_BUILD_DIR} NO_DEFAULT_PATH REQUIRED)

# Create imported targets
add_library(sparse_tensor STATIC IMPORTED)
set_target_properties(sparse_tensor PROPERTIES
    IMPORTED_LOCATION ${SPARSE_TENSOR_LIB}
    INTERFACE_LINK_LIBRARIES "Eigen3::Eigen;OpenMP::OpenMP_CXX"
)

add_library(hamiltonian_learning STATIC IMPORTED)
set_target_properties(hamiltonian_learning PROPERTIES
    IMPORTED_LOCATION ${HAMILTONIAN_LEARNING_LIB}
    INTERFACE_LINK_LIBRARIES "sparse_tensor;Eigen3::Eigen;mosek64;fusion64;lapack;blas"
)

message(STATUS "Linking to core libraries from: ${PARENT_BUILD_DIR}")

# ============================================================================
# Benchmark Libraries
# ============================================================================

# Pauli utilities library
add_library(pauli_utils STATIC
    src/pauli_utils.cpp
)
target_include_directories(pauli_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pauli_utils
    PUBLIC
        sparse_tensor
        OpenMP::OpenMP_CXX
)

# Simulation library (1D MPS-based thermal states)
add_library(simulation STATIC
    src/simulation.cpp
)
target_include_directories(simulation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(simulation
    PUBLIC
        pauli_utils
        itensor
        yaml-cpp
        OpenMP::OpenMP_CXX
        lapack
        blas
)

# ============================================================================
# Benchmark Executables
# ============================================================================

add_executable(1d_certified_bounds_benchmark 1d_certified_bounds_benchmark.cpp)
target_link_libraries(1d_certified_bounds_benchmark
    PRIVATE
        simulation
        hamiltonian_learning
)

add_executable(error_estimate error_estimate.cpp)
target_link_libraries(error_estimate
    PRIVATE
        simulation
)

add_executable(optimize_parameters optimize_parameters.cpp)
target_link_libraries(optimize_parameters
    PRIVATE
        simulation
)

# ============================================================================
# Test Executables
# ============================================================================

add_executable(test_pauli_utils tests/test_pauli_utils.cpp)
target_link_libraries(test_pauli_utils
    PRIVATE
        pauli_utils
)

add_executable(test_simulation tests/test_simulation.cpp)
target_link_libraries(test_simulation
    PRIVATE
        simulation
)

add_executable(test_trotter_error tests/test_trotter_error.cpp)
target_link_libraries(test_trotter_error
    PRIVATE
        simulation
)

# ============================================================================
# Copy Setup and Hamiltonian Files to Build Directory
# ============================================================================

# Copy setup directory to build directory for easy access
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/setup
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Copy hamiltonians directory to build directory for easy access
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/hamiltonians
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "Copied setup/ and hamiltonians/ to build directory")

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "Hamiltonian Learning Benchmarks Configuration:")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3 Include:      ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  ITensor Dir:         ${ITENSOR_DIR}")
message(STATUS "  Core Libraries:      ${PARENT_BUILD_DIR}")
message(STATUS "")
