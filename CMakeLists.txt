cmake_minimum_required(VERSION 3.15)
project(HamiltonianLearning VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
    # Suppress warnings from external libraries (MOSEK, iTensor)
    add_compile_options(-Wno-unused-parameter -Wno-sign-compare -Wno-unused-function -Wno-ignored-qualifiers)
endif()

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# MOSEK paths - users may need to customize these
set(MOSEK_ROOT "$ENV{HOME}/mosek/10.2/tools/platform/linux64x86" CACHE PATH "MOSEK installation directory")
set(MOSEK_INCLUDE_DIR "${MOSEK_ROOT}/h" CACHE PATH "MOSEK include directory")
set(MOSEK_LIB_DIR "${MOSEK_ROOT}/bin" CACHE PATH "MOSEK library directory")
set(MOSEK_FUSION_SRC "${MOSEK_ROOT}/src/fusion_cxx" CACHE PATH "MOSEK Fusion C++ source directory")

# Check if MOSEK paths exist
if(NOT EXISTS ${MOSEK_INCLUDE_DIR})
    message(WARNING "MOSEK include directory not found at ${MOSEK_INCLUDE_DIR}. Please set MOSEK_ROOT correctly.")
endif()

# Include external libraries as SYSTEM to suppress their warnings
include_directories(SYSTEM
    ${MOSEK_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(
    ${MOSEK_LIB_DIR}
)

# ============================================================================
# Core Libraries
# ============================================================================

# SparseTensor library (minimal dependencies)
add_library(sparse_tensor STATIC
    src/sparse_tensor.cpp
)
target_include_directories(sparse_tensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(sparse_tensor
    PUBLIC Eigen3::Eigen
    PRIVATE OpenMP::OpenMP_CXX  # Only for contractRight parallelization
)

# Hamiltonian learning library (core algorithm)
add_library(hamiltonian_learning STATIC
    src/hamiltonian_learning.cpp
)
target_include_directories(hamiltonian_learning PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(hamiltonian_learning
    PUBLIC
        sparse_tensor
        Eigen3::Eigen
    PRIVATE
        mosek64
        fusion64
        lapack
        blas
)

# ============================================================================
# Installation Rules
# ============================================================================

install(TARGETS sparse_tensor hamiltonian_learning
    EXPORT HamiltonianLearningTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets
install(EXPORT HamiltonianLearningTargets
    FILE HamiltonianLearningTargets.cmake
    NAMESPACE HamiltonianLearning::
    DESTINATION lib/cmake/HamiltonianLearning
)

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "HamiltonianLearning Configuration Summary:")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3 Include:      ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  MOSEK Root:          ${MOSEK_ROOT}")
message(STATUS "  MOSEK Include:       ${MOSEK_INCLUDE_DIR}")
message(STATUS "  MOSEK Lib:           ${MOSEK_LIB_DIR}")
message(STATUS "")
message(STATUS "Core Libraries:")
message(STATUS "  - sparse_tensor (SparseTensor class)")
message(STATUS "  - hamiltonian_learning (certified bounds algorithm)")
message(STATUS "")
