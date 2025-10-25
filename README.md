# Hamiltonian Learning (C++ Port)

C++ implementation of the certified Hamiltonian learning algorithm from the paper _Efficient Hamiltonian reconstruction from equilibrium states_ (https://arxiv.org/abs/2403.18061).

This is a port of the Python implementation focusing on the `certifiedBoundsV2` function, which computes certified bounds on linear functionals of Hamiltonian parameters.

## Features

- **Certified Bounds Algorithm**: Implements `certifiedBoundsV2` for computing rigorous upper and lower bounds on Hamiltonian coefficients
- **Sparse Tensor Operations**: Custom `SparseTensor` class for efficient manipulation of high-dimensional sparse data
- **Pauli String Utilities**: Tools for working with Pauli operator strings
- **MOSEK Integration**: Uses MOSEK Fusion API for semidefinite programming

## Dependencies

### Required

1. **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2017+)

2. **Eigen3** (version 3.3+)
   - Linear algebra library
   - Installation:
     ```bash
     # Ubuntu/Debian
     sudo apt-get install libeigen3-dev

     # macOS
     brew install eigen

     # Or download from: https://eigen.tuxfamily.org/
     ```

3. **MOSEK** (version 10.0+)
   - Commercial optimization solver (free academic license available)
   - Download from: https://www.mosek.com/downloads/
   - You'll need both:
     - MOSEK Optimizer API
     - MOSEK Fusion API for C++
   - Installation:
     ```bash
     # Download and extract MOSEK
     tar -xvzf mosektoolslinux64x86.tar.bz2

     # Set environment variable (add to ~/.bashrc)
     export MOSEK_ROOT=$HOME/mosek/10.2/tools/platform/linux64x86

     # License: Place mosek.lic in $HOME/mosek/
     ```

4. **CMake** (version 3.15+)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake

   # macOS
   brew install cmake
   ```

## Building

### Basic Build

```bash
cd hamiltonian-learning-C
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Or with make
make -j4
```

### Custom MOSEK Installation

If MOSEK is installed in a non-standard location:

```bash
cmake -DMOSEK_ROOT=/path/to/mosek/platform/dir ..
```

For example:
```bash
cmake -DMOSEK_ROOT=$HOME/mosek/10.2/tools/platform/osx64x86 ..
```

### Build Options

```bash
# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Usage

### Example Program

Run the example after building:

```bash
./example_bounds
```

### Using in Your Code

```cpp
#include "hamiltonian_learning.hpp"

using namespace hamiltonian_learning;

int main() {
    // Set up problem parameters
    int r = 10;  // Number of perturbing operators
    int s = 8;   // Number of Hamiltonian terms

    // Expectation values of Hamiltonian terms
    VectorXd h_terms_exp(s);
    // ... fill with your data

    // J matrix: b_i* = sum_j J_ij b_j
    MatrixXc J(r, r);
    // ... fill with your data

    // Covariance matrix: C_ij = omega(b_i* b_j)
    MatrixXc C(r, r);
    // ... fill with your data

    // Sparse tensor F: F_ijk = omega(b_i* [h_k, b_j])
    std::vector<std::vector<int>> F_indices;
    std::vector<Complex> F_values;
    // ... fill with your data

    // Algorithm parameters
    double epsilon_W = 1e-6;   // W eigenvalue threshold
    double epsilon_0 = 0.01;   // Measurement error
    double beta = 1.0;         // Inverse temperature

    // Direction vector for bounds
    VectorXd v = VectorXd::Zero(s);
    v(0) = 1.0;  // Compute bounds on first coefficient

    // Run algorithm
    CertifiedBoundsResult result = certifiedBoundsV2(
        r, s, h_terms_exp, J, C,
        F_indices, F_values,
        epsilon_W, epsilon_0, beta, v,
        2  // printing_level
    );

    // Access results
    std::cout << "Lower bound: " << result.lower_bound << std::endl;
    std::cout << "Upper bound: " << result.upper_bound << std::endl;

    return 0;
}
```

### Linking Against the Library

In your CMakeLists.txt:

```cmake
find_package(Eigen3 REQUIRED)

add_executable(my_program my_program.cpp)

target_link_libraries(my_program
    PRIVATE
        hamiltonian_learning
        Eigen3::Eigen
        mosek64
        fusion64
)

target_include_directories(my_program PRIVATE
    /path/to/hamiltonian-learning-C/include
    ${MOSEK_INCLUDE_DIR}
)
```

## API Reference

### `certifiedBoundsV2`

Computes certified lower and upper bounds on a linear functional `v^T h` of the Hamiltonian coefficients.

**Parameters:**
- `r` (int): Number of perturbing operators
- `s` (int): Number of variational Hamiltonian terms
- `h_terms_exp` (VectorXd): Expectations ω(h_α) for α = 1,...,s
- `J` (MatrixXc): r×r matrix satisfying b_i* = Σ_j J_ij b_j
- `C` (MatrixXc): r×r covariance matrix C_ij = ω(b_i* b_j)
- `F_indices` (vector<vector<int>>): COO sparse indices for F tensor
- `F_values` (vector<Complex>): COO sparse values for F_ijk = ω(b_i* [h_k, b_j])
- `epsilon_W` (double): Eigenvalue threshold for W kernel computation
- `epsilon_0` (double): Measurement error bound
- `beta` (double): Inverse temperature
- `v` (VectorXd): Direction vector (length s) for the linear functional
- `printing_level` (int): Verbosity (0=quiet, 1=normal, 2=verbose)

**Returns:** `CertifiedBoundsResult` with fields:
- `lower_bound`: Lower bound on v^T h
- `upper_bound`: Upper bound on v^T h
- `mu_1`, `mu_2`: Regularization parameters
- `kappa`, `nu`: Conditioning parameters

## Differences from Python Version

1. **Type Safety**: Explicit types instead of dynamic typing
2. **Memory Management**: Manual resource management (RAII patterns used)
3. **Return Values**: Structs instead of tuples for multiple returns
4. **Error Handling**: Exceptions instead of Python's try/except
5. **Not Ported**: The full simulation.py (TenPy-based state generation) is not included - this port focuses on the core certified bounds algorithm

## Project Structure

```
hamiltonian-learning-C/
├── include/
│   ├── utils.hpp                    # Utility functions and SparseTensor
│   └── hamiltonian_learning.hpp     # Main algorithm interface
├── src/
│   ├── utils.cpp                    # Utility implementations
│   └── hamiltonian_learning.cpp     # certifiedBoundsV2 implementation
├── examples/
│   └── example_certified_bounds.cpp # Example usage
├── CMakeLists.txt                   # Build configuration
└── README.md                        # This file
```

## Troubleshooting

### MOSEK Not Found

If CMake can't find MOSEK:
```bash
cmake -DMOSEK_ROOT=/path/to/mosek/installation ..
```

Check that `$MOSEK_ROOT/h` contains `fusion.h` and `$MOSEK_ROOT/bin` contains `libmosek64.so` (Linux) or `libmosek64.dylib` (macOS).

### Eigen Not Found

If CMake can't find Eigen:
```bash
cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 ..
```

### Runtime Library Errors

Add MOSEK library path to your environment:
```bash
# Linux
export LD_LIBRARY_PATH=$MOSEK_ROOT/bin:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=$MOSEK_ROOT/bin:$DYLD_LIBRARY_PATH
```

## License

This implementation follows the same licensing as the original Python version.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{haah2024efficient,
  title={Efficient Hamiltonian reconstruction from equilibrium states},
  author={Your Authors Here},
  journal={arXiv preprint arXiv:2403.18061},
  year={2024}
}
```

## Contributing

This is a research code port. For contributions or questions, please refer to the original Python implementation.
