#ifndef PAULI_UTILS_HPP
#define PAULI_UTILS_HPP

#include "../include/sparse_tensor.hpp"
#include <vector>
#include <string>
#include <map>

namespace hamiltonian_learning {

// Pauli string operations
std::string compressPauli(const std::string& pauli);
std::string decompressPauli(const std::string& pauli_compressed, int n);
std::vector<std::string> compressPauliToList(const std::string& pauli);

// Parse human-readable operator notation (e.g., "Z_11 Z_15 X_14" or "Z11 Z15 X14")
// Returns full Pauli string of length n (e.g., "IIIIIIIIIIZIIIXZ")
// Site indices are 0-indexed
std::string parseHumanReadablePauli(const std::string& op_str, int n);

std::pair<std::string, Complex> multiplyPaulis(const std::string& pauli1,
                                               const std::string& pauli2);
bool checkCommute(const std::string& pauli1, const std::string& pauli2);
bool checkOverlap(const std::string& pauli1, const std::string& pauli2);
std::vector<bool> determineSupport(const std::string& pauli_string);
int weight(const std::string& pauli_string);

// Build k-local Paulis
std::vector<std::string> buildKLocalPaulis1D(int n, int k, bool periodic_bc);
std::vector<std::string> buildKLocalCorrelators1D(int n, int k, bool periodic_bc, int d_max = -1);
std::string embedPauli(int n, const std::string& p, const std::vector<int>& region);
std::vector<std::string> restrictOperators(int n,
                                          const std::vector<std::string>& operators,
                                          const std::vector<int>& region);

// Note: complexToReal() and realToComplex() are now in sparse_tensor.hpp
// They are general linear algebra utilities, not Pauli-specific

// Matrix operations
VectorXd vectorizeLowerTriangular(const MatrixXd& a, bool strict = true,
                                  double scale_off_diagonals = 0.0);

// Pauli matrices
MatrixXc pauliMatrix(const std::string& pauli_string);

// Three-body operations
std::pair<SparseTensor, std::vector<std::string>>
buildMultiplicationTensor(const std::vector<std::string>& onebody_operators);

std::pair<SparseTensor, std::vector<std::string>>
buildMultiplicationTensorSingleThreaded(const std::vector<std::string>& onebody_operators);

std::pair<SparseTensor, std::vector<std::string>>
buildTripleProductTensor(const std::vector<std::string>& onebody_operators,
                        const std::vector<std::string>& hamiltonian_terms);

// Utility functions
template<typename T>
std::map<T, int> invertStringList(const std::vector<T>& list);

// Note: tprint/tprintf are declared in hamiltonian_learning.hpp
// Note: createSaveDirectory() commented out - not used
// std::string createSaveDirectory();

} // namespace hamiltonian_learning

#endif // PAULI_UTILS_HPP
