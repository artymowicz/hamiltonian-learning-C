#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP

#include <vector>
#include <string>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace hamiltonian_learning {

using Complex = std::complex<double>;
using VectorXc = Eigen::VectorXcd;
using MatrixXc = Eigen::MatrixXcd;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<Complex>;

// Sparse Tensor class
class SparseTensor {
public:
    std::vector<int> shape;
    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;

    // Constructors
    SparseTensor(const std::vector<int>& shape,
                 const std::vector<std::vector<int>>& indices,
                 const std::vector<Complex>& values,
                 bool order = false);

    // Initialize from dense array (for testing)
    template<typename Derived>
    SparseTensor(const Eigen::DenseBase<Derived>& array);

    // Operators
    SparseTensor operator+(const SparseTensor& other) const;
    SparseTensor operator-(const SparseTensor& other) const;

    // Methods
    void consistencyCheck() const;
    void order(bool little_endian = false);
    void addUpRedundantEntries();
    void removeZeros();
    SparseTensor conjugate() const;
    SparseTensor copy() const;
    SparseTensor transpose(const std::vector<int>& permutation) const;

    // Complex to real conversion for specific axes
    SparseTensor complexToReal(const std::vector<int>& axes) const;

    // Vectorization operations
    SparseTensor vectorize(const std::vector<int>& axes, bool order = true) const;
    SparseTensor vectorizeLowerTriangular(const std::vector<int>& axes,
                                         bool strict = true,
                                         bool order = true,
                                         double scale_off_diagonals = 0.0) const;

    // Contraction operations
    SparseTensor contractRight(const VectorXc& v, bool order = true) const;
    MatrixXd contractRight(const MatrixXd& M) const;  // Sparse matrix × dense matrix (rank-2 only, parallelized)
    MatrixXd contractRightSingleThreaded(const MatrixXd& M) const;  // Single-threaded version for testing
    SparseTensor contractLeft(const VectorXc& v, bool order = true) const;

    // Conversion methods
    SparseMatrix toEigenSparse() const;  // For 2D tensors
    Eigen::MatrixXcd toEigenDense() const;

    std::string toString() const;
    bool isEqual(const SparseTensor& other) const;

private:
    void orderIndices(bool little_endian = false);
};

// Matrix utility functions
MatrixXd complexToReal(const MatrixXc& M);
MatrixXc realToComplex(const MatrixXd& M, bool sanity_check = false);

} // namespace hamiltonian_learning

#endif // SPARSE_TENSOR_HPP
