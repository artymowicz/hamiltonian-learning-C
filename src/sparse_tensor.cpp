#include "sparse_tensor.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <omp.h>

namespace hamiltonian_learning {

// SparseTensor implementation
SparseTensor::SparseTensor(const std::vector<int>& shape_,
                           const std::vector<std::vector<int>>& indices_,
                           const std::vector<Complex>& values_,
                           bool do_order)
    : shape(shape_), indices(indices_), values(values_) {
    if (do_order) {
        order();
    }
}

void SparseTensor::order(bool little_endian) {
    std::vector<int> sort_indices(values.size());
    for (size_t i = 0; i < sort_indices.size(); ++i) {
        sort_indices[i] = i;
    }

    if (little_endian) {
        std::sort(sort_indices.begin(), sort_indices.end(),
                 [this](int i, int j) {
                     auto ri = indices[i];
                     auto rj = indices[j];
                     std::reverse(ri.begin(), ri.end());
                     std::reverse(rj.begin(), rj.end());
                     return ri < rj;
                 });
    } else {
        std::sort(sort_indices.begin(), sort_indices.end(),
                 [this](int i, int j) {
                     return indices[i] < indices[j];
                 });
    }

    std::vector<std::vector<int>> new_indices;
    std::vector<Complex> new_values;
    for (int idx : sort_indices) {
        new_indices.push_back(indices[idx]);
        new_values.push_back(values[idx]);
    }
    indices = new_indices;
    values = new_values;
}

void SparseTensor::addUpRedundantEntries() {
    if (indices.empty()) return;

    std::vector<std::vector<int>> out_indices = {indices[0]};
    std::vector<Complex> out_values;
    Complex current_total = 0.0;

    for (size_t i = 0; i < values.size(); ++i) {
        if (indices[i] == out_indices.back()) {
            current_total += values[i];
        } else {
            out_values.push_back(current_total);
            current_total = 0.0;
            out_indices.push_back(indices[i]);
            current_total += values[i];
        }
    }
    out_values.push_back(current_total);

    indices = out_indices;
    values = out_values;
}

void SparseTensor::removeZeros() {
    std::vector<std::vector<int>> new_indices;
    std::vector<Complex> new_values;

    for (size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i]) > 1e-15) {
            new_indices.push_back(indices[i]);
            new_values.push_back(values[i]);
        }
    }

    indices = new_indices;
    values = new_values;
}

SparseTensor SparseTensor::conjugate() const {
    std::vector<Complex> conj_values;
    for (const auto& v : values) {
        conj_values.push_back(std::conj(v));
    }
    return SparseTensor(shape, indices, conj_values, false);
}

SparseTensor SparseTensor::operator+(const SparseTensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("SparseTensor shapes don't match for addition");
    }

    std::vector<std::vector<int>> new_indices = indices;
    new_indices.insert(new_indices.end(), other.indices.begin(), other.indices.end());

    std::vector<Complex> new_values = values;
    new_values.insert(new_values.end(), other.values.begin(), other.values.end());

    SparseTensor result(shape, new_indices, new_values);
    result.order();
    result.addUpRedundantEntries();
    return result;
}

SparseTensor SparseTensor::operator-(const SparseTensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("SparseTensor shapes don't match for subtraction");
    }

    std::vector<std::vector<int>> new_indices = indices;
    new_indices.insert(new_indices.end(), other.indices.begin(), other.indices.end());

    std::vector<Complex> new_values = values;
    for (const auto& v : other.values) {
        new_values.push_back(-v);
    }

    SparseTensor result(shape, new_indices, new_values);
    result.order();
    result.addUpRedundantEntries();
    return result;
}

SparseTensor SparseTensor::transpose(const std::vector<int>& permutation) const {
    if (permutation.size() != shape.size()) {
        throw std::runtime_error("Permutation size doesn't match tensor rank");
    }

    std::vector<int> new_shape(shape.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        new_shape[i] = shape[permutation[i]];
    }

    std::vector<std::vector<int>> new_indices;
    for (const auto& idx : indices) {
        std::vector<int> new_idx(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            new_idx[i] = idx[permutation[i]];
        }
        new_indices.push_back(new_idx);
    }

    SparseTensor result(new_shape, new_indices, values, true);
    return result;
}

SparseTensor SparseTensor::vectorize(const std::vector<int>& axes, bool do_order) const {
    // Vectorize specified axes in Fortran (column-major) order
    std::vector<int> axes_complement;
    for (size_t i = 0; i < shape.size(); ++i) {
        bool in_axes = false;
        for (int ax : axes) {
            if ((int)i == ax) {
                in_axes = true;
                break;
            }
        }
        if (!in_axes) {
            axes_complement.push_back(i);
        }
    }

    // Compute strides for vectorization (Fortran order)
    std::vector<int> prefactors;
    prefactors.push_back(1);
    for (size_t i = 0; i < axes.size() - 1; ++i) {
        prefactors.push_back(prefactors[i] * shape[axes[i]]);
    }

    // Compute new shape
    int vec_dim = 1;
    for (int ax : axes) {
        vec_dim *= shape[ax];
    }

    std::vector<int> new_shape;
    new_shape.push_back(vec_dim);
    for (int ax : axes_complement) {
        new_shape.push_back(shape[ax]);
    }

    // Transform indices
    std::vector<std::vector<int>> new_indices;
    for (const auto& idx : indices) {
        std::vector<int> new_idx;

        // Compute vectorized first index
        int vec_idx = 0;
        for (size_t i = 0; i < axes.size(); ++i) {
            vec_idx += prefactors[i] * idx[axes[i]];
        }
        new_idx.push_back(vec_idx);

        // Add complement indices
        for (int ax : axes_complement) {
            new_idx.push_back(idx[ax]);
        }

        new_indices.push_back(new_idx);
    }

    return SparseTensor(new_shape, new_indices, values, do_order);
}

SparseTensor SparseTensor::vectorizeLowerTriangular(const std::vector<int>& axes, bool strict, bool do_order, double scale_off_diagonals) const {
    if (axes.size() != 2) {
        throw std::runtime_error("vectorizeLowerTriangular requires exactly 2 axes");
    }
    if (shape[axes[0]] != shape[axes[1]]) {
        throw std::runtime_error("Matrix axes must have same dimension");
    }

    int n = shape[axes[0]];

    std::vector<int> axes_complement;
    for (size_t i = 0; i < shape.size(); ++i) {
        if ((int)i != axes[0] && (int)i != axes[1]) {
            axes_complement.push_back(i);
        }
    }

    // Compute vectorized dimension
    int vec_dim = strict ? (n * (n - 1) / 2) : (n * (n + 1) / 2);

    // Compute new shape
    std::vector<int> new_shape;
    new_shape.push_back(vec_dim);
    for (int ax : axes_complement) {
        new_shape.push_back(shape[ax]);
    }

    // Precompute mapping for lower triangular indexing
    auto get_vec_index = [n, strict](int i, int j) -> int {
        if (strict) {
            if (i > j) {
                // Lower triangular (strict): map (i,j) where i>j
                int base = 0;
                for (int k = 0; k < j; ++k) {
                    base += n - k - 1;
                }
                return base + (i - j - 1);
            }
            return -1;
        } else {
            if (i >= j) {
                // Lower triangular (non-strict): map (i,j) where i>=j
                int base = 0;
                for (int k = 0; k < j; ++k) {
                    base += n - k;
                }
                return base + (i - j);
            }
            return -1;
        }
    };

    // Transform indices
    std::vector<std::vector<int>> new_indices;
    std::vector<Complex> new_values;

    for (size_t k = 0; k < indices.size(); ++k) {
        const auto& idx = indices[k];
        int i = idx[axes[0]];
        int j = idx[axes[1]];

        int vec_idx = get_vec_index(i, j);
        if (vec_idx >= 0) {
            std::vector<int> new_idx;
            new_idx.push_back(vec_idx);

            for (int ax : axes_complement) {
                new_idx.push_back(idx[ax]);
            }

            new_indices.push_back(new_idx);

            // Apply scaling if requested
            if (scale_off_diagonals != 0.0 && i != j) {
                new_values.push_back(values[k] * scale_off_diagonals);
            } else {
                new_values.push_back(values[k]);
            }
        }
    }

    return SparseTensor(new_shape, new_indices, new_values, do_order);
}

SparseTensor SparseTensor::complexToReal(const std::vector<int>& axes) const {
    if (axes.size() != 2) {
        throw std::runtime_error("complexToReal requires exactly 2 axes");
    }

    int m = shape[axes[0]];
    int n = shape[axes[1]];

    std::vector<int> new_shape = shape;
    new_shape[axes[0]] = 2 * m;
    new_shape[axes[1]] = 2 * n;

    std::vector<std::vector<int>> new_indices;
    std::vector<Complex> new_values;

    for (size_t k = 0; k < indices.size(); ++k) {
        const auto& idx = indices[k];
        Complex val = values[k];
        double re = std::real(val);
        double im = std::imag(val);

        // Upper-left block: real part
        auto idx_ul = idx;
        new_indices.push_back(idx_ul);
        new_values.push_back(re);

        // Lower-right block: real part
        auto idx_lr = idx;
        idx_lr[axes[0]] = idx[axes[0]] + m;
        idx_lr[axes[1]] = idx[axes[1]] + n;
        new_indices.push_back(idx_lr);
        new_values.push_back(re);

        // Lower-left block: imaginary part
        auto idx_ll = idx;
        idx_ll[axes[0]] = idx[axes[0]] + m;
        new_indices.push_back(idx_ll);
        new_values.push_back(im);

        // Upper-right block: negative imaginary part
        auto idx_ur = idx;
        idx_ur[axes[1]] = idx[axes[1]] + n;
        new_indices.push_back(idx_ur);
        new_values.push_back(-im);
    }

    return SparseTensor(new_shape, new_indices, new_values, true);
}

SparseTensor SparseTensor::contractRight(const VectorXc& v, bool do_order) const {
    // Check dimension compatibility
    if (shape.empty()) {
        throw std::runtime_error("Cannot contract empty tensor");
    }
    if (shape.back() != v.size()) {
        throw std::runtime_error("Rightmost tensor dimension " + std::to_string(shape.back()) +
                                 " does not match vector dimension " + std::to_string(v.size()));
    }

    // Order if requested (big-endian/lexicographic)
    SparseTensor temp = *this;
    if (do_order) {
        temp.order(false);  // big-endian
    }

    // Compute output shape (remove last dimension)
    std::vector<int> out_shape(shape.begin(), shape.end() - 1);
    std::vector<std::vector<int>> out_indices;
    std::vector<Complex> out_values;

    Complex current_sum = 0.0;

    for (size_t i = 0; i < temp.values.size(); ++i) {
        const auto& current_index = temp.indices[i];
        current_sum += temp.values[i] * v[current_index.back()];

        bool is_last = (i == temp.values.size() - 1);
        bool prefix_differs = false;

        if (!is_last) {
            const auto& next_index = temp.indices[i + 1];
            // Check if prefix (all but last element) differs
            for (size_t j = 0; j < current_index.size() - 1; ++j) {
                if (next_index[j] != current_index[j]) {
                    prefix_differs = true;
                    break;
                }
            }
        }

        if (is_last || prefix_differs) {
            out_values.push_back(current_sum);
            std::vector<int> out_idx(current_index.begin(), current_index.end() - 1);
            out_indices.push_back(out_idx);
            current_sum = 0.0;
        }
    }

    SparseTensor out(out_shape, out_indices, out_values, false);
    out.removeZeros();
    return out;
}

MatrixXd SparseTensor::contractRight(const MatrixXd& M) const {
    // Validate that this is a rank-2 tensor (matrix)
    if (shape.size() != 2) {
        throw std::runtime_error("contractRight(MatrixXd) requires rank-2 tensor, got rank " + std::to_string(shape.size()));
    }

    // Check dimension compatibility
    if (shape[1] != M.rows()) {
        throw std::runtime_error("Sparse matrix column dimension " + std::to_string(shape[1]) +
                                 " does not match dense matrix row dimension " + std::to_string(M.rows()));
    }

    // Initialize output matrix: [m, k]
    int m = shape[0];
    int k = static_cast<int>(M.cols());
    MatrixXd output = MatrixXd::Zero(m, k);

    // Parallel computation with thread-local accumulation to avoid race conditions
    #pragma omp parallel
    {
        // Each thread maintains its own local output matrix
        MatrixXd local_output = MatrixXd::Zero(m, k);

        // Distribute loop iterations across threads
        #pragma omp for nowait
        for (size_t idx = 0; idx < values.size(); ++idx) {
            int i = indices[idx][0];  // row index
            int j = indices[idx][1];  // column index
            double value = values[idx].real();  // Extract real part

            // Accumulate to thread-local output (no race condition)
            local_output.row(i) += value * M.row(j);
        }

        // Merge thread-local results into global output (critical section)
        #pragma omp critical
        {
            output += local_output;
        }
    }

    return output;
}

MatrixXd SparseTensor::contractRightSingleThreaded(const MatrixXd& M) const {
    // Validate that this is a rank-2 tensor (matrix)
    if (shape.size() != 2) {
        throw std::runtime_error("contractRightSingleThreaded(MatrixXd) requires rank-2 tensor, got rank " + std::to_string(shape.size()));
    }

    // Check dimension compatibility
    if (shape[1] != M.rows()) {
        throw std::runtime_error("Sparse matrix column dimension " + std::to_string(shape[1]) +
                                 " does not match dense matrix row dimension " + std::to_string(M.rows()));
    }

    // Initialize output matrix: [m, k]
    int m = shape[0];
    int k = static_cast<int>(M.cols());
    MatrixXd output = MatrixXd::Zero(m, k);

    // For each nonzero entry (i, j, value) in sparse matrix
    for (size_t idx = 0; idx < values.size(); ++idx) {
        int i = indices[idx][0];  // row index
        int j = indices[idx][1];  // column index
        double value = values[idx].real();  // Extract real part

        // output.row(i) += value * M.row(j)
        output.row(i) += value * M.row(j);
    }

    return output;
}

SparseTensor SparseTensor::contractLeft(const VectorXc& v, bool do_order) const {
    // Check dimension compatibility
    if (shape.empty()) {
        throw std::runtime_error("Cannot contract empty tensor");
    }
    if (shape[0] != v.size()) {
        throw std::runtime_error("Leftmost tensor dimension " + std::to_string(shape[0]) +
                                 " does not match vector dimension " + std::to_string(v.size()));
    }

    // Order if requested (little-endian/reverse-lexicographic)
    SparseTensor temp = *this;
    if (do_order) {
        temp.order(true);  // little-endian
    }

    // Compute output shape (remove first dimension)
    std::vector<int> out_shape(shape.begin() + 1, shape.end());
    std::vector<std::vector<int>> out_indices;
    std::vector<Complex> out_values;

    Complex current_sum = 0.0;

    for (size_t i = 0; i < temp.values.size(); ++i) {
        const auto& current_index = temp.indices[i];
        current_sum += temp.values[i] * v[current_index[0]];

        bool is_last = (i == temp.values.size() - 1);
        bool suffix_differs = false;

        if (!is_last) {
            const auto& next_index = temp.indices[i + 1];
            // Check if suffix (all but first element) differs
            for (size_t j = 1; j < current_index.size(); ++j) {
                if (next_index[j] != current_index[j]) {
                    suffix_differs = true;
                    break;
                }
            }
        }

        if (is_last || suffix_differs) {
            out_values.push_back(current_sum);
            std::vector<int> out_idx(current_index.begin() + 1, current_index.end());
            out_indices.push_back(out_idx);
            current_sum = 0.0;
        }
    }

    SparseTensor out(out_shape, out_indices, out_values, false);
    out.removeZeros();
    return out;
}

SparseMatrix SparseTensor::toEigenSparse() const {
    if (shape.size() != 2) {
        throw std::runtime_error("toEigenSparse only works for 2D tensors");
    }

    std::vector<Eigen::Triplet<Complex>> triplets;
    for (size_t i = 0; i < values.size(); ++i) {
        triplets.push_back(Eigen::Triplet<Complex>(indices[i][0], indices[i][1], values[i]));
    }

    SparseMatrix mat(shape[0], shape[1]);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

Eigen::MatrixXcd SparseTensor::toEigenDense() const {
    if (shape.size() != 2) {
        throw std::runtime_error("toEigenDense only works for 2D tensors");
    }

    MatrixXc result = MatrixXc::Zero(shape[0], shape[1]);
    for (size_t i = 0; i < values.size(); ++i) {
        result(indices[i][0], indices[i][1]) += values[i];
    }

    return result;
}

// Matrix utility functions
MatrixXd complexToReal(const MatrixXc& M) {
    int m = M.rows();
    int n = M.cols();
    MatrixXd result(2 * m, 2 * n);

    result.block(0, 0, m, n) = M.real();
    result.block(0, n, m, n) = -M.imag();
    result.block(m, 0, m, n) = M.imag();
    result.block(m, n, m, n) = M.real();

    return result;
}

MatrixXc realToComplex(const MatrixXd& M, bool sanity_check) {
    int m = M.rows();
    int n = M.cols();

    if (m % 2 != 0 || n % 2 != 0) {
        throw std::runtime_error("Matrix dimensions must be even");
    }

    int half_m = m / 2;
    int half_n = n / 2;

    if (sanity_check) {
        // Check structure
        if (!M.block(0, 0, half_m, half_n).isApprox(M.block(half_m, half_n, half_m, half_n))) {
            throw std::runtime_error("Real part blocks don't match");
        }
        if (!M.block(half_m, 0, half_m, half_n).isApprox(-M.block(0, half_n, half_m, half_n))) {
            throw std::runtime_error("Imaginary part blocks don't match");
        }
    }

    MatrixXc result(half_m, half_n);
    result.real() = M.block(0, 0, half_m, half_n);
    result.imag() = M.block(half_m, 0, half_m, half_n);

    return result;
}

} // namespace hamiltonian_learning
