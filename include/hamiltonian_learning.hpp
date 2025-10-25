#ifndef HAMILTONIAN_LEARNING_HPP
#define HAMILTONIAN_LEARNING_HPP

#include "sparse_tensor.hpp"
#include <Eigen/Dense>
#include <fusion.h>
#include <string>

namespace hamiltonian_learning {

// Utility printing functions
void tprint(const std::string& s, bool flush = true);

template<typename... Args>
void tprintf(const char* format, Args... args);

// Implementation must be in header for template
template<typename... Args>
void tprintf(const char* format, Args... args) {
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), format, args...);
    tprint(std::string(buffer));
}

struct CertifiedBoundsResult {
    double lower_bound;
    double upper_bound;
    double mu_1;
    double mu_2;
    double kappa;
    double nu;
};

/**
 * Compute certified bounds on a linear functional of the Hamiltonian
 *
 * This implements the certified bounds algorithm from the paper.
 *
 * @param r Number of perturbing operators
 * @param s Number of variational Hamiltonian terms
 * @param h_terms_exp Expectations of Hamiltonian terms
 * @param J Matrix satisfying b_i* = sum_j J_ij b_j
 * @param C Covariance matrix: C_ij = omega(b_i* b_j)
 * @param F_indices Sparse tensor indices (COO format) for F_ijk = omega(b_i* [h_k, b_j])
 * @param F_values Sparse tensor values (COO format)
 * @param epsilon_W W eigenvalue threshold
 * @param epsilon_0 Measurement error bound
 * @param beta Inverse temperature
 * @param v Direction vector for the linear functional
 * @param printing_level Verbosity (0=silent, 1-2=progress only, 3+=show MOSEK output)
 * @return CertifiedBoundsResult with lower/upper bounds and parameters
 */
CertifiedBoundsResult certifiedBoundsV2(
    int r,
    int s,
    const VectorXd& h_terms_exp,
    const MatrixXc& J,
    const MatrixXc& C,
    const std::vector<std::vector<int>>& F_indices,
    const std::vector<Complex>& F_values,
    double epsilon_W,
    double epsilon_0,
    double beta,
    const VectorXd& v,
    int printing_level = 0
);

// Helper functions
MatrixXc dag(const MatrixXc& M);
MatrixXc hermitianPart(const MatrixXc& M);
MatrixXc antiHermitianPart(const MatrixXc& M);

} // namespace hamiltonian_learning

#endif // HAMILTONIAN_LEARNING_HPP
