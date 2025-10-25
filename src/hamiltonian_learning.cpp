#include "hamiltonian_learning.hpp"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace hamiltonian_learning {

// Utility printing functions
void tprint(const std::string& s, bool flush) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t_now);

    std::cout << std::put_time(&tm, "%H:%M:%S") << ": " << s;
    if (flush) {
        std::cout << std::flush;
    }
    std::cout << std::endl;
}

MatrixXc dag(const MatrixXc& M) {
    return M.adjoint();
}

MatrixXc hermitianPart(const MatrixXc& M) {
    return (M + dag(M)) / 2.0;
}

MatrixXc antiHermitianPart(const MatrixXc& M) {
    return (M - dag(M)) / 2.0;
}

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
    int printing_level)
{
    CertifiedBoundsResult result;

    // Create sparse tensor F
    std::vector<int> F_shape = {r, r, s};
    SparseTensor F(F_shape, F_indices, F_values);

    // Step 1: Diagonalize the covariance matrix C
    Eigen::SelfAdjointEigenSolver<MatrixXc> C_solver(C);
    VectorXd C_eigvals = C_solver.eigenvalues();
    MatrixXc C_eigvecs = C_solver.eigenvectors();

    const double small = 1e-9;
    const double very_small = 1e-15;

    if (C_eigvals(0) < very_small) {
        throw std::runtime_error("Covariance matrix is singular. The given expectation values do not correspond to a Gibbs state");
    } else if (C_eigvals(0) < small) {
        if (printing_level > 0) {
            tprint("WARNING: Covariance matrix is near singular");
        }
    }

    // Step 2: Computing quasi-symmetries

    // Build W matrix
    if (printing_level > 2) {
        tprint("building W matrix");
    }

    SparseTensor F_dag = F.conjugate();
    std::vector<int> transpose_perm = {1, 0, 2};
    F_dag = F_dag.transpose(transpose_perm);

    SparseTensor Z = F - F_dag;
    std::vector<int> lower_tri_axes = {0, 1};
    SparseTensor Z_vec = Z.vectorizeLowerTriangular(lower_tri_axes);

    // Convert to Eigen sparse for matrix multiplication
    SparseMatrix Z_vec_sparse = Z_vec.toEigenSparse();
    MatrixXc W = Z_vec_sparse.adjoint() * Z_vec_sparse;

    // Computing the matrix S (orthonormal basis of approximate null space of W)
    Eigen::SelfAdjointEigenSolver<MatrixXc> W_solver(W);
    VectorXd W_eigvals = W_solver.eigenvalues();
    MatrixXc W_eigvecs = W_solver.eigenvectors();

    int q = 0;
    while (q < s) {
        if (W_eigvals(q) > epsilon_W) {
            break;
        }
        q++;
    }

    if (printing_level > 1) {
        tprint("lowest 10 eigenvalues of W:");
        for (int i = 0; i < std::min(10, (int)W_eigvals.size()); ++i) {
            tprintf("  %.6e", W_eigvals(i));
        }
    }

    if (q < 1) {
        throw std::runtime_error("all W eigenvalues are above the cutoff " + std::to_string(epsilon_W));
    }

    if (printing_level > 1) {
        tprint("W eigenvalue threshold = " + std::to_string(epsilon_W));
        tprint("approximate kernel of W has dimension " + std::to_string(q));
    }

    MatrixXc S = W_eigvecs.leftCols(q);

    // S should be real, cast to float
    MatrixXd S_real = S.real();
    if (!S.imag().isZero(1e-10)) {
        if (printing_level > 0) {
            tprint("WARNING: S has non-zero imaginary part");
        }
    }

    // Step 3: Convex optimization

    if (printing_level > 2) {
        tprint("computing modular operator");
    }

    VectorXd sqrtD = C_eigvals.array().sqrt();
    VectorXd sqrtDinv = C_eigvals.array().rsqrt();
    MatrixXc U = C_eigvecs;

    MatrixXc X = sqrtDinv.asDiagonal() * dag(U) * C.transpose() * U * sqrtDinv.asDiagonal();

    Eigen::SelfAdjointEigenSolver<MatrixXc> X_solver(X);
    VectorXd X_eigvals = X_solver.eigenvalues();

    if (X_eigvals(0) < very_small) {
        throw std::runtime_error("modular operator is singular. The given expectation values do not correspond to a Gibbs state");
    } else if (X_eigvals(0) < small) {
        if (printing_level > 0) {
            tprint("WARNING: Modular operator is near singular");
        }
    }

    if (printing_level > 2) {
        tprint("computing log of modular operator");
    }

    MatrixXc logDelta = sqrtD.asDiagonal() * X.log() * sqrtD.asDiagonal();

    // Computing mu_1 and mu_2
    double kappa = 2.0 / C_eigvals.minCoeff();
    double nu = 2.0 * C_eigvals.maxCoeff();

    double mu_1 = (nu * kappa * (3.0 + 2.0 * nu * kappa) + 2.0 * s * beta) * r * epsilon_0;
    double mu_2 = s * beta * r * epsilon_0;

    result.mu_1 = mu_1;
    result.mu_2 = mu_2;
    result.kappa = kappa;
    result.nu = nu;

    if (printing_level > 2) {
        tprint("setting up MOSEK optimization problem");
    }

    // MOSEK optimization
    mosek::fusion::Model::t M = new mosek::fusion::Model("certified_bounds");
    auto _M = monty::finally([&]() { M->dispose(); });

    try {
        // Set number of threads to number of available CPU cores
        int num_threads = std::thread::hardware_concurrency();
        if (printing_level > 0) {
            tprintf("Hardware concurrency: %d threads", num_threads);
        }
        if (num_threads > 0) {
            M->setSolverParam("numThreads", num_threads);
            if (printing_level > 0) {
                tprintf("Setting MOSEK numThreads: %d", num_threads);
            }
        }

        // Configure MOSEK logging based on printing_level
        if (printing_level >= 3) {
            // Enable detailed MOSEK output (only at high verbosity)
            M->setLogHandler([](const std::string &msg) {
                std::cout << msg << std::flush;
            });
            M->setSolverParam("log", 10);  // Maximum verbosity
            M->setSolverParam("logCutSecondOpt", 0);  // Full log for second optimization
        } else {
            // Suppress MOSEK output for printing_level 0-2
            M->setLogHandler([](const std::string &msg) {});
        }

        // Variables
        auto A = M->variable("A", mosek::fusion::Domain::inPSDCone(2*r));
        auto B = M->variable("B", mosek::fusion::Domain::inPSDCone(2*r));
        auto C_var = M->variable("C", mosek::fusion::Domain::inPSDCone(2*r));

        // Precomputing tensor contraction of F with S
        if (printing_level > 2) {
            tprint("computing tensor contraction F with S");
        }

        std::vector<int> complex_axes = {0, 1};
        SparseTensor F_real = F.complexToReal(complex_axes);
        SparseTensor F_real_vectorized = F_real.vectorize({0, 1});

        // Contract with S: (F_real_vectorized @ S_real).reshape((2*r, 2*r, q))
        // Convert to dense for matrix multiplication
        //MatrixXd F_dense = F_real_vectorized.toEigenDense().real();
        //MatrixXd FS_real_vectorized = F_dense * S_real;
        MatrixXd FS_real_vectorized = F_real_vectorized.contractRight(S_real);

        // Reshape to (2*r, 2*r, q) and compute B_prime_alphas
        MatrixXd U_real = complexToReal(U);
        std::vector<MatrixXd> B_prime_alphas;
        for (int i = 0; i < q; ++i) {
            Eigen::Map<MatrixXd> FS_slice(FS_real_vectorized.col(i).data(), 2*r, 2*r);
            MatrixXd B_prime_alpha = U_real.transpose() * FS_slice * U_real;
            B_prime_alphas.push_back(B_prime_alpha);
        }

        // Build constraint expressions
        if (printing_level > 2) {
            tprint("building constraint expressions");
        }

        VectorXd omega_h_tilde = h_terms_exp.transpose() * S_real;

        std::vector<mosek::fusion::Expression::t> constraint_exprs;
        for (int i = 0; i < q; ++i) {
            // B_prime_alphas are real matrices, but we need hermitian/anti-hermitian parts
            // Treat them as the real representation of complex matrices
            MatrixXd& B_alpha = B_prime_alphas[i];

            // For a real 2n×2n matrix representing a complex n×n matrix:
            // Hermitian part is symmetric part
            // Anti-hermitian part is skew-symmetric part
            MatrixXd herm_part = (B_alpha + B_alpha.transpose()) / 2.0;
            MatrixXd anti_herm_part = (B_alpha - B_alpha.transpose()) / 2.0;

            // Build constraint: -dot(herm, A) + dot(anti_herm, B) - dot(anti_herm, C)
            auto expr = mosek::fusion::Expr::dot(
                mosek::fusion::Matrix::dense(
                    std::make_shared<monty::ndarray<double, 2>>(
                        herm_part.data(), monty::shape(2*r, 2*r))),
                A);
            expr = mosek::fusion::Expr::mul(-1.0, expr);

            expr = mosek::fusion::Expr::add(expr,
                mosek::fusion::Expr::dot(
                    mosek::fusion::Matrix::dense(
                        std::make_shared<monty::ndarray<double, 2>>(
                            anti_herm_part.data(), monty::shape(2*r, 2*r))),
                    B));

            expr = mosek::fusion::Expr::add(expr,
                mosek::fusion::Expr::dot(
                    mosek::fusion::Matrix::dense(
                        std::make_shared<monty::ndarray<double, 2>>(
                            (-anti_herm_part).eval().data(), monty::shape(2*r, 2*r))),
                    C_var));

            constraint_exprs.push_back(expr);
        }

        // Convert vector to monty::ndarray for vstack
        auto exprs_array = std::make_shared<monty::ndarray<mosek::fusion::Expression::t, 1>>(
            monty::shape(q));
        for (int i = 0; i < q; ++i) {
            (*exprs_array)[i] = constraint_exprs[i];
        }
        auto constraint_expr = mosek::fusion::Expr::vstack(exprs_array);

        if (q == 1) {
            tprint("warning: q = 1 and MOSEK has weird behaviour when this is the case :(");
        }

        // First constraint: equals to -v @ S
        VectorXd constraint_rhs = -v.transpose() * S_real;
        auto constraint_a = M->constraint("a", constraint_expr,
            mosek::fusion::Domain::equalsTo(
                std::make_shared<monty::ndarray<double, 1>>(
                    constraint_rhs.data(), monty::shape(q))));

        // Set up objective function
        if (printing_level > 2) {
            tprint("setting up objective function");
        }

        MatrixXd logDelta_real = complexToReal(logDelta);
        auto objective_expr = mosek::fusion::Expr::dot(
            mosek::fusion::Matrix::dense(
                std::make_shared<monty::ndarray<double, 2>>(
                    (-logDelta_real).eval().data(), monty::shape(2*r, 2*r))),
            A);
        objective_expr = mosek::fusion::Expr::add(objective_expr,
            mosek::fusion::Expr::mul(-mu_1,
                mosek::fusion::Expr::dot(mosek::fusion::Matrix::eye(2*r), A)));
        objective_expr = mosek::fusion::Expr::add(objective_expr,
            mosek::fusion::Expr::mul(-mu_2,
                mosek::fusion::Expr::dot(mosek::fusion::Matrix::eye(2*r), B)));
        objective_expr = mosek::fusion::Expr::add(objective_expr,
            mosek::fusion::Expr::mul(-mu_2,
                mosek::fusion::Expr::dot(mosek::fusion::Matrix::eye(2*r), C_var)));

        M->objective(mosek::fusion::ObjectiveSense::Maximize, objective_expr);

        if (printing_level > 2) {
            tprint("running MOSEK:");
            tprint("");
        }

        // Solve for lower bound
        M->solve();
        result.lower_bound = M->primalObjValue();

        // Solve for upper bound by replacing v with -v
        constraint_a->remove();
        VectorXd constraint_rhs_upper = v.transpose() * S_real;
        M->constraint("b", constraint_expr,
            mosek::fusion::Domain::equalsTo(
                std::make_shared<monty::ndarray<double, 1>>(
                    constraint_rhs_upper.data(), monty::shape(q))));

        M->solve();
        result.upper_bound = -M->primalObjValue();

    } catch (const mosek::fusion::FusionException& e) {
        std::cerr << "MOSEK error: " << e.what() << std::endl;
        throw;
    }

    return result;
}

} // namespace hamiltonian_learning
