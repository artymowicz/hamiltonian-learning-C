#include "../examples/simulation.hpp"
#include "../examples/pauli_utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>

using namespace hamiltonian_learning;

// ============================================================================
// From test_ED.cpp
// ============================================================================

std::pair<bool, std::string> test_ED_analytical(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing ED Against Analytical Expectations ===" << std::endl;
    log << "Using threshold: " << std::scientific << threshold << std::endl;
    log << std::endl;

    // Use classical Ising ferromagnet with small g (product state |+++>)
    int n = 5;
    std::map<std::string, double> couplings = {{"g", 0.1}};

    log << "Loading classical_ising_ferro Hamiltonian with n=" << n << ", g=0.1..." << std::endl;
    log << "With g << 1, ground state is approximately |++...+> (all spins up)" << std::endl;
    log << std::endl;

    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/classical_ising_ferro.yml", couplings);

    log << "Hamiltonian: H = -sum_i Z_i Z_{i+1} + g * sum_i Z_i" << std::endl;
    log << "Number of terms: " << H.terms.size() << std::endl;
    log << std::endl;

    // Compute ground state using ED
    log << "Computing ground state using ED..." << std::endl;
    VectorXcd psi_ed = computeGroundStateED(H);
    log << "Ground state computed (dimension = " << psi_ed.size() << ")" << std::endl;
    log << std::endl;

    log << "Computing expectation values and comparing to analytical predictions:" << std::endl;
    log << std::endl;

    // Test single-site Z operators
    log << "Single-site Z operators (analytical: +1.0 for all sites):" << std::endl;
    log << "Operator     ED                Analytical        Difference" << std::endl;
    log << "------------------------------------------------------------------------" << std::endl;

    double max_z_error = 0.0;
    for (int i = 0; i < n; ++i) {
        std::string op(n, 'I');
        op[i] = 'Z';
        double exp_ed = computeExpectation(op, psi_ed);
        double exp_analytical = 1.0;
        double diff = std::abs(exp_ed - exp_analytical);
        max_z_error = std::max(max_z_error, diff);

        log << std::left << std::setw(12) << op << " "
            << std::scientific << std::setprecision(4)
            << std::setw(18) << exp_ed << " "
            << std::setw(18) << exp_analytical << " "
            << std::setw(18) << diff << std::endl;
    }
    log << std::endl;

    // Test nearest-neighbor ZZ operators
    log << "Nearest-neighbor ZZ operators (analytical: +1.0 for all pairs):" << std::endl;
    log << "Operator     ED                Analytical        Difference" << std::endl;
    log << "------------------------------------------------------------------------" << std::endl;

    double max_zz_error = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        std::string op(n, 'I');
        op[i] = 'Z';
        op[i + 1] = 'Z';
        double exp_ed = computeExpectation(op, psi_ed);
        double exp_analytical = 1.0;
        double diff = std::abs(exp_ed - exp_analytical);
        max_zz_error = std::max(max_zz_error, diff);

        log << std::left << std::setw(12) << op << " "
            << std::scientific << std::setprecision(4)
            << std::setw(18) << exp_ed << " "
            << std::setw(18) << exp_analytical << " "
            << std::setw(18) << diff << std::endl;
    }
    log << std::endl;

    // Test single-site X operators (should be exactly zero)
    log << "Single-site X operators (analytical: exactly 0):" << std::endl;
    log << "Operator     ED                Analytical        Difference" << std::endl;
    log << "------------------------------------------------------------------------" << std::endl;

    double max_x_error = 0.0;
    for (int i = 0; i < n; ++i) {
        std::string op(n, 'I');
        op[i] = 'X';
        double exp_ed = computeExpectation(op, psi_ed);
        double exp_analytical = 0.0;
        double diff = std::abs(exp_ed - exp_analytical);
        max_x_error = std::max(max_x_error, diff);

        log << std::left << std::setw(12) << op << " "
            << std::scientific << std::setprecision(4)
            << std::setw(18) << exp_ed << " "
            << std::setw(18) << exp_analytical << " "
            << std::setw(18) << diff << std::endl;
    }
    log << std::endl;

    // Summary
    log << "========================================" << std::endl;
    log << "Test Summary:" << std::endl;
    log << "========================================" << std::endl;
    log << "Maximum error in <Z_i>:        " << std::scientific << std::setprecision(4) << max_z_error << std::endl;
    log << "Maximum error in <Z_i Z_{i+1}>: " << std::scientific << std::setprecision(4) << max_zz_error << std::endl;
    log << "Maximum error in <X_i>:        " << std::scientific << std::setprecision(4) << max_x_error << std::endl;
    log << std::endl;

    // Check pass/fail against threshold
    bool z_pass = max_z_error < threshold;
    bool zz_pass = max_zz_error < threshold;
    bool x_pass = max_x_error < threshold;

    if (z_pass && zz_pass && x_pass) {
        log << "✓ Test PASSED" << std::endl;
        log << "  ED results agree with analytical expectations for product state" << std::endl;
    } else {
        passed = false;
        log << "✗ Test FAILED" << std::endl;
        if (!z_pass) log << "  Z expectations deviate from analytical (max error: " << max_z_error << " > threshold: " << threshold << ")" << std::endl;
        if (!zz_pass) log << "  ZZ expectations deviate from analytical (max error: " << max_zz_error << " > threshold: " << threshold << ")" << std::endl;
        if (!x_pass) log << "  X expectations deviate from analytical (max error: " << max_x_error << " > threshold: " << threshold << ")" << std::endl;
    }

    return {passed, log.str()};
}

// ============================================================================
// From test_dmrg_against_ED.cpp
// ============================================================================

std::pair<bool, std::string> test_dmrg_against_ED(double threshold, const std::string& hamiltonian_arg, int n,
                                                   const std::map<std::string, double>& couplings,
                                                   const std::vector<std::string>& operators) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing DMRG vs ED for Ground State ===" << std::endl;
    log << std::endl;

    Hamiltonian H(n, {}, {});  // Will be populated below

    if (hamiltonian_arg == "random") {
        // Generate random Hamiltonian
        log << "Generating random nearest-neighbor Hamiltonian with n=" << n << std::endl;

        // Random number generator
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        std::vector<std::string> terms;
        std::vector<double> coefficients;

        // Add single-site terms (X, Y, Z on each site)
        for (int i = 0; i < n; ++i) {
            // X term
            std::string x_term(n, 'I');
            x_term[i] = 'X';
            terms.push_back(x_term);
            coefficients.push_back(dis(gen));

            // Y term
            std::string y_term(n, 'I');
            y_term[i] = 'Y';
            terms.push_back(y_term);
            coefficients.push_back(dis(gen));

            // Z term
            std::string z_term(n, 'I');
            z_term[i] = 'Z';
            terms.push_back(z_term);
            coefficients.push_back(dis(gen));
        }

        // Add two-site nearest-neighbor terms
        for (int i = 0; i < n - 1; ++i) {
            // XX term
            std::string xx_term(n, 'I');
            xx_term[i] = 'X';
            xx_term[i + 1] = 'X';
            terms.push_back(xx_term);
            coefficients.push_back(dis(gen));

            // YY term
            std::string yy_term(n, 'I');
            yy_term[i] = 'Y';
            yy_term[i + 1] = 'Y';
            terms.push_back(yy_term);
            coefficients.push_back(dis(gen));

            // ZZ term
            std::string zz_term(n, 'I');
            zz_term[i] = 'Z';
            zz_term[i + 1] = 'Z';
            terms.push_back(zz_term);
            coefficients.push_back(dis(gen));

            // XY term
            std::string xy_term(n, 'I');
            xy_term[i] = 'X';
            xy_term[i + 1] = 'Y';
            terms.push_back(xy_term);
            coefficients.push_back(dis(gen));

            // XZ term
            std::string xz_term(n, 'I');
            xz_term[i] = 'X';
            xz_term[i + 1] = 'Z';
            terms.push_back(xz_term);
            coefficients.push_back(dis(gen));

            // YZ term
            std::string yz_term(n, 'I');
            yz_term[i] = 'Y';
            yz_term[i + 1] = 'Z';
            terms.push_back(yz_term);
            coefficients.push_back(dis(gen));
        }

        H = Hamiltonian(n, terms, coefficients);

        log << std::endl;
        log << "Generated Hamiltonian with " << terms.size() << " terms:" << std::endl;
        // Don't print full Hamiltonian to reduce log size
        log << std::endl;
    } else {
        // Load from YAML file

        std::ostringstream couplings_oss;

        couplings_oss << "{";
        for (auto it = couplings.begin(); it != couplings.end(); ++it) {
            couplings_oss << it->first << ": " << it->second;
            if (std::next(it) != couplings.end())
                couplings_oss << ", ";
        }
        couplings_oss << "}";

        //cout << couplings_oss.str();

        log << "Loading Hamiltonian: " << hamiltonian_arg << " with couplings " << couplings_oss.str() << std::endl;
        log << "System size: n=" << n << std::endl;
        H = Hamiltonian::loadFromYAML(n, "../hamiltonians/" + hamiltonian_arg, couplings);

        log << std::endl;
        log << "Hamiltonian has " << H.terms.size() << " terms" << std::endl;
        log << std::endl;
    }

    // Compute ground state using ED
    log << "Computing ground state using ED..." << std::endl;
    VectorXcd psi_ed = computeGroundStateED(H);

    // Compute ground state energy from ED
    MatrixXcd H_matrix = MatrixXcd::Zero(1 << n, 1 << n);
    for (size_t i = 0; i < H.terms.size(); ++i) {
        H_matrix += H.coefficients[i] * pauliMatrix(H.terms[i]);
    }
    Complex energy_complex = (psi_ed.adjoint() * H_matrix * psi_ed)(0,0);
    double energy_ed = energy_complex.real();
    log << "ED ground state energy: " << std::fixed << std::setprecision(10) << energy_ed << std::endl;
    log << std::endl;

    // Compute ground state using DMRG
    log << "Computing ground state using DMRG..." << std::endl;
    DMRGParams dmrg_params;
    dmrg_params.chi_max = 100;
    dmrg_params.max_sweeps = 20;
    dmrg_params.svd_min = 1e-10;
    dmrg_params.quiet = true;  // Quiet for cleaner logs

    auto [psi_dmrg, energy_dmrg] = computeGroundState(H, dmrg_params);

    log << "DMRG completed successfully!" << std::endl;
    log << "Maximum MPS bond dimension: " << itensor::maxLinkDim(psi_dmrg) << std::endl;
    log << "DMRG ground state energy: " << std::fixed << std::setprecision(10) << energy_dmrg << std::endl;
    log << std::endl;

    if (!operators.empty()) {
        // Compute expectation values for requested operators
        log << "Expectation values:" << std::endl;
        log << "Operator     ED                DMRG              Difference" << std::endl;
        log << "------------------------------------------------------------------------" << std::endl;

        double max_diff = 0.0;
        for (const auto& op : operators) {
            if (op.length() != static_cast<size_t>(n)) {
                log << "Warning: operator " << op << " has wrong length (expected " << n << ")" << std::endl;
                passed = false;
                continue;
            }
            double exp_ed = computeExpectation(op, psi_ed);
            double exp_dmrg = computeExpectation(psi_dmrg, op);
            double diff = std::abs(exp_ed - exp_dmrg);
            max_diff = std::max(max_diff, diff);

            log << std::left << std::setw(12) << op << " "
                      << std::scientific << std::setprecision(10)
                      << std::setw(18) << exp_ed << " "
                      << std::setw(18) << exp_dmrg << " "
                      << std::setw(18) << diff << std::endl;
        }
        log << std::endl;
        log << "Maximum difference: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
        log << "Threshold: " << threshold << std::endl;

        if (max_diff > threshold) {
            passed = false;
        }
    } else {
        // No operators specified, just note energy comparison
        log << "No operators specified for comparison" << std::endl;
        log << "  ED energy:   " << std::fixed << std::setprecision(10) << energy_ed << std::endl;
        log << std::endl;
    }

    return {passed, log.str()};
}

// ============================================================================
// From test_purification_vs_ED.cpp
// ============================================================================

std::pair<bool, std::string> test_purification_vs_ED(double threshold, const std::string& hamiltonian_file, int n,
                                                      double beta, const ThermalParams& thermal_params,
                                                      bool all_local_operators, bool naive,
                                                      const std::map<std::string, double>& couplings,
                                                      const std::vector<std::string>& operators) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing Thermal State: Purification (supersites) vs ED ===" << std::endl;
    log << std::endl;

    log << "Loading Hamiltonian: " << hamiltonian_file << std::endl;
    log << "System size: n=" << n << std::endl;
    log << "Inverse temperature: beta=" << beta << std::endl;
    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/" + hamiltonian_file, couplings);

    log << std::endl;
    log << "Hamiltonian has " << H.terms.size() << " terms" << std::endl;
    log << std::endl;

    // Compute thermal state using ED
    log << "Computing thermal state using ED..." << std::endl;
    MatrixXcd rho_ed = computeEquilibriumStateED(H, beta);
    log << "ED completed successfully!" << std::endl;
    log << std::endl;

    // Compute purified thermal state using MPS (supersites)
    log << "Computing purified thermal state (dt=" << thermal_params.dt << ", order=" << thermal_params.order << ")..." << std::endl;

    itensor::MPS psi_thermal = computePurifiedEquilibriumState(H, beta, thermal_params);
    log << "Purified state computed successfully!" << std::endl;
    log << "Maximum MPS bond dimension: " << itensor::maxLinkDim(psi_thermal) << std::endl;
    log << std::endl;

    // Generate or validate operators
    std::vector<std::string> valid_operators;

    if (all_local_operators) {
        // Single-site operators: X, Y, Z on each site
        for (int site = 0; site < n; ++site) {
            for (char pauli : {'X', 'Y', 'Z'}) {
                std::string op(n, 'I');
                op[site] = pauli;
                valid_operators.push_back(op);
            }
        }

        // Nearest-neighbor operators: all combinations of {X,Y,Z} × {X,Y,Z}
        for (int site = 0; site < n - 1; ++site) {
            for (char pauli1 : {'X', 'Y', 'Z'}) {
                for (char pauli2 : {'X', 'Y', 'Z'}) {
                    std::string op(n, 'I');
                    op[site] = pauli1;
                    op[site + 1] = pauli2;
                    valid_operators.push_back(op);
                }
            }
        }
    } else {
        // Filter operators to valid length
        for (const auto& op : operators) {
            if (op.length() != static_cast<size_t>(n)) {
                log << "Warning: operator " << op << " has wrong length (expected " << n << ")" << std::endl;
                passed = false;
            } else {
                valid_operators.push_back(op);
            }
        }
    }

    // Compute expectation values using batch functions
    log << "Computing expectations using " << (naive ? "naive" : "DFS-optimized") << " method..." << std::endl;
    std::vector<double> exp_ed = computeExpectationsED(rho_ed, valid_operators);
    std::vector<double> exp_mps;
    if (naive) {
        exp_mps = computeExpectationsMixedNaive(psi_thermal, valid_operators);
    } else {
        exp_mps = computeExpectationsMixed(psi_thermal, valid_operators);
    }

    // Print comparison
    log << std::endl;
    log << "Expectation values:" << std::endl;
    log << "Operator     ED                MPS               Difference" << std::endl;
    log << "------------------------------------------------------------------------" << std::endl;

    double max_diff = 0.0;
    for (size_t i = 0; i < valid_operators.size(); ++i) {
        double diff = std::abs(exp_ed[i] - exp_mps[i]);
        max_diff = std::max(max_diff, diff);

        log << std::left << std::setw(12) << valid_operators[i] << " "
                  << std::scientific << std::setprecision(10)
                  << std::setw(18) << exp_ed[i] << " "
                  << std::setw(18) << exp_mps[i] << " "
                  << std::setw(18) << diff << std::endl;
    }

    log << std::endl;
    log << "Maximum difference: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    log << "Threshold: " << threshold << std::endl;

    if (max_diff > threshold) {
        passed = false;
    }

    return {passed, log.str()};
}

// ============================================================================
// From test_expectations_naive.cpp
// ============================================================================

std::pair<bool, std::string> test_expectations_naive(double threshold, const std::string& hamiltonian_file, int n,
                                                      double beta, const ThermalParams& thermal_params,
                                                      bool all_local_operators, bool naive,
                                                      const std::map<std::string, double>& couplings,
                                                      const std::vector<std::string>& operators) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing Naive Expectations Functions ===" << std::endl;
    log << std::endl;

    log << "Loading Hamiltonian: " << hamiltonian_file << std::endl;
    log << "System size: n=" << n << std::endl;
    log << "Inverse temperature: beta=" << beta << std::endl;
    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/" + hamiltonian_file, couplings);

    log << std::endl;
    log << "Hamiltonian has " << H.terms.size() << " terms" << std::endl;
    log << std::endl;

    // Compute ground state
    log << "Computing ground state using DMRG..." << std::endl;
    DMRGParams dmrg_params;
    dmrg_params.quiet = true;
    auto [psi_ground, energy_ground] = computeGroundState(H, dmrg_params);
    log << std::endl;

    // Compute thermal state
    log << "Computing thermal state (beta=" << beta << ", dt=" << thermal_params.dt << ", order=" << thermal_params.order << ") using purification..." << std::endl;
    itensor::MPS psi_thermal = computePurifiedEquilibriumState(H, beta, thermal_params);
    log << std::endl;

    // Test pure state expectations
    log << "Testing computeExpectationsPureNaive on ground state:" << std::endl;
    auto exp_pure = computeExpectationsPureNaive(psi_ground, operators);
    for (size_t i = 0; i < operators.size(); ++i) {
        log << "  <" << operators[i] << "> = " << std::fixed << std::setprecision(6) << exp_pure[i] << std::endl;
    }
    log << std::endl;

    // Test mixed state expectations
    log << "Testing computeExpectationsMixedNaive on thermal state:" << std::endl;
    auto exp_mixed = computeExpectationsMixedNaive(psi_thermal, operators);
    for (size_t i = 0; i < operators.size(); ++i) {
        log << "  <" << operators[i] << "> = " << std::fixed << std::setprecision(6) << exp_mixed[i] << std::endl;
    }
    log << std::endl;

    // Verify against ED for ground state
    log << "Verifying pure state against ED:" << std::endl;
    VectorXcd psi_ed = computeGroundStateED(H);
    auto exp_ed_pure = computeExpectationsED(psi_ed, operators);

    log << "  Operator  | MPS       | ED        | Diff" << std::endl;
    log << "  ----------|-----------|-----------|----------" << std::endl;
    double max_diff_pure = 0.0;
    for (size_t i = 0; i < operators.size(); ++i) {
        double diff = std::abs(exp_pure[i] - exp_ed_pure[i]);
        max_diff_pure = std::max(max_diff_pure, diff);
        log << "  " << operators[i] << "       | "
                  << std::fixed << std::setprecision(6) << exp_pure[i] << " | "
                  << exp_ed_pure[i] << " | "
                  << std::scientific << std::setprecision(2) << diff << std::endl;
    }
    log << std::endl;

    // Verify against ED for thermal state
    log << "Verifying mixed state against ED:" << std::endl;
    MatrixXcd rho_ed = computeEquilibriumStateED(H, beta);
    auto exp_ed_mixed = computeExpectationsED(rho_ed, operators);

    log << "  Operator  | MPS       | ED        | Diff" << std::endl;
    log << "  ----------|-----------|-----------|----------" << std::endl;
    double max_diff_mixed = 0.0;
    for (size_t i = 0; i < operators.size(); ++i) {
        double diff = std::abs(exp_mixed[i] - exp_ed_mixed[i]);
        max_diff_mixed = std::max(max_diff_mixed, diff);
        log << "  " << operators[i] << "       | "
                  << std::fixed << std::setprecision(6) << exp_mixed[i] << " | "
                  << exp_ed_mixed[i] << " | "
                  << std::scientific << std::setprecision(2) << diff << std::endl;
    }
    log << std::endl;

    log << "Maximum difference (pure state):  " << std::scientific << std::setprecision(4) << max_diff_pure << std::endl;
    log << "Maximum difference (mixed state): " << std::scientific << std::setprecision(4) << max_diff_mixed << std::endl;
    log << "Threshold: " << threshold << std::endl;

    if (max_diff_pure > threshold || max_diff_mixed > threshold) {
        passed = false;
    }

    return {passed, log.str()};
}

// ============================================================================
// From test_expectations_dfs.cpp
// ============================================================================

std::pair<bool, std::string> test_expectations_dfs(const std::string& hamiltonian_file, int n,
                                                     const std::map<std::string, double>& couplings,
                                                     double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing DFS-Optimized Expectations (Pure State) ===" << std::endl;
    log << "Using threshold: " << std::scientific << threshold << std::endl;
    log << std::endl;

    log << "Loading " << hamiltonian_file << " with n=" << n;
    for (const auto& [name, value] : couplings) {
        log << ", " << name << "=" << value;
    }
    log << std::endl << std::endl;

    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/" + hamiltonian_file, couplings);

    // Compute ground state
    log << "Computing ground state using DMRG..." << std::endl;
    DMRGParams dmrg_params;
    dmrg_params.quiet = true;
    auto [psi_ground, energy_ground] = computeGroundState(H, dmrg_params);
    log << "Maximum MPS bond dimension: " << itensor::maxLinkDim(psi_ground) << std::endl;
    log << std::endl;

    // Generate operators
    auto operators = buildKLocalPaulis1D(n, 2, false);
    // Sort in reverse alphabetical order for DFS optimization
    std::sort(operators.begin(), operators.end(), std::greater<std::string>());

    log << "Number of operators: " << operators.size() << std::endl;
    log << std::endl;

    // Compute using DFS method
    log << "Computing expectations using DFS method..." << std::endl;
    auto exp_dfs = computeExpectationsPure(psi_ground, operators);

    // Compute using naive method for comparison
    log << "Computing expectations using naive method..." << std::endl;
    auto exp_naive = computeExpectationsPureNaive(psi_ground, operators);

    log << std::endl;

    // Compare results
    log << "Comparing DFS vs Naive:" << std::endl;
    log << "  Operator      | DFS       | Naive     | Diff" << std::endl;
    log << "  --------------|-----------|-----------|----------" << std::endl;

    double max_diff = 0.0;
    for (size_t i = 0; i < operators.size(); ++i) {
        double diff = std::abs(exp_dfs[i] - exp_naive[i]);
        max_diff = std::max(max_diff, diff);
        log << "  " << operators[i] << "     | "
            << std::fixed << std::setprecision(6) << exp_dfs[i] << " | "
            << exp_naive[i] << " | "
            << std::scientific << std::setprecision(2) << diff << std::endl;
    }
    log << std::endl;

    log << "Maximum difference: " << std::scientific << std::setprecision(3) << max_diff << std::endl;
    log << std::endl;

    if (max_diff < threshold) {
        log << "✓ Test PASSED: DFS matches naive within tolerance!" << std::endl;
    } else {
        passed = false;
        log << "✗ Test FAILED: DFS does not match naive" << std::endl;
        log << "  Max difference: " << max_diff << " > threshold: " << threshold << std::endl;
    }

    return {passed, log.str()};
}

// ============================================================================
// From test_expectations_dfs_mixed.cpp
// ============================================================================

std::pair<bool, std::string> test_expectations_dfs_mixed(double threshold, double beta, double dt) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing DFS-Optimized Expectations (Mixed State) ===" << std::endl;
    log << "Using threshold: " << std::scientific << threshold << std::endl;
    log << std::endl;

    int n = 5;

    log << "Loading classical_ising_ferro with n=" << n << ", g=-0.1..." << std::endl;
    std::map<std::string, double> couplings = {{"g", -0.1}};
    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/classical_ising_ferro.yml", couplings);

    // Compute thermal state
    log << "Computing thermal state (beta=" << beta << ", dt=" << dt << ") using purification..." << std::endl;
    ThermalParams thermal_params;
    thermal_params.dt = dt;
    thermal_params.quiet = true;
    itensor::MPS psi_thermal = computePurifiedEquilibriumState(H, beta, thermal_params);
    log << "Maximum MPS bond dimension: " << itensor::maxLinkDim(psi_thermal) << std::endl;
    log << std::endl;

    // Define operators
    std::vector<std::string> operators = {
        "ZIIII", "IZIII", "IIZII", "IIIIZ",
        "ZZIII", "IZZII", "IIZZI", "IIIZZ",
        "XXIII", "IXXII", "IIXXI", "IIIXX"
    };

    // Sort in reverse alphabetical order for DFS optimization
    std::sort(operators.begin(), operators.end(), std::greater<std::string>());

    log << "Number of operators: " << operators.size() << std::endl;
    log << std::endl;

    // Compute using DFS method
    log << "Computing expectations using DFS method..." << std::endl;
    auto exp_dfs = computeExpectationsMixed(psi_thermal, operators);

    // Compute using naive method for comparison
    log << "Computing expectations using naive method..." << std::endl;
    auto exp_naive = computeExpectationsMixedNaive(psi_thermal, operators);

    log << std::endl;

    // Compare results
    log << "Comparing DFS vs Naive:" << std::endl;
    log << "  Operator  | DFS       | Naive     | Diff" << std::endl;
    log << "  ----------|-----------|-----------|----------" << std::endl;

    double max_diff = 0.0;
    for (size_t i = 0; i < operators.size(); ++i) {
        double diff = std::abs(exp_dfs[i] - exp_naive[i]);
        max_diff = std::max(max_diff, diff);
        log << "  " << operators[i] << " | "
            << std::fixed << std::setprecision(6) << exp_dfs[i] << " | "
            << exp_naive[i] << " | "
            << std::scientific << std::setprecision(2) << diff << std::endl;
    }
    log << std::endl;

    log << "Maximum difference: " << std::scientific << std::setprecision(3) << max_diff << std::endl;
    log << std::endl;

    if (max_diff < threshold) {
        log << "✓ Test PASSED: DFS matches naive within tolerance!" << std::endl;
    } else {
        passed = false;
        log << "✗ Test FAILED: DFS does not match naive" << std::endl;
        log << "  Max difference: " << max_diff << " > threshold: " << threshold << std::endl;
    }

    return {passed, log.str()};
}

// ============================================================================
// Main function to run all tests
// ============================================================================

int main(int argc, char* argv[]) {
    double threshold = 1e-10;
    bool verbose = false;
    double beta = 0.1;      // Default beta value
    double dt = 1e-5;       // Default Trotter timestep
    int chi_max = 100;      // Default max bond dimension
    double svd_min = 1e-12; // Default SVD cutoff

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS] [THRESHOLD]" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -v, --verbose        Enable verbose output (show all test logs)" << std::endl;
            std::cout << "  -b, --beta VALUE     Set inverse temperature (default: 0.1)" << std::endl;
            std::cout << "  -d, --dt VALUE       Set Trotter timestep for thermal state (default: 1e-5)" << std::endl;
            std::cout << "  --chi-max VALUE      Set maximum bond dimension (default: 100)" << std::endl;
            std::cout << "  --svd-min VALUE      Set SVD cutoff (default: 1e-12)" << std::endl;
            std::cout << "  -h, --help           Show this help message" << std::endl;
            std::cout << std::endl;
            std::cout << "  THRESHOLD            Error threshold for tests (default: 1e-10)" << std::endl;
            std::cout << std::endl;
            std::cout << "Examples:" << std::endl;
            std::cout << "  " << argv[0] << "                          # Run with defaults" << std::endl;
            std::cout << "  " << argv[0] << " -v                      # Verbose mode" << std::endl;
            std::cout << "  " << argv[0] << " --beta 0.5              # Set beta=0.5" << std::endl;
            std::cout << "  " << argv[0] << " --dt 1e-2               # Set dt=1e-2" << std::endl;
            std::cout << "  " << argv[0] << " --chi-max 200           # Set chi_max=200" << std::endl;
            std::cout << "  " << argv[0] << " --svd-min 1e-10         # Set svd_min=1e-10" << std::endl;
            std::cout << "  " << argv[0] << " 1e-8                    # Set threshold=1e-8" << std::endl;
            std::cout << "  " << argv[0] << " -v -b 0.5 --dt 1e-2 1e-8  # Multiple options" << std::endl;
            return 0;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--beta" || arg == "-b") {
            if (i + 1 < argc) {
                beta = std::stod(argv[++i]);
            } else {
                std::cerr << "Error: --beta requires a value" << std::endl;
                return 1;
            }
        } else if (arg == "--dt" || arg == "-d") {
            if (i + 1 < argc) {
                dt = std::stod(argv[++i]);
            } else {
                std::cerr << "Error: --dt requires a value" << std::endl;
                return 1;
            }
        } else if (arg == "--chi-max") {
            if (i + 1 < argc) {
                chi_max = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --chi-max requires a value" << std::endl;
                return 1;
            }
        } else if (arg == "--svd-min") {
            if (i + 1 < argc) {
                svd_min = std::stod(argv[++i]);
            } else {
                std::cerr << "Error: --svd-min requires a value" << std::endl;
                return 1;
            }
        } else {
            threshold = std::stod(arg);
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "     Simulation Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Threshold: " << std::scientific << threshold << std::endl;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Trotter dt: " << dt << std::endl;
    std::cout << "Chi max: " << chi_max << std::endl;
    std::cout << "SVD min: " << svd_min << std::endl;
    std::cout << "Verbose: " << (verbose ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    int total = 0, passed_count = 0;

    // Test 1: ED against analytical
    auto [passed1, log1] = test_ED_analytical(threshold);
    total++;
    if (passed1) {
        passed_count++;
        if (verbose) {
            std::cout << "PASSED TEST 1: test_ED_analytical" << std::endl;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << log1;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << std::endl;
        }
    } else {
        std::cout << "FAILED TEST 1: test_ED_analytical" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log1;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 2: DMRG against ED
    std::vector<std::string> test2_ops = {"IIIII", "ZIIII", "ZZIII"};
    auto [passed2, log2] = test_dmrg_against_ED(threshold, "classical_ising_ferro.yml", 5, {{"g", 0.1}}, test2_ops);
    total++;
    if (passed2) {
        passed_count++;
        if (verbose) {
            std::cout << "PASSED TEST 2: test_dmrg_against_ED" << std::endl;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << log2;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << std::endl;
        }
    } else {
        std::cout << "FAILED TEST 2: test_dmrg_against_ED" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log2;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 3: Purification vs ED
    ThermalParams thermal_params3;
    thermal_params3.dt = dt;
    thermal_params3.order = 1;
    thermal_params3.chi_max = chi_max;
    thermal_params3.svd_min = svd_min;
    thermal_params3.quiet = true;

    std::vector<std::string> test3_ops = {"IIIII", "ZIIII", "ZZIII", "XXIII", "XXZII" };
    auto [passed3, log3] = test_purification_vs_ED(threshold, "classical_ising_ferro.yml", 5, beta, thermal_params3,
                                                     false, false, {{"g", -0.01}}, test3_ops);
    total++;
    if (passed3) {
        passed_count++;
        if (verbose) {
            std::cout << "PASSED TEST 3: test_purification_vs_ED" << std::endl;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << log3;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << std::endl;
        }
    } else {
        std::cout << "FAILED TEST 3: test_purification_vs_ED" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log3;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 4: Naive expectations
    ThermalParams thermal_params4;
    thermal_params4.dt = dt;
    thermal_params4.order = 1;
    thermal_params4.chi_max = chi_max;
    thermal_params4.svd_min = svd_min;
    thermal_params4.quiet = true;

    std::vector<std::string> test4_ops = {"IIIII", "ZIIII", "ZZIII", "XXIII", "XXZII" };
    auto [passed4, log4] = test_expectations_naive(threshold, "classical_ising_ferro.yml", 5, beta, thermal_params4,
                                                     false, false, {{"g", -0.1}}, test4_ops);
    total++;
    if (passed4) {
        passed_count++;
        if (verbose) {
            std::cout << "PASSED TEST 4: test_expectations_naive" << std::endl;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << log4;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << std::endl;
        }
    } else {
        std::cout << "FAILED TEST 4: test_expectations_naive" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log4;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 5: DFS expectations (pure state)
    auto [passed5, log5] = test_expectations_dfs("classical_ising_ferro.yml", 5, {{"g", -0.1}}, threshold);
    total++;
    if (passed5) {
        passed_count++;
        if (verbose) {
            std::cout << "PASSED TEST 5: test_expectations_dfs" << std::endl;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << log5;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << std::endl;
        }
    } else {
        std::cout << "FAILED TEST 5: test_expectations_dfs" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log5;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 6: DFS expectations (mixed state)
    auto [passed6, log6] = test_expectations_dfs_mixed(threshold, beta, dt);
    total++;
    if (passed6) {
        passed_count++;
        if (verbose) {
            std::cout << "PASSED TEST 6: test_expectations_dfs_mixed" << std::endl;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << log6;
            std::cout << std::string(60, '-') << std::endl;
            std::cout << std::endl;
        }
    } else {
        std::cout << "FAILED TEST 6: test_expectations_dfs_mixed" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log6;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Print summary
    std::cout << "========================================" << std::endl;
    if (passed_count == total) {
        std::cout << "✓ All " << total << " tests PASSED" << std::endl;
    } else {
        std::cout << "✗ " << (total - passed_count) << "/" << total << " tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return (passed_count == total) ? 0 : 1;
}
