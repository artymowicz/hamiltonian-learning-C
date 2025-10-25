#include "simulation.hpp"
#include "pauli_utils.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <chrono>

using namespace hamiltonian_learning;

// Utility printing functions (duplicated here since this file won't be in GitHub release)
void tprint(const std::string& s, bool flush = true) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t_now);

    std::cout << std::put_time(&tm, "%H:%M:%S") << ": " << s;
    if (flush) {
        std::cout << std::flush;
    }
    std::cout << std::endl;
}

template<typename... Args>
void tprintf(const char* format, Args... args) {
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), format, args...);
    tprint(std::string(buffer));
}

// Helper function: adjust dt to be commensurate with beta/2
double adjustDtToBeCommensurate(double dt_desired, double beta) {
    double ttotal = beta / 2.0;
    double num_steps_desired = ttotal / dt_desired;
    double num_steps_rounded = std::round(num_steps_desired);

    // Ensure at least 1 step
    if (num_steps_rounded < 1) {
        num_steps_rounded = 1;
    }

    double dt_adjusted = ttotal / num_steps_rounded;
    return dt_adjusted;
}

// Helper function: compute max absolute error between two vectors
double computeMaxAbsoluteError(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Vectors must have the same size");
    }

    double max_error = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        double error = std::abs(vec1[i] - vec2[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}

int main(int argc, char* argv[]) {
    tprint("=== Automatic Parameter Optimization ===");
    tprint("");

    // Parse command line arguments
    if (argc < 3) {
        tprintf("Usage: %s <setup_file> <optimize_setup_file>", argv[0]);
        tprintf("Example: %s setup.yml optimize_parameters_setup.yml", argv[0]);
        tprint("");
        tprint("The setup_file contains hamiltonian, beta, couplings, etc.");
        tprint("The optimize_setup_file contains target_error, n, algorithm parameters, etc.");
        return 1;
    }

    std::string setup_file = argv[1];
    std::string optimize_setup_file = argv[2];

    // Load optimization parameters
    tprintf("Loading optimization setup: %s", optimize_setup_file.c_str());
    YAML::Node opt_config = YAML::LoadFile("../setup/" + optimize_setup_file);

    double target_error = opt_config["target_error"].as<double>();
    int n = opt_config["n"].as<int>();
    double svd_min_step = opt_config["svd_min_step"].as<double>();
    double dt_step = opt_config["dt_step"].as<double>();
    double initial_dt = opt_config["initial_dt"].as<double>();
    double initial_svd_min = opt_config["initial_svd_min"].as<double>();
    int chi_max = opt_config["chi_max"].as<int>();

    // Algorithm parameters
    const int MAX_ITERATIONS = 100;

    // Load setup parameters
    tprintf("Loading setup: %s", setup_file.c_str());
    YAML::Node setup_config = YAML::LoadFile("../setup/" + setup_file);

    // System parameters
    std::string hamiltonian_file = setup_config["hamiltonian"].as<std::string>();
    double beta = setup_config["beta"].as<double>();

    // Expectations computation parameters
    int expectations_n_threads = setup_config["expectations_n_threads"] ?
                                  setup_config["expectations_n_threads"].as<int>() : 0;

    // Couplings from YAML
    std::map<std::string, double> couplings;
    if (setup_config["couplings"]) {
        for (const auto& kv : setup_config["couplings"]) {
            couplings[kv.first.as<std::string>()] = kv.second.as<double>();
        }
    }

    // Print configuration
    tprint("Configuration:");
    tprintf("  System size: n=%d", n);
    tprintf("  Hamiltonian: %s", hamiltonian_file.c_str());
    tprintf("  Inverse temperature: beta=%g", beta);
    for (const auto& [name, value] : couplings) {
        tprintf("  Coupling %s=%g", name.c_str(), value);
    }
    tprint("");
    tprint("Optimization parameters:");
    tprintf("  Target error: %g", target_error);
    tprintf("  Initial dt: %g", initial_dt);
    tprintf("  Initial svd_min: %.2e", initial_svd_min);
    tprintf("  svd_min_step: %g", svd_min_step);
    tprintf("  dt_step: %g", dt_step);
    tprintf("  chi_max: %d", chi_max);
    tprintf("  Max iterations: %d", MAX_ITERATIONS);
    if (expectations_n_threads == 0) {
        tprint("  expectations_n_threads=0 (all cores)");
    } else if (expectations_n_threads == 1) {
        tprint("  expectations_n_threads=1 (single-threaded DFS)");
    } else {
        tprintf("  expectations_n_threads=%d", expectations_n_threads);
    }
    tprint("");

    // Load Hamiltonian
    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/" + hamiltonian_file, couplings);
    tprintf("Hamiltonian has %zu terms", H.terms.size());
    tprint("");

    // Generate threebody operators
    int k = 2;  // 2-local operators
    int k_H = 2;  // 2-local Hamiltonian terms
    bool periodic = false;

    tprintf("Generating %d-local operators...", k);
    auto onebody_operators = buildKLocalPaulis1D(n, k, periodic);
    tprintf("Number of onebody operators: %zu", onebody_operators.size());

    tprintf("Generating %d-local Hamiltonian terms...", k_H);
    auto hamiltonian_terms_with_identity = buildKLocalPaulis1D(n, k_H, periodic);

    // Remove identity from Hamiltonian terms
    std::vector<std::string> hamiltonian_terms;
    std::string identity(n, 'I');
    for (const auto& term : hamiltonian_terms_with_identity) {
        if (term != identity) {
            hamiltonian_terms.push_back(term);
        }
    }
    tprintf("Number of Hamiltonian terms (without identity): %zu", hamiltonian_terms.size());

    // Build tensors to get threebody operators
    tprint("Building multiplication tensor...");
    auto [mult_tensor, twobody_operators] = buildMultiplicationTensor(onebody_operators);
    tprintf("Multiplication tensor built: %zu nonzero entries", mult_tensor.values.size());

    tprint("Building triple product tensor...");
    auto [triple_product_tensor, threebody_operators] = buildTripleProductTensor(onebody_operators, hamiltonian_terms);
    tprintf("Triple product tensor built: %zu nonzero entries", triple_product_tensor.values.size());
    tprintf("Number of threebody operators: %zu", threebody_operators.size());
    tprint("");

    // Sort operators in REVERSE alphabetical order for DFS optimization
    std::sort(threebody_operators.begin(), threebody_operators.end(), std::greater<std::string>());

    // Compute ED thermal state (once, at the beginning)
    tprint("Computing thermal state using Exact Diagonalization...");
    auto rho_ED = computeEquilibriumStateED(H, beta);
    tprint("ED thermal state computed successfully");

    // Compute expectations using ED (once, at the beginning)
    tprint("Computing expectations using ED...");
    std::vector<double> expectations_ED;
    expectations_ED.reserve(threebody_operators.size());

    for (size_t i = 0; i < threebody_operators.size(); ++i) {
        expectations_ED.push_back(computeExpectation(threebody_operators[i], rho_ED));

        // Print progress every 10% or at least every 100 operators
        size_t interval = std::max(size_t(1), std::min(threebody_operators.size() / 10, size_t(100)));
        if ((i + 1) % interval == 0 || i == threebody_operators.size() - 1) {
            printf("\r  Progress: %zu / %zu (%.1f%%)    ",
                   i + 1, threebody_operators.size(),
                   100.0 * (i + 1) / threebody_operators.size());
            fflush(stdout);
        }
    }
    printf("\r                                        \r");  // Clear the progress line
    tprintf("Computed %zu expectations using ED", expectations_ED.size());
    tprint("");

    // Initialize parameters from config
    double dt = adjustDtToBeCommensurate(initial_dt, beta);
    double svd_min = initial_svd_min;

    // Compute initial error
    tprint("=== Starting Optimization ===");
    tprint("");
    ThermalParams thermal_params;
    thermal_params.svd_min = svd_min;
    thermal_params.chi_max = chi_max;
    thermal_params.quiet = true;  // Always quiet to avoid progress messages cluttering the table
    thermal_params.dt = dt;

    auto psi = computePurifiedEquilibriumState(H, beta, thermal_params);
    std::vector<double> expectations_pur;
    if (expectations_n_threads == 1) {
        expectations_pur = computeExpectationsMixed(psi, threebody_operators);
    } else {
        expectations_pur = computeExpectationsMixedParallel(psi, threebody_operators, expectations_n_threads);
    }
    double current_error = computeMaxAbsoluteError(expectations_ED, expectations_pur);

    printf("\r     \r");  // Clear the "100%" line from gateTEvol
    tprintf("Initial: dt=%.6g, svd_min=%.2e, error=%.6e", dt, svd_min, current_error);
    tprint("");

    // Check if already at target
    if (current_error <= target_error) {
        tprint("✓ Already at target error!");
        tprintf("Optimal parameters: dt=%.6g, svd_min=%.2e", dt, svd_min);
        return 0;
    }

    // Main optimization loop
    int iteration = 0;
    int consecutive_dt_decreases = 0;
    int max_chi = 0;  // Track max bond dimension

    // Track best parameters seen so far
    double best_dt = dt;
    double best_svd_min = svd_min;
    double best_error = current_error;

    tprint("Iter | dt       | svd_min   | max_chi | error");
    tprint("-----|----------|-----------|---------|------------");

    while (iteration < MAX_ITERATIONS && current_error > target_error) {
        // STATE: DECREASING_SVD_MIN
        bool svd_improving = true;
        while (svd_improving) {
            double svd_min_new = svd_min / svd_min_step;

            thermal_params.svd_min = svd_min_new;
            thermal_params.dt = dt;

            psi = computePurifiedEquilibriumState(H, beta, thermal_params);
            max_chi = maxLinkDim(psi);
            if (expectations_n_threads == 1) {
                expectations_pur = computeExpectationsMixed(psi, threebody_operators);
            } else {
                expectations_pur = computeExpectationsMixedParallel(psi, threebody_operators, expectations_n_threads);
            }
            double error_new = computeMaxAbsoluteError(expectations_ED, expectations_pur);

            iteration++;
            printf("\r     \r");  // Clear the "100%" line from gateTEvol
            tprintf("%4d | %.3e | %.2e | %7d | %.6e", iteration, dt, svd_min_new, max_chi, error_new);

            // Check improvement: truncation error ~ svd_min^(2/3)
            double improvement_threshold = current_error / std::pow(svd_min_step, 1.0/3.0);

            if (error_new < improvement_threshold) {
                // Improvement!
                svd_min = svd_min_new;
                current_error = error_new;
                consecutive_dt_decreases = 0;

                // Update best if this is better
                if (error_new < best_error) {
                    best_dt = dt;
                    best_svd_min = svd_min;
                    best_error = error_new;
                }

                if (current_error <= target_error) {
                    tprint("");
                    tprint("✓ Target error achieved!");
                    tprintf("Optimal parameters: dt=%.6g, svd_min=%.2e, error=%.6e", dt, svd_min, current_error);
                    return 0;
                }
            } else {
                // No improvement, but keep the new value and skip extra step
                svd_min = svd_min_new;
                current_error = error_new;

                // Update best if this is better
                if (error_new < best_error) {
                    best_dt = dt;
                    best_svd_min = svd_min;
                    best_error = error_new;
                }

                svd_improving = false;
            }
        }

        // STATE: SWITCHING TO DT
        // svd_min already updated to last tested value, no extra step needed
        double error_after_extra_svd = current_error;

        // STATE: DECREASING_DT
        bool dt_improving = true;
        while (dt_improving) {
            double dt_new = dt / dt_step;
            dt_new = adjustDtToBeCommensurate(dt_new, beta);

            thermal_params.svd_min = svd_min;
            thermal_params.dt = dt_new;

            psi = computePurifiedEquilibriumState(H, beta, thermal_params);
            max_chi = maxLinkDim(psi);
            if (expectations_n_threads == 1) {
                expectations_pur = computeExpectationsMixed(psi, threebody_operators);
            } else {
                expectations_pur = computeExpectationsMixedParallel(psi, threebody_operators, expectations_n_threads);
            }
            double error_new = computeMaxAbsoluteError(expectations_ED, expectations_pur);

            iteration++;
            printf("\r     \r");  // Clear the "100%" line from gateTEvol
            tprintf("%4d | %.3e | %.2e | %7d | %.6e", iteration, dt_new, svd_min, max_chi, error_new);

            // Check improvement: trotter error ~ dt^2
            double improvement_threshold = error_after_extra_svd / dt_step;

            if (error_new < improvement_threshold) {
                // Improvement!
                dt = dt_new;
                current_error = error_new;
                error_after_extra_svd = error_new;
                consecutive_dt_decreases++;

                // Update best if this is better
                if (error_new < best_error) {
                    best_dt = dt;
                    best_svd_min = svd_min;
                    best_error = error_new;
                }

                if (current_error <= target_error) {
                    tprint("");
                    tprint("✓ Target error achieved!");
                    tprintf("Optimal parameters: dt=%.6g, svd_min=%.2e, error=%.6e", dt, svd_min, current_error);
                    return 0;
                }
            } else {
                // No improvement
                if (consecutive_dt_decreases == 0) {
                    // First time decreasing dt - plateau detected
                    // Update best if this is better (even though not enough improvement)
                    if (error_new < best_error) {
                        best_dt = dt_new;
                        best_svd_min = svd_min;
                        best_error = error_new;
                    }

                    tprint("");
                    tprint("⚠ PLATEAU DETECTED!");
                    tprint("Error not improving with either dt or svd_min.");
                    tprint("Possible causes:");
                    tprint("  - Bond dimension saturated (truncation error plateaued)");
                    tprint("  - Already at numerical precision limit");
                    tprintf("Best found: dt=%.6g, svd_min=%.2e, error=%.6e", best_dt, best_svd_min, best_error);
                    tprint("Consider manual intervention or different target error.");
                    return 1;
                } else {
                    // No improvement, but keep the new value and skip extra step
                    dt = dt_new;
                    current_error = error_new;

                    // Update best if this is better
                    if (error_new < best_error) {
                        best_dt = dt;
                        best_svd_min = svd_min;
                        best_error = error_new;
                    }

                    dt_improving = false;
                }
            }
        }

        // STATE: SWITCHING TO SVD_MIN
        // dt already updated to last tested value, no extra step needed
        consecutive_dt_decreases = 0;

        if (current_error <= target_error) {
            tprint("");
            tprint("✓ Target error achieved!");
            tprintf("Optimal parameters: dt=%.6g, svd_min=%.2e, error=%.6e", dt, svd_min, current_error);
            return 0;
        }
    }

    // Reached max iterations or other stopping condition
    if (current_error > target_error) {
        tprint("⚠ WARNING: Maximum iterations reached without achieving target error");
        tprintf("Best found: dt=%.6g, svd_min=%.2e, error=%.6e", dt, svd_min, current_error);
        tprintf("Target was: %.6e", target_error);
        return 1;
    }

    return 0;
}
