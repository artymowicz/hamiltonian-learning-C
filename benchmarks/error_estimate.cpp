#include "simulation.hpp"
#include "pauli_utils.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <cmath>
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

int main(int argc, char* argv[]) {
    tprint("=== Error Estimation: ED vs Purification ===");
    tprint("");

    // Parse command line arguments
    if (argc < 2) {
        tprintf("Usage: %s <setup_file> [n] [\"op1\" \"op2\" ...]", argv[0]);
        tprintf("Example: %s setup.yml", argv[0]);
        tprintf("Example: %s setup.yml 12", argv[0]);
        tprintf("Example: %s setup.yml 10 \"Z0 Z1\" \"X5\"", argv[0]);
        tprint("Note: n defaults to 10 for ED comparison (ED becomes slow for n > 12)");
        tprint("Note: If operators are provided, only those are compared (skips threebody generation)");
        return 1;
    }

    std::string setup_file = argv[1];

    // System size (default to 10 if not provided)
    int n = 10;
    if (argc >= 3) {
        // Check if argv[2] looks like a number or an operator
        std::string arg2 = argv[2];
        bool is_number = !arg2.empty() && (std::isdigit(arg2[0]) || arg2[0] == '-');

        if (is_number) {
            n = std::stoi(argv[2]);
        }
    }

    // Parse custom operators if provided
    std::vector<std::string> custom_operators_raw;
    int first_op_idx = (argc >= 3 && std::isdigit(std::string(argv[2])[0])) ? 3 : 2;

    for (int i = first_op_idx; i < argc; ++i) {
        custom_operators_raw.push_back(argv[i]);
    }

    // Load setup parameters
    tprintf("Loading setup: %s", setup_file.c_str());
    YAML::Node setup_config = YAML::LoadFile("../setup/" + setup_file);

    // Thermal state parameters
    double dt = setup_config["dt"].as<double>();
    double svd_min = setup_config["svd_min"].as<double>();
    int chi_max = setup_config["chi_max"].as<int>();
    bool naive = setup_config["naive"].as<bool>();
    bool quiet = setup_config["quiet"].as<bool>();

    // Expectations computation parameters
    int expectations_n_threads = setup_config["expectations_n_threads"] ?
                                  setup_config["expectations_n_threads"].as<int>() : 0;

    // System parameters
    std::string hamiltonian_file = setup_config["hamiltonian"].as<std::string>();
    double beta = setup_config["beta"].as<double>();

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
    tprintf("  dt=%g", dt);
    tprintf("  svd_min=%g", svd_min);
    tprintf("  chi_max=%d", chi_max);
    tprintf("  naive=%s", naive ? "true" : "false");
    tprintf("  quiet=%s", quiet ? "true" : "false");
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

    // Determine which operators to use
    std::vector<std::string> operators_to_test;

    if (!custom_operators_raw.empty()) {
        // Use custom operators
        tprintf("Using %zu custom operators provided via command line", custom_operators_raw.size());
        for (const auto& op_str : custom_operators_raw) {
            operators_to_test.push_back(parseHumanReadablePauli(op_str, n));
        }
        tprint("");
    } else {
        // Generate threebody operators (same as experiment.cpp)
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

        operators_to_test = threebody_operators;
    }

    // Sort operators in REVERSE alphabetical order for DFS optimization
    // This is CRITICAL for correctness of the DFS algorithm!
    std::sort(operators_to_test.begin(), operators_to_test.end(), std::greater<std::string>());

    // Compute ED thermal state
    tprint("Computing thermal state using Exact Diagonalization...");
    auto rho_ED = computeEquilibriumStateED(H, beta);
    tprint("ED thermal state computed successfully");

    // Compute expectations using ED
    tprint("Computing expectations using ED...");
    std::vector<double> expectations_ED;
    expectations_ED.reserve(operators_to_test.size());

    for (size_t i = 0; i < operators_to_test.size(); ++i) {
        expectations_ED.push_back(computeExpectation(operators_to_test[i], rho_ED));

        // Print progress every 10% or at least every 100 operators
        size_t interval = std::max(size_t(1), std::min(operators_to_test.size() / 10, size_t(100)));
        if ((i + 1) % interval == 0 || i == operators_to_test.size() - 1) {
            printf("\r  Progress: %zu / %zu (%.1f%%)    ",
                   i + 1, operators_to_test.size(),
                   100.0 * (i + 1) / operators_to_test.size());
            fflush(stdout);
        }
    }
    printf("\n");  // Final newline after progress
    tprintf("Computed %zu expectations using ED", expectations_ED.size());
    tprint("");

    // Compute purified thermal state
    tprint("Computing thermal state using Purification...");
    tprintf("  beta = %g", beta);
    tprintf("  dt = %g", dt);
    tprintf("  svd_min = %g", svd_min);
    tprintf("  chi_max = %d", chi_max);
    ThermalParams thermal_params;
    thermal_params.dt = dt;
    thermal_params.svd_min = svd_min;
    thermal_params.chi_max = chi_max;
    thermal_params.quiet = quiet;
    auto psi = computePurifiedEquilibriumState(H, beta, thermal_params);
    tprint("Purified thermal state computed successfully");

    // Print all bond dimensions
    std::string bond_dims_str = "  Bond dimensions: [";
    for (int b = 1; b < length(psi); ++b) {
        bond_dims_str += std::to_string(dim(linkIndex(psi, b)));
        if (b < length(psi) - 1) bond_dims_str += ", ";
    }
    bond_dims_str += "]";
    tprint(bond_dims_str.c_str());
    tprintf("  Maximum bond dimension: %d", maxLinkDim(psi));
    tprint("");

    // Compute expectations using purification
    tprint("Computing expectations using Purification...");
    std::vector<double> expectations_purified;
    if (naive) {
        tprint("  Using naive method (simple loop)");
        expectations_purified = computeExpectationsMixedNaive(psi, operators_to_test);
    } else if (expectations_n_threads == 1) {
        tprint("  Using single-threaded DFS method");
        expectations_purified = computeExpectationsMixed(psi, operators_to_test);
    } else {
        if (expectations_n_threads == 0) {
            tprint("  Using DFS-optimized parallel method (all cores)");
        } else {
            tprintf("  Using DFS-optimized parallel method (%d threads)", expectations_n_threads);
        }
        expectations_purified = computeExpectationsMixedParallel(psi, operators_to_test, expectations_n_threads);
    }
    tprintf("Computed %zu expectations using Purification", expectations_purified.size());
    tprint("");

    // Compute error statistics
    tprint("=== Error Analysis ===");
    tprint("");

    std::vector<double> abs_errors;
    std::vector<double> rel_errors;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    std::string max_abs_error_op;
    std::string max_rel_error_op;

    for (size_t i = 0; i < operators_to_test.size(); ++i) {
        double ed_val = expectations_ED[i];
        double pur_val = expectations_purified[i];
        double abs_err = std::abs(pur_val - ed_val);
        double rel_err = (std::abs(ed_val) > 1e-14) ? abs_err / std::abs(ed_val) : 0.0;

        abs_errors.push_back(abs_err);
        rel_errors.push_back(rel_err);

        if (abs_err > max_abs_error) {
            max_abs_error = abs_err;
            max_abs_error_op = operators_to_test[i];
        }

        if (rel_err > max_rel_error) {
            max_rel_error = rel_err;
            max_rel_error_op = operators_to_test[i];
        }
    }

    // Compute mean and RMSE
    double mean_abs_error = 0.0;
    double mean_rel_error = 0.0;
    for (size_t i = 0; i < abs_errors.size(); ++i) {
        mean_abs_error += abs_errors[i];
        mean_rel_error += rel_errors[i];
    }
    mean_abs_error /= abs_errors.size();
    mean_rel_error /= rel_errors.size();

    double rmse = 0.0;
    for (double err : abs_errors) {
        rmse += err * err;
    }
    rmse = std::sqrt(rmse / abs_errors.size());

    // Print summary statistics
    tprint("Error Statistics:");
    tprintf("  Total operators compared: %zu", operators_to_test.size());
    tprint("");
    tprintf("  Maximum absolute error: %.6e", max_abs_error);
    tprintf("    Operator: %s", max_abs_error_op.c_str());
    tprintf("    ED value:  %.6e", expectations_ED[std::distance(operators_to_test.begin(),
                                                  std::find(operators_to_test.begin(),
                                                           operators_to_test.end(),
                                                           max_abs_error_op))]);
    tprintf("    Pur value: %.6e", expectations_purified[std::distance(operators_to_test.begin(),
                                                         std::find(operators_to_test.begin(),
                                                                  operators_to_test.end(),
                                                                  max_abs_error_op))]);
    tprint("");
    tprintf("  Maximum relative error: %.6e (%.3f%%)", max_rel_error, max_rel_error * 100);
    tprintf("    Operator: %s", max_rel_error_op.c_str());
    tprint("");
    tprintf("  Mean absolute error: %.6e", mean_abs_error);
    tprintf("  Mean relative error: %.6e (%.3f%%)", mean_rel_error, mean_rel_error * 100);
    tprintf("  RMSE: %.6e", rmse);
    tprint("");

    // Show histogram of errors
    tprint("Error Distribution:");
    std::vector<int> error_bins(5, 0);  // 5 bins: <1e-10, <1e-8, <1e-6, <1e-4, >=1e-4
    for (double err : abs_errors) {
        if (err < 1e-10) error_bins[0]++;
        else if (err < 1e-8) error_bins[1]++;
        else if (err < 1e-6) error_bins[2]++;
        else if (err < 1e-4) error_bins[3]++;
        else error_bins[4]++;
    }

    tprintf("  < 1e-10: %d operators (%.1f%%)", error_bins[0], 100.0 * error_bins[0] / abs_errors.size());
    tprintf("  < 1e-08: %d operators (%.1f%%)", error_bins[1], 100.0 * error_bins[1] / abs_errors.size());
    tprintf("  < 1e-06: %d operators (%.1f%%)", error_bins[2], 100.0 * error_bins[2] / abs_errors.size());
    tprintf("  < 1e-04: %d operators (%.1f%%)", error_bins[3], 100.0 * error_bins[3] / abs_errors.size());
    tprintf("  >= 1e-04: %d operators (%.1f%%)", error_bins[4], 100.0 * error_bins[4] / abs_errors.size());
    tprint("");

    // Show top 10 worst errors
    tprint("Top 10 Operators by Absolute Error:");
    tprint("  Rank | Operator   | ED Value      | Pur Value     | Abs Error     | Rel Error");
    tprint("  -----|------------|---------------|---------------|---------------|------------");

    // Create sorted indices by absolute error (descending)
    std::vector<size_t> sorted_indices(abs_errors.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&abs_errors](size_t i1, size_t i2) { return abs_errors[i1] > abs_errors[i2]; });

    for (size_t rank = 0; rank < std::min(size_t(10), sorted_indices.size()); ++rank) {
        size_t idx = sorted_indices[rank];
        tprintf("  %4zu | %s | %13.6e | %13.6e | %13.6e | %.3f%%",
                rank + 1,
                operators_to_test[idx].c_str(),
                expectations_ED[idx],
                expectations_purified[idx],
                abs_errors[idx],
                rel_errors[idx] * 100);
    }
    tprint("");

    tprint("✓ Error estimation completed successfully!");

    return 0;
}
