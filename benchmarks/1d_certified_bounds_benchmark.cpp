#include "simulation.hpp"
#include "pauli_utils.hpp"
#include "hamiltonian_learning.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <yaml-cpp/yaml.h>

using namespace hamiltonian_learning;

void printResults(const std::vector<double>& a_list,
                 const std::vector<double>& b_list,
                 const std::vector<double>& true_values,
                 const std::vector<std::string>& descriptions,
                 double beta) {
    tprint("");
    tprint("Results:");
    tprint("  Operator      | True Value | a          | b          | (a-true)/beta | (b-true)/beta");
    tprint("  --------------|------------|------------|------------|---------------|---------------");

    for (size_t i = 0; i < descriptions.size(); ++i) {
        double a_error = (a_list[i] - beta * true_values[i]) / beta;
        double b_error = (b_list[i] - beta * true_values[i]) / beta;

        tprintf("  %-12s  | %.3e | %.3e | %.3e | %13.3e | %13.3e",
                descriptions[i].c_str(), beta * true_values[i], a_list[i], b_list[i], a_error, b_error);
    }
    tprint("");
}

int main(int argc, char* argv[]) {
    tprint("=== Hamiltonian Learning Experiment ===");
    tprint("");

    // Parse command line arguments
    if (argc < 2) {
        tprintf("Usage: %s <setup_file> [<hamiltonian_file>] [<n>] [<beta>] [<coupling_name1>=<value1> ...]", argv[0]);
        tprintf("Example: %s setup.yml", argv[0]);
        tprintf("Example: %s setup.yml classical_ising_ferro.yml 5 0.1 g=-0.01", argv[0]);
        tprint("Note: Command-line arguments override values in setup.yml");
        return 1;
    }

    std::string setup_file = argv[1];

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

    // Certified bounds parameters
    double epsilon_W = setup_config["epsilon_W"].as<double>();
    double eps_0 = setup_config["eps_0"].as<double>();
    int printing_level = setup_config["printing_level"] ? setup_config["printing_level"].as<int>() : 3;

    // System parameters (can be overridden by command line)
    std::string hamiltonian_file = setup_config["hamiltonian"].as<std::string>();
    int n = setup_config["n"].as<int>();
    double beta = setup_config["beta"].as<double>();

    // Couplings from YAML (will be merged with command line)
    std::map<std::string, double> couplings;
    if (setup_config["couplings"]) {
        for (const auto& kv : setup_config["couplings"]) {
            couplings[kv.first.as<std::string>()] = kv.second.as<double>();
        }
    }

    // Target operators for bounds computation (will be parsed after n is determined)
    std::vector<std::string> target_operators_raw;
    if (setup_config["target_operators"]) {
        for (const auto& op : setup_config["target_operators"]) {
            target_operators_raw.push_back(op.as<std::string>());
        }
    }

    // Override with command-line arguments
    if (argc >= 3) {
        hamiltonian_file = argv[2];
    }
    if (argc >= 4) {
        n = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        beta = std::stod(argv[4]);
    }

    // Parse coupling constants from command line (overrides YAML)
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq_pos = arg.find('=');
        if (eq_pos == std::string::npos) {
            tprintf("Invalid coupling format: %s (expected name=value)", arg.c_str());
            return 1;
        }
        std::string name = arg.substr(0, eq_pos);
        double value = std::stod(arg.substr(eq_pos + 1));
        couplings[name] = value;
    }

    // Parse target operators now that n is finalized
    std::vector<std::string> target_operators;
    for (const auto& op_str : target_operators_raw) {
        target_operators.push_back(parseHumanReadablePauli(op_str, n));
    }

    // Print configuration
    tprint("Configuration:");
    tprintf("  dt=%g", dt);
    tprintf("  svd_min=%g", svd_min);
    tprintf("  chi_max=%d", chi_max);
    tprintf("  epsilon_W=%g", epsilon_W);
    tprintf("  eps_0=%g", eps_0);
    tprintf("  printing_level=%d", printing_level);
    tprintf("  naive=%s", naive ? "true" : "false");
    tprintf("  quiet=%s", quiet ? "true" : "false");
    tprint("");

    tprint("System parameters:");
    tprintf("  Hamiltonian: %s", hamiltonian_file.c_str());
    tprintf("  System size: n=%d", n);
    tprintf("  Inverse temperature: beta=%g", beta);
    for (const auto& [name, value] : couplings) {
        tprintf("  Coupling %s=%g", name.c_str(), value);
    }
    tprint("");

    // Load Hamiltonian
    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/" + hamiltonian_file, couplings);
    tprintf("Hamiltonian has %zu terms", H.terms.size());
    tprint("");

    // Compute thermal state
    tprint("Computing thermal state...");
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
    tprint("Thermal state computed successfully");
    tprintf("  Maximum bond dimension: %d", maxLinkDim(psi));

    // Generate operators
    int k = 2;  // 2-local operators (to match Python setup)
    bool periodic = false;

    tprintf("Generating %d-local operators...", k);
    auto onebody_operators = buildKLocalPaulis1D(n, k, periodic);
    tprintf("Number of onebody operators: %zu", onebody_operators.size());

    // Generate Hamiltonian terms (variational basis) - use k_H-local Paulis
    int k_H = 2;  // Use 2-local Paulis as variational Hamiltonian basis
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

    // Build tensors
    tprint("Building multiplication tensor...");
    auto [mult_tensor, twobody_operators] = buildMultiplicationTensor(onebody_operators);
    tprintf("Multiplication tensor built: %zu nonzero entries", mult_tensor.values.size());

    tprint("Building triple product tensor...");
    auto [triple_product_tensor, threebody_operators] = buildTripleProductTensor(onebody_operators, hamiltonian_terms);
    tprintf("Triple product tensor built: %zu nonzero entries", triple_product_tensor.values.size());

    /*
    // Debug: print first 10 three-body operators
    std::cout << "First 10 three-body operators: ";
    for (size_t i = 0; i < std::min(size_t(10), threebody_operators.size()); ++i) {
        std::cout << threebody_operators[i] << " ";
    }
    std::cout << std::endl << std::endl;
    */

    // Build all_operators
    tprint("Building list of all operators...");
    std::vector<std::string> all_operators;
    all_operators.insert(all_operators.end(), hamiltonian_terms.begin(), hamiltonian_terms.end());
    all_operators.insert(all_operators.end(), threebody_operators.begin(), threebody_operators.end());

    // Remove duplicates - sort forward first, then reverse
    std::sort(all_operators.begin(), all_operators.end());
    all_operators.erase(std::unique(all_operators.begin(), all_operators.end()), all_operators.end());

    // Sort in REVERSE alphabetical order for DFS optimization
    std::sort(all_operators.begin(), all_operators.end(), std::greater<std::string>());

    // Compute all expectations
    tprintf("Computing expectations for %zu operators...", all_operators.size());
    std::vector<double> all_expectations;
    if (naive) {
        all_expectations = computeExpectationsMixedNaive(psi, all_operators);
    }
    else if (expectations_n_threads == 1) {
        all_expectations = computeExpectationsMixed(psi, all_operators);
    }
    else {
        all_expectations = computeExpectationsMixedParallel(psi, all_operators, expectations_n_threads);
    }

    tprint("Building expectation values dictionary");

    // Build expectation value dictionary
    std::map<std::string, double> expectations_dict;
    for (size_t i = 0; i < all_operators.size(); ++i) {
        expectations_dict[all_operators[i]] = all_expectations[i];
    }

    // Extract expectations for specific operator types
    tprint("Building expectations vectors");

    std::vector<double> twobody_expectations;
    for (const auto& op : twobody_operators) {
        twobody_expectations.push_back(expectations_dict[op]);
    }

    std::vector<double> hamiltonian_terms_expectations;
    for (const auto& term : hamiltonian_terms) {
        hamiltonian_terms_expectations.push_back(expectations_dict[term]);
    }

    std::vector<double> threebody_expectations;
    for (const auto& op : threebody_operators) {
        threebody_expectations.push_back(expectations_dict[op]);
    }

    // Contract to get covariance matrix C and tensor F
    tprint("Computing covariance matrix C...");
    VectorXc twobody_exp_vec = Eigen::Map<const Eigen::VectorXd>(twobody_expectations.data(), twobody_expectations.size()).cast<Complex>();
    auto C_tensor = mult_tensor.contractRight(twobody_exp_vec);
    MatrixXc C = C_tensor.toEigenDense();
    tprintf("C matrix: %ld x %ld", C.rows(), C.cols());

    // Transpose triple product tensor: we want Hamiltonian index to be second-last
    tprint("Computing F tensor...");
    auto triple_product_transposed = triple_product_tensor.transpose({0, 2, 1, 3});
    VectorXc threebody_exp_vec = Eigen::Map<const Eigen::VectorXd>(threebody_expectations.data(), threebody_expectations.size()).cast<Complex>();
    auto F = triple_product_transposed.contractRight(threebody_exp_vec);
    tprint("F tensor built");
    tprint("");

    // Compute certified bounds for target operators
    tprint("Computing certified bounds...");
    int r = onebody_operators.size();
    int s = hamiltonian_terms.size();
    MatrixXc J = MatrixXc::Identity(r, r);  // Self-adjoint operators

    VectorXd hamiltonian_exp_vec = Eigen::Map<const Eigen::VectorXd>(hamiltonian_terms_expectations.data(), hamiltonian_terms_expectations.size());

    // If no target operators specified, compute bounds for first Hamiltonian term
    if (target_operators.empty()) {
        tprint("Warning: No target operators specified in setup. Using first Hamiltonian term.");
        target_operators.push_back(hamiltonian_terms[0]);
    }

    // Storage for results
    std::vector<double> a_list, b_list, true_values;
    std::vector<std::string> descriptions;

    // Compute bounds for each target operator
    for (const auto& target_op : target_operators) {
        // Find target operator in hamiltonian_terms
        auto it = std::find(hamiltonian_terms.begin(), hamiltonian_terms.end(), target_op);
        if (it == hamiltonian_terms.end()) {
            tprintf("Warning: Target operator %s not found in Hamiltonian terms. Skipping.", target_op.c_str());
            continue;
        }
        size_t target_idx = std::distance(hamiltonian_terms.begin(), it);

        // Create indicator vector v for this operator
        VectorXd v = VectorXd::Zero(s);
        v(target_idx) = 1.0;

        tprint("");
        tprintf("Computing bounds for operator: %s", target_op.c_str());

        auto [a, b, mu_1, mu_2, kappa, nu] = certifiedBoundsV2(
            r, s,
            hamiltonian_exp_vec,
            J, C,
            F.indices, F.values,
            epsilon_W, eps_0, beta, v,
            printing_level
        );

        // Compute true value
        double true_value = 0.0;
        auto h_it = std::find(H.terms.begin(), H.terms.end(), target_op);
        if (h_it != H.terms.end()) {
            size_t h_idx = std::distance(H.terms.begin(), h_it);
            true_value = H.coefficients[h_idx];
        }

        // Store results
        a_list.push_back(a);
        b_list.push_back(b);
        true_values.push_back(true_value);
        descriptions.push_back(target_op);

        tprintf("True value: %.6e", beta * true_value);
        tprintf("Lower bound (a): %.6e", a);
        tprintf("Upper bound (b): %.6e", b);
        tprintf("Bound width: %.6e", b - a);
    }

    // Print summary if multiple operators
    if (target_operators.size() > 1) {
        printResults(a_list, b_list, true_values, descriptions, beta);
    }

    tprint("");
    tprint("✓ Experiment completed successfully!");

    return 0;
}
