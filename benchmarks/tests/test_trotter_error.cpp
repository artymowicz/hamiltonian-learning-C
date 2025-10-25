#include "simulation.hpp"
#include "utils.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <yaml-cpp/yaml.h>

using namespace hamiltonian_learning;

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "    Trotter Error Analysis Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <setup_file.yml>" << std::endl;
        std::cerr << "Example: " << argv[0] << " trotter_error_test_setup.yml" << std::endl;
        return 1;
    }

    std::string setup_file = argv[1];

    // Load setup parameters
    std::cout << "Loading setup file: " << setup_file << std::endl;
    YAML::Node config = YAML::LoadFile("../setup/" + setup_file);

    int n = config["n"].as<int>();
    std::string hamiltonian_file = config["hamiltonian"].as<std::string>();
    double beta = config["beta"].as<double>();

    // Load couplings
    std::map<std::string, double> couplings;
    if (config["couplings"]) {
        for (const auto& kv : config["couplings"]) {
            couplings[kv.first.as<std::string>()] = kv.second.as<double>();
        }
    }

    // Load operators
    std::vector<std::string> operators;
    if (config["operators"]) {
        for (const auto& op : config["operators"]) {
            operators.push_back(op.as<std::string>());
        }
    }

    // Sort operators in reverse alphabetical order for DFS optimization
    std::sort(operators.begin(), operators.end(), std::greater<std::string>());

    // Load dts
    std::vector<double> dts;
    if (config["dts"]) {
        for (const auto& dt : config["dts"]) {
            dts.push_back(dt.as<double>());
        }
    }

    // Load svd_mins
    std::vector<double> svd_mins;
    if (config["svd_mins"]) {
        for (const auto& svd_min : config["svd_mins"]) {
            svd_mins.push_back(svd_min.as<double>());
        }
    }

    // Load chi_maxs
    std::vector<int> chi_maxs;
    if (config["chi_maxs"]) {
        for (const auto& chi_max : config["chi_maxs"]) {
            chi_maxs.push_back(chi_max.as<int>());
        }
    }

    // Print configuration
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  System size: n = " << n << std::endl;
    std::cout << "  Hamiltonian: " << hamiltonian_file << std::endl;
    std::cout << "  Beta: " << beta << std::endl;
    std::cout << "  Couplings: ";
    for (const auto& [name, value] : couplings) {
        std::cout << name << "=" << value << " ";
    }
    std::cout << std::endl;
    std::cout << "  Number of operators: " << operators.size() << std::endl;
    std::cout << "  Number of dt values: " << dts.size() << std::endl;
    std::cout << "  Number of svd_min values: " << svd_mins.size() << std::endl;
    std::cout << "  Number of chi_max values: " << chi_maxs.size() << std::endl;
    std::cout << "  Total runs: " << dts.size() * svd_mins.size() * chi_maxs.size() << std::endl;
    std::cout << std::endl;

    // Load Hamiltonian
    std::cout << "Loading Hamiltonian..." << std::endl;
    Hamiltonian H = Hamiltonian::loadFromYAML(n, "../hamiltonians/" + hamiltonian_file, couplings);
    std::cout << "Hamiltonian loaded: " << H.terms.size() << " terms" << std::endl;
    std::cout << std::endl;

    // Compute exact expectations using ED
    std::cout << "Computing exact expectations using ED..." << std::endl;
    MatrixXcd rho_ed = computeEquilibriumStateED(H, beta);
    std::vector<double> exp_ed = computeExpectationsED(rho_ed, operators);
    std::cout << "ED completed successfully!" << std::endl;
    std::cout << std::endl;

    // Print exact values
    std::cout << "Exact expectation values (ED):" << std::endl;
    std::cout << "  Operator      Value" << std::endl;
    std::cout << "  ------------  ----------------" << std::endl;
    for (size_t i = 0; i < operators.size(); ++i) {
        std::cout << "  " << std::setw(12) << std::left << operators[i] << "  ";
        std::cout << std::scientific << std::setprecision(10) << exp_ed[i] << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Open output file
    std::string output_file = "trotter_error_results.txt";
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return 1;
    }

    // Write header with metadata
    outfile << "# Trotter Error Test Results" << std::endl;
    outfile << "# n = " << n << std::endl;
    outfile << "# hamiltonian = " << hamiltonian_file << std::endl;
    outfile << "# beta = " << beta << std::endl;
    outfile << "# couplings: ";
    for (const auto& [name, value] : couplings) {
        outfile << name << "=" << value << " ";
    }
    outfile << std::endl;
    outfile << "#" << std::endl;
    outfile << "# ED results (exact):" << std::endl;
    for (size_t i = 0; i < operators.size(); ++i) {
        outfile << "# " << operators[i] << "," << std::scientific
                << std::setprecision(10) << exp_ed[i] << std::endl;
    }
    outfile << "#" << std::endl;
    outfile << "# Data columns: dt,svd_min,chi_max,max_bond_dim,max_error" << std::endl;
    outfile << "dt,svd_min,chi_max,max_bond_dim,max_error" << std::endl;

    // Loop over all (dt, svd_min, chi_max) combinations
    std::cout << "========================================" << std::endl;
    std::cout << "Testing all (dt, svd_min, chi_max) combinations" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    for (double dt : dts) {
        for (double svd_min : svd_mins) {
            for (int chi_max : chi_maxs) {
                std::cout << "----------------------------------------" << std::endl;
                std::cout << "dt = " << std::scientific << std::setprecision(1) << dt;
                std::cout << ",  svd_min = ";
                if (svd_min == 0.0) {
                    std::cout << "0";
                } else {
                    std::cout << std::scientific << std::setprecision(1) << svd_min;
                }
                std::cout << ",  chi_max = " << chi_max;
                std::cout << std::endl;
                std::cout << "----------------------------------------" << std::endl;

                // Set up thermal parameters
                ThermalParams thermal_params;
                thermal_params.dt = dt;
                thermal_params.order = 1;
                thermal_params.chi_max = chi_max;
                thermal_params.svd_min = svd_min;
                thermal_params.quiet = true;

            // Compute thermal state
            itensor::MPS psi_thermal = computePurifiedEquilibriumState(H, beta, thermal_params);
            int max_bond_dim = itensor::maxLinkDim(psi_thermal);
            std::cout << "Thermal state computed, bond dim: " << max_bond_dim << std::endl;

            // Compute expectations
            std::cout << "Computing expectations..." << std::endl;
            std::vector<double> exp_mps = computeExpectationsMixed(psi_thermal, operators);
            std::cout << "Expectations computed successfully!" << std::endl;

            // Print results
            std::cout << "Max bond dimension: " << max_bond_dim << std::endl;
            std::cout << std::endl;
            std::cout << "Operator      ED                MPS               Error" << std::endl;
            std::cout << "------------  ----------------  ----------------  ----------------" << std::endl;

            double max_error = 0.0;
            for (size_t i = 0; i < operators.size(); ++i) {
                double error = std::abs(exp_mps[i] - exp_ed[i]);
                max_error = std::max(max_error, error);

                std::cout << std::setw(12) << std::left << operators[i] << "  ";
                std::cout << std::scientific << std::setprecision(10) << exp_ed[i] << "  ";
                std::cout << std::scientific << std::setprecision(10) << exp_mps[i] << "  ";
                std::cout << std::scientific << std::setprecision(4) << error;
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << "Maximum error: " << std::scientific << std::setprecision(4) << max_error << std::endl;
            std::cout << std::endl;

            // Write result to file
            outfile << std::scientific << std::setprecision(10);
            outfile << dt << "," << svd_min << "," << chi_max << ","
                    << max_bond_dim << "," << max_error << std::endl;
            }
        }
    }

    outfile.close();
    std::cout << "Results written to: " << output_file << std::endl;
    std::cout << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "Test completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
