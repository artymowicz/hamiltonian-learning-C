#include "simulation.hpp"
#include "pauli_utils.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <random>
#include <thread>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

namespace hamiltonian_learning {

// ============================================================================
// Hamiltonian implementation
// ============================================================================

Hamiltonian::Hamiltonian(int n, const std::vector<std::string>& terms,
                         const std::vector<double>& coeffs)
    : n(n), terms(terms), coefficients(coeffs) {
    if (terms.size() != coeffs.size()) {
        throw std::invalid_argument("Number of terms and coefficients must match");
    }
    sort();
}

void Hamiltonian::sort() {
    // Create pairs of (term, coefficient)
    std::vector<std::pair<std::string, double>> pairs;
    for (size_t i = 0; i < terms.size(); ++i) {
        pairs.push_back({terms[i], coefficients[i]});
    }

    // Sort by compressed Pauli representation
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) {
                  return compressPauliToList(a.first) < compressPauliToList(b.first);
              });

    // Extract sorted terms and coefficients
    for (size_t i = 0; i < pairs.size(); ++i) {
        terms[i] = pairs[i].first;
        coefficients[i] = pairs[i].second;
    }
}

void Hamiltonian::print(double threshold) const {
    int max_weight = 0;
    for (const auto& term : terms) {
        max_weight = std::max(max_weight, weight(term));
    }

    std::string header = "term" + std::string(4 * max_weight - 1, ' ') + ": coefficient";
    std::cout << header << std::endl;
    std::cout << std::string(header.length(), '-') << std::endl;

    for (size_t i = 0; i < terms.size(); ++i) {
        if (std::abs(coefficients[i]) > threshold) {
            std::string compressed = compressPauli(terms[i]);
            int padding = 4 * max_weight - compressed.length() + 2;
            std::cout << compressed << std::string(padding, ' ')
                     << ":  " << std::showpos << coefficients[i] << std::noshowpos
                     << std::endl;
        }
    }
}

double Hamiltonian::normalizeCoeffs(const std::map<std::string, double>& expectations_dict) {
    std::string identity = std::string(n, 'I');

    // Check if identity term exists, add it if not
    auto it = std::find(terms.begin(), terms.end(), identity);
    if (it == terms.end()) {
        terms.insert(terms.begin(), identity);
        coefficients.insert(coefficients.begin(), 1.0);
    }

    sort();

    // Compute normalization: identity coefficient = -sum(other_coeffs * expectations)
    double normalization = 0.0;
    for (size_t i = 1; i < terms.size(); ++i) {
        auto exp_it = expectations_dict.find(terms[i]);
        if (exp_it == expectations_dict.end()) {
            throw std::runtime_error("Expectation value not found for term: " + terms[i]);
        }
        normalization -= coefficients[i] * exp_it->second;
    }

    // Normalize all coefficients
    for (auto& coeff : coefficients) {
        coeff /= normalization;
    }

    return normalization;
}

std::vector<double> Hamiltonian::normalizedCoeffs(
    const std::map<std::string, double>& expectations_dict) const {

    std::string identity = std::string(n, 'I');
    std::vector<double> out = coefficients;

    // Find identity term
    auto it = std::find(terms.begin(), terms.end(), identity);
    if (it == terms.end()) {
        throw std::runtime_error("Identity term not found in Hamiltonian");
    }
    size_t identity_idx = std::distance(terms.begin(), it);

    // Compute identity coefficient
    double identity_coeff = 0.0;
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i != identity_idx) {
            auto exp_it = expectations_dict.find(terms[i]);
            if (exp_it == expectations_dict.end()) {
                throw std::runtime_error("Expectation value not found for term: " + terms[i]);
            }
            identity_coeff -= out[i] * exp_it->second;
        }
    }

    out[identity_idx] = identity_coeff;

    // Normalize
    for (auto& coeff : out) {
        coeff /= identity_coeff;
    }

    return out;
}

Hamiltonian Hamiltonian::restricted(const std::vector<int>& region) const {
    std::map<std::string, double> coefficients_dict;
    for (size_t i = 0; i < terms.size(); ++i) {
        coefficients_dict[terms[i]] = coefficients[i];
    }

    std::vector<std::string> terms_restricted = restrictOperators(n, terms, region);
    std::vector<double> coefficients_restricted;

    for (const auto& term : terms_restricted) {
        std::string embedded = embedPauli(n, term, region);
        coefficients_restricted.push_back(coefficients_dict[embedded]);
    }

    return Hamiltonian(region.size(), terms_restricted, coefficients_restricted);
}

void Hamiltonian::makeTranslationInvariant(bool periodic) {
    std::vector<std::string> new_terms = terms;
    std::set<std::string> new_terms_set(terms.begin(), terms.end());
    std::map<std::string, double> new_coeffs_dict;

    for (size_t i = 0; i < terms.size(); ++i) {
        new_coeffs_dict[terms[i]] = coefficients[i];
    }

    // Generate all translates
    for (size_t i = 0; i < terms.size(); ++i) {
        if (terms[i][0] == 'I') continue;  // Skip if starts with identity

        // Generate translates of this term
        int term_length = 0;
        for (int j = n - 1; j >= 0; --j) {
            if (terms[i][j] != 'I') {
                term_length = j + 1;
                break;
            }
        }

        int max_shift = periodic ? n : (n - term_length + 1);
        for (int shift = 1; shift < max_shift; ++shift) {
            std::string translated = std::string(n, 'I');
            for (int j = 0; j < n; ++j) {
                if (terms[i][j] != 'I') {
                    int new_pos = (j + shift) % n;
                    translated[new_pos] = terms[i][j];
                }
            }

            if (new_terms_set.find(translated) != new_terms_set.end()) {
                throw std::runtime_error("Encountered duplicate when adding translates: " + translated);
            }

            new_terms.push_back(translated);
            new_terms_set.insert(translated);
            new_coeffs_dict[translated] = coefficients[i];
        }
    }

    terms = new_terms;
    coefficients.clear();
    for (const auto& term : terms) {
        coefficients.push_back(new_coeffs_dict[term]);
    }

    sort();
}

void Hamiltonian::addDisorder(double magnitude,
                              const std::vector<std::string>& disorder_terms,
                              const std::string& distribution) {
    std::vector<std::string> terms_to_disorder = disorder_terms.empty() ? terms : disorder_terms;

    std::vector<double> disorder_coefficients;
    if (distribution == "normal") {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, magnitude);
        for (size_t i = 0; i < terms_to_disorder.size(); ++i) {
            disorder_coefficients.push_back(dist(gen));
        }
    } else {
        throw std::invalid_argument("Unknown distribution: " + distribution);
    }

    std::map<std::string, size_t> terms_indices;
    for (size_t i = 0; i < terms.size(); ++i) {
        terms_indices[terms[i]] = i;
    }

    std::set<std::string> disorder_terms_seen;
    for (size_t i = 0; i < terms_to_disorder.size(); ++i) {
        const auto& disorder_term = terms_to_disorder[i];

        if (disorder_terms_seen.find(disorder_term) != disorder_terms_seen.end()) {
            throw std::runtime_error("Found duplicate in disorder terms: " + disorder_term);
        }
        disorder_terms_seen.insert(disorder_term);

        auto it = terms_indices.find(disorder_term);
        if (it != terms_indices.end()) {
            coefficients[it->second] += disorder_coefficients[i];
        } else {
            terms.push_back(disorder_term);
            coefficients.push_back(disorder_coefficients[i]);
        }
    }

    sort();
}

Hamiltonian Hamiltonian::loadFromYAML(int n, const std::string& filename,
                                       const std::map<std::string, double>& couplings) {
    YAML::Node config = YAML::LoadFile(filename);

    if (config["n"]) {
        if (config["n"].as<int>() != n) {
            throw std::runtime_error("Hamiltonian n does not match requested n");
        }
    }

    auto term_coeff_pairs = config["term_coefficient_pairs"];

    std::vector<std::string> terms;
    std::vector<double> coeffs;

    for (const auto& pair : term_coeff_pairs) {
        std::string term_compressed = pair[0].as<std::string>();
        std::string term = decompressPauli(term_compressed, n);
        terms.push_back(term);

        // Handle coefficient
        // First try to parse as number directly
        try {
            coeffs.push_back(pair[1].as<double>());
        } catch (...) {
            // Not a number, treat as coupling name
            std::string coeff_str = pair[1].as<std::string>();

            // Check if it has a sign prefix
            if (coeff_str[0] == '-' || coeff_str[0] == '+') {
                double sign = (coeff_str[0] == '-') ? -1.0 : 1.0;
                std::string key = coeff_str.substr(1);
                auto it = couplings.find(key);
                if (it == couplings.end()) {
                    throw std::runtime_error("Coupling not found: " + key);
                }
                coeffs.push_back(sign * it->second);
            } else {
                // Coupling name without sign
                auto it = couplings.find(coeff_str);
                if (it == couplings.end()) {
                    throw std::runtime_error("Coupling not found: " + coeff_str);
                }
                coeffs.push_back(it->second);
            }
        }
    }

    Hamiltonian H(n, terms, coeffs);

    // Handle translation invariance
    if (config["translation_invariant"] && config["translation_invariant"].as<bool>()) {
        bool periodic = config["periodic"] ? config["periodic"].as<bool>() : false;
        H.makeTranslationInvariant(periodic);
    }

    return H;
}

void Hamiltonian::saveToYAML(const std::string& filename) const {
    std::string output_filename = filename;
    if (filename.length() < 4 || filename.substr(filename.length() - 4) != ".yml") {
        output_filename += ".yml";
    }

    YAML::Node config;
    config["n"] = n;

    YAML::Node pairs;
    for (size_t i = 0; i < terms.size(); ++i) {
        YAML::Node pair;
        pair.push_back(compressPauli(terms[i]));
        pair.push_back(coefficients[i]);
        pairs.push_back(pair);
    }
    config["term_coefficient_pairs"] = pairs;

    std::ofstream fout(output_filename);
    fout << config;
}

// ============================================================================
// State computation - iTensor implementation
// ============================================================================

std::pair<itensor::MPS, double> computeGroundState(const Hamiltonian& hamiltonian,
                                                     const DMRGParams& params) {
    using namespace itensor;

    int N = hamiltonian.n;

    // Create spin-1/2 sites (qubits)
    auto sites = SpinHalf(N, {"ConserveQNs=", false});

    // Build Hamiltonian using AutoMPO
    auto ampo = AutoMPO(sites);

    for (size_t i = 0; i < hamiltonian.terms.size(); ++i) {
        const auto& pauli_string = hamiltonian.terms[i];
        double coeff = hamiltonian.coefficients[i];

        // Parse Pauli string and add to AutoMPO
        // Note: iTensor's Sx, Sy, Sz are spin-1/2 operators (eigenvalues ±1/2)
        // Pauli matrices have eigenvalues ±1, so Pauli = 2 * S
        // We need to account for this factor of 2^k for k operators
        std::vector<std::pair<std::string, int>> ops;
        int num_ops = 0;
        for (int j = 0; j < N; ++j) {
            if (pauli_string[j] != 'I') {
                std::string op_name;
                if (pauli_string[j] == 'X') op_name = "Sx";
                else if (pauli_string[j] == 'Y') op_name = "Sy";
                else if (pauli_string[j] == 'Z') op_name = "Sz";
                ops.push_back({op_name, j + 1}); // iTensor uses 1-based indexing
                num_ops++;
            }
        }

        // Account for the factor of 2^num_ops from Pauli = 2*S convention
        double adjusted_coeff = coeff * std::pow(2.0, num_ops);

        // Add term to AutoMPO
        if (ops.empty()) {
            // Identity term - add as a constant energy shift
            // ITensor doesn't handle pure constants well, skip for now
            continue;
        } else if (ops.size() == 1) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second;
        } else if (ops.size() == 2) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second, ops[1].first, ops[1].second;
        } else if (ops.size() == 3) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second,
                           ops[1].first, ops[1].second,
                           ops[2].first, ops[2].second;
        } else if (ops.size() == 4) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second,
                           ops[1].first, ops[1].second,
                           ops[2].first, ops[2].second,
                           ops[3].first, ops[3].second;
        } else {
            // Higher-weight Hamiltonians not yet tested
            throw std::runtime_error("Hamiltonians with more than 4-body terms are not yet supported");
            // TODO: Test and implement support for higher-weight terms
            // auto term = ampo;
            // term += adjusted_coeff;
            // for (const auto& op : ops) {
            //     term, op.first, op.second;
            // }
        }
    }

    auto H = toMPO(ampo);

    // Create initial random MPS
    auto psi = randomMPS(sites);

    // Set up DMRG sweeps
    auto sweeps = Sweeps(params.max_sweeps);
    sweeps.maxdim() = params.chi_max;
    sweeps.cutoff() = params.svd_min;
    sweeps.niter() = 2;  // Number of Davidson iterations

    // Run DMRG
    auto args = Args("Silent", params.quiet);
    auto energy = dmrg(psi, H, sweeps, args);

    if (!params.quiet) {
        std::cout << "Ground state energy: " << energy << std::endl;
    }

    return {psi, energy};
}

// OLD VERSION: Uses 2N sites with dimension 2 (physical + ancilla separate)
// Commented out in favor of V2 supersite version
/*
itensor::MPS computePurifiedEquilibriumStateOld(const Hamiltonian& hamiltonian,
                                                 double beta,
                                                 const ThermalParams& params) {
    using namespace itensor;

    int N = hamiltonian.n;

    // Create sites for purification: 2N sites (N physical + N ancilla)
    // Physical sites are at odd indices (1, 3, 5, ...), ancilla at even (2, 4, 6, ...)
    auto sites = SpinHalf(2 * N, {"ConserveQNs=", false});

    // Build Hamiltonian MPO acting only on physical sites
    auto ampo = AutoMPO(sites);

    for (size_t i = 0; i < hamiltonian.terms.size(); ++i) {
        const auto& pauli_string = hamiltonian.terms[i];
        double coeff = hamiltonian.coefficients[i];

        // Parse Pauli string and add to AutoMPO
        std::vector<std::pair<std::string, int>> ops;
        int num_ops = 0;
        for (int j = 0; j < N; ++j) {
            if (pauli_string[j] != 'I') {
                std::string op_name;
                if (pauli_string[j] == 'X') op_name = "Sx";
                else if (pauli_string[j] == 'Y') op_name = "Sy";
                else if (pauli_string[j] == 'Z') op_name = "Sz";
                // Map to physical sites: site j -> 2*j+1 (odd indices)
                ops.push_back({op_name, 2 * j + 1});
                num_ops++;
            }
        }

        // Account for Pauli = 2*S convention
        double adjusted_coeff = coeff * std::pow(2.0, num_ops);

        // Add term to AutoMPO
        if (ops.empty()) {
            continue;  // Skip identity terms
        } else if (ops.size() == 1) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second;
        } else if (ops.size() == 2) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second, ops[1].first, ops[1].second;
        } else if (ops.size() == 3) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second,
                           ops[1].first, ops[1].second,
                           ops[2].first, ops[2].second;
        } else if (ops.size() == 4) {
            ampo += adjusted_coeff, ops[0].first, ops[0].second,
                           ops[1].first, ops[1].second,
                           ops[2].first, ops[2].second,
                           ops[3].first, ops[3].second;
        } else {
            throw std::runtime_error("Hamiltonian terms with more than 4 operators not yet supported");
        }
    }

    // Create evolution operators based on order
    // For order=1: exp(-dt * H)
    // For order=2: exp(-(0.5+0.5i)*dt * H) and exp(-(0.5-0.5i)*dt * H)
    std::vector<MPO> evolution_operators;

    if (params.order == 1) {
        auto expH = toExpH(ampo, Cplx(params.dt, 0.0));
        evolution_operators.push_back(expH);
    } else if (params.order == 2) {
        // Second-order Trotter decomposition
        // Apply U1 = exp(-(0.5-0.5i)*dt*H) then U2 = exp(-(0.5+0.5i)*dt*H)
        // Note: Reversed order from TenPy to match iTensor conventions
        Cplx tau1(0.5 * params.dt, -0.5 * params.dt);  // (0.5-0.5i)*dt
        Cplx tau2(0.5 * params.dt, 0.5 * params.dt);   // (0.5+0.5i)*dt
        auto expH1 = toExpH(ampo, tau1);
        auto expH2 = toExpH(ampo, tau2);
        evolution_operators.push_back(expH1);
        evolution_operators.push_back(expH2);
    } else {
        throw std::runtime_error("Only order=1 and order=2 are supported");
    }

    // Initialize MPS as product of singlets between physical-ancilla pairs
    // Singlet state: (|↑↓⟩ - |↓↑⟩)/√2
    auto psi = MPS(sites);
    const double ISqrt2 = 1.0 / std::sqrt(2.0);

    for (int n = 1; n <= 2 * N; n += 2) {
        auto s1 = sites(n);      // Physical site
        auto s2 = sites(n + 1);  // Ancilla site
        auto wf = ITensor(s1, s2);
        wf.set(s1(1), s2(2), ISqrt2);   // |↑↓⟩
        wf.set(s1(2), s2(1), -ISqrt2);  // -|↓↑⟩

        ITensor D;
        psi.ref(n) = ITensor(s1);
        psi.ref(n + 1) = ITensor(s2);
        svd(wf, psi.ref(n), D, psi.ref(n + 1));
        psi.ref(n) *= D;
    }

    // Imaginary time evolution for total time beta/2
    double ttotal = beta / 2.0;
    int nt = static_cast<int>(ttotal / params.dt + 1e-9 * (ttotal / params.dt));

    if (std::fabs(nt * params.dt - ttotal) > 1e-9) {
        throw std::runtime_error("Time step not commensurate with total time. Adjust dt or beta.");
    }

    if (!params.quiet) {
        std::cout << "Purification: doing " << nt << " steps of dt=" << params.dt
                  << " (order=" << params.order << ") for beta=" << beta << std::endl;
    }

    // Set up arguments for applyMPO
    auto args = Args("MaxDim", params.chi_max,
                     "Cutoff", params.svd_min,
                     "Method", "DensityMatrix");

    // Open dump file if requested
    std::ofstream dump_stream;
    if (!params.dump_file.empty()) {
        dump_stream.open(params.dump_file);
        dump_stream << std::scientific << std::setprecision(16);
    }

    // Evolve in imaginary time
    for (int tt = 1; tt <= nt; ++tt) {
        // Apply evolution operator(s)
        for (const auto& U : evolution_operators) {
            psi = applyMPO(U, psi, args);
            psi.noPrime();
        }

        // Normalize the entire MPS
        psi.normalize();

        // Dump expectation values if requested (to track evolution)
        if (!params.dump_file.empty() && dump_stream.is_open()) {
            dump_stream << "Step " << tt << std::endl;

            // Compute expectation values for key operators
            std::vector<std::string> test_ops = {"ZIIII", "ZZIII", "ZZZII"};

            for (const auto& op : test_ops) {
                // Embed operator into purified system
                std::string embedded_op = embedOperatorInPurifiedSystem(op);
                double exp_val = computeExpectation(psi, embedded_op);
                dump_stream << op << " " << exp_val << std::endl;
            }
        }

        if (!params.quiet && (tt % 10 == 0 || tt == nt)) {
            std::cout << "  Step " << tt << "/" << nt
                      << ", max bond dim = " << maxLinkDim(psi) << std::endl;
        }
    }

    if (dump_stream.is_open()) {
        dump_stream.close();
    }

    if (!params.quiet) {
        std::cout << "Purification complete. Final max bond dim: " << maxLinkDim(psi) << std::endl;
    }

    return psi;
}
*/

itensor::MPS computePurifiedEquilibriumState(const Hamiltonian& hamiltonian,
                                                double beta,
                                                const ThermalParams& params) {
    using namespace itensor;

    int N = hamiltonian.n;

    // Create N supersites with local dimension 4 (2x2 for physical + ancilla)
    // Each supersite represents one physical qubit + one ancilla qubit
    auto sites = SiteSet(N, 4);

    // Initialize MPS as product state with maximally entangled state at each supersite
    // Maximally entangled state: (|00⟩ + |11⟩)/√2
    // In the 4-dimensional basis: |1⟩=|00⟩, |2⟩=|01⟩, |3⟩=|10⟩, |4⟩=|11⟩
    // So we want: (|1⟩ + |4⟩)/√2

    auto psi = MPS(sites);
    const double ISqrt2 = 1.0 / std::sqrt(2.0);

    for (int n = 1; n <= N; ++n) {
        auto s = sites(n);
        auto wf = ITensor(s);
        wf.set(s(1), ISqrt2);   // |00⟩
        wf.set(s(4), ISqrt2);   // |11⟩

        psi.ref(n) = wf;
    }

    psi.position(1);  // Orthogonalize to site 1

    // Build 4x4 operators: Pauli ⊗ I (acting on physical, identity on ancilla)
    // Basis ordering: |00⟩, |01⟩, |10⟩, |11⟩
    // Physical qubit is first, ancilla is second

    // Helper functions to create Pauli ⊗ I operators
    auto make_Sx_tensor_I = [&sites](int n) {
        auto s = sites(n);
        auto op = ITensor(s, prime(s));
        op.set(s(1), prime(s)(3), 0.5);  // |00⟩ -> |10⟩
        op.set(s(2), prime(s)(4), 0.5);  // |01⟩ -> |11⟩
        op.set(s(3), prime(s)(1), 0.5);  // |10⟩ -> |00⟩
        op.set(s(4), prime(s)(2), 0.5);  // |11⟩ -> |01⟩
        return op;
    };

    auto make_Sy_tensor_I = [&sites](int n) {
        auto s = sites(n);
        auto op = ITensor(s, prime(s));
        op.set(s(1), prime(s)(3), Cplx(0, -0.5));  // |00⟩ -> -i|10⟩
        op.set(s(2), prime(s)(4), Cplx(0, -0.5));  // |01⟩ -> -i|11⟩
        op.set(s(3), prime(s)(1), Cplx(0, 0.5));   // |10⟩ -> i|00⟩
        op.set(s(4), prime(s)(2), Cplx(0, 0.5));   // |11⟩ -> i|01⟩
        return op;
    };

    auto make_Sz_tensor_I = [&sites](int n) {
        auto s = sites(n);
        auto op = ITensor(s, prime(s));
        op.set(s(1), prime(s)(1), 0.5);   // |00⟩
        op.set(s(2), prime(s)(2), 0.5);   // |01⟩
        op.set(s(3), prime(s)(3), -0.5);  // |10⟩
        op.set(s(4), prime(s)(4), -0.5);  // |11⟩
        return op;
    };

    // Build Trotter gates with forward and backward passes
    // Following iTensor convention: timestep is halved, gates applied forward then backward
    std::vector<BondGate> gates;

    // Use half time step for each gate (imaginary time evolution)
    double tau = params.dt / 2.0;
    auto gate_type = BondGate::tImag;

    // Group Hamiltonian terms by bond for efficient gate construction
    // Map: bond index -> list of terms acting on that bond
    std::map<int, std::vector<ITensor>> bond_terms;

    // Process each Hamiltonian term
    for (size_t i = 0; i < hamiltonian.terms.size(); ++i) {
        const auto& pauli_string = hamiltonian.terms[i];
        double coeff = hamiltonian.coefficients[i];

        // Find non-identity operators
        std::vector<std::pair<int, char>> ops;  // (site_index, pauli_char)
        for (int j = 0; j < N; ++j) {
            if (pauli_string[j] != 'I') {
                ops.push_back({j + 1, pauli_string[j]});  // 1-indexed
            }
        }

        if (ops.empty()) continue;  // Skip identity terms

        // Account for Pauli = 2*S convention
        double adjusted_coeff = coeff * std::pow(2.0, ops.size());

        // Determine which bond this term acts on and build complete operator
        if (ops.size() == 1) {
            // Single-site term
            int site = ops[0].first;
            char pauli = ops[0].second;

            ITensor op_tensor;
            if (pauli == 'X') op_tensor = make_Sx_tensor_I(site);
            else if (pauli == 'Y') op_tensor = make_Sy_tensor_I(site);
            else if (pauli == 'Z') op_tensor = make_Sz_tensor_I(site);

            if (site < N) {
                // Not on terminal site: assign to bond (site, site+1) as X_site ⊗ I_{site+1}
                auto s_next = sites(site + 1);
                auto sp_next = prime(s_next);
                auto id_next = ITensor(s_next, sp_next);
                id_next.set(s_next(1), sp_next(1), 1.0);
                id_next.set(s_next(2), sp_next(2), 1.0);
                id_next.set(s_next(3), sp_next(3), 1.0);
                id_next.set(s_next(4), sp_next(4), 1.0);

                ITensor hterm = adjusted_coeff * op_tensor * id_next;
                bond_terms[site].push_back(hterm);
            } else {
                // On terminal site N: assign to bond (N-1, N) as I_{N-1} ⊗ X_N
                auto s_prev = sites(N - 1);
                auto sp_prev = prime(s_prev);
                auto id_prev = ITensor(s_prev, sp_prev);
                id_prev.set(s_prev(1), sp_prev(1), 1.0);
                id_prev.set(s_prev(2), sp_prev(2), 1.0);
                id_prev.set(s_prev(3), sp_prev(3), 1.0);
                id_prev.set(s_prev(4), sp_prev(4), 1.0);

                ITensor hterm = adjusted_coeff * id_prev * op_tensor;
                bond_terms[N - 1].push_back(hterm);
            }
        } else if (ops.size() == 2 && ops[1].first == ops[0].first + 1) {
            // Nearest-neighbor term: build product of operators
            int bond = ops[0].first;
            int site1 = ops[0].first;
            int site2 = ops[1].first;
            char pauli1 = ops[0].second;
            char pauli2 = ops[1].second;

            ITensor op_tensor1;
            if (pauli1 == 'X') op_tensor1 = make_Sx_tensor_I(site1);
            else if (pauli1 == 'Y') op_tensor1 = make_Sy_tensor_I(site1);
            else if (pauli1 == 'Z') op_tensor1 = make_Sz_tensor_I(site1);

            ITensor op_tensor2;
            if (pauli2 == 'X') op_tensor2 = make_Sx_tensor_I(site2);
            else if (pauli2 == 'Y') op_tensor2 = make_Sy_tensor_I(site2);
            else if (pauli2 == 'Z') op_tensor2 = make_Sz_tensor_I(site2);

            ITensor hterm = adjusted_coeff * op_tensor1 * op_tensor2;
            bond_terms[bond].push_back(hterm);
        } else {
            // Longer-range terms not supported with simple BondGate approach
            if (!params.quiet) {
                std::cerr << "Warning: Skipping non-nearest-neighbor term " << pauli_string << std::endl;
            }
        }
    }

    // Create forward pass gates (bonds 1 to N-1)
    for (int b = 1; b <= N - 1; ++b) {
        if (bond_terms.count(b)) {
            // Sum all terms acting on this bond
            ITensor hterm_sum;
            for (size_t i = 0; i < bond_terms[b].size(); ++i) {
                if (i == 0) {
                    hterm_sum = bond_terms[b][i];
                } else {
                    hterm_sum += bond_terms[b][i];
                }
            }
            auto g = BondGate(sites, b, b + 1, gate_type, tau, hterm_sum);
            gates.push_back(g);
        }
    }

    // Create backward pass gates (bonds N-1 to 1)
    for (int b = N - 1; b >= 1; --b) {
        if (bond_terms.count(b)) {
            // Sum all terms acting on this bond
            ITensor hterm_sum;
            for (size_t i = 0; i < bond_terms[b].size(); ++i) {
                if (i == 0) {
                    hterm_sum = bond_terms[b][i];
                } else {
                    hterm_sum += bond_terms[b][i];
                }
            }
            auto g = BondGate(sites, b, b + 1, gate_type, tau, hterm_sum);
            gates.push_back(g);
        }
    }

    // Apply gates for imaginary time evolution to beta/2
    double ttotal = beta / 2.0;
    auto args = Args("Cutoff=", params.svd_min, "MaxDim=", params.chi_max, "Verbose=", !params.quiet);

    gateTEvol(gates, ttotal, params.dt, psi, args);

    if (!params.quiet) {
        std::cout << "Purification complete. Final max bond dim: " << maxLinkDim(psi) << std::endl;
    }

    return psi;
}

// ============================================================================
// State computation - Exact Diagonalization
// ============================================================================

VectorXcd computeGroundStateED(const Hamiltonian& hamiltonian,
                               const EDParams& params) {
    int n = hamiltonian.n;
    int dim = 1 << n;  // 2^n

    // Build many-body Hamiltonian matrix
    MatrixXcd H_mb = MatrixXcd::Zero(dim, dim);
    for (size_t i = 0; i < hamiltonian.terms.size(); ++i) {
        H_mb += hamiltonian.coefficients[i] * pauliMatrix(hamiltonian.terms[i]);
    }

    // Check Hermiticity
    if (!H_mb.isApprox(H_mb.adjoint(), 1e-10)) {
        throw std::runtime_error("Hamiltonian matrix is not Hermitian");
    }

    // Compute ground state
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(H_mb);
    VectorXcd ground_state = solver.eigenvectors().col(0);

    return ground_state;
}

MatrixXcd computeEquilibriumStateED(const Hamiltonian& hamiltonian,
                                    double beta,
                                    const EDParams& params) {
    int n = hamiltonian.n;
    int dim = 1 << n;  // 2^n

    // Build many-body Hamiltonian matrix
    MatrixXcd H_mb = MatrixXcd::Zero(dim, dim);
    for (size_t i = 0; i < hamiltonian.terms.size(); ++i) {
        H_mb += hamiltonian.coefficients[i] * pauliMatrix(hamiltonian.terms[i]);
    }

    // Check Hermiticity
    if (!H_mb.isApprox(H_mb.adjoint(), 1e-10)) {
        throw std::runtime_error("Hamiltonian matrix is not Hermitian");
    }

    // Compute thermal state: rho = exp(-beta * H) / Z
    MatrixXcd rho = (-beta * H_mb).exp();
    Complex trace = rho.trace();
    rho /= trace;

    return rho;
}

// ============================================================================
// Expectation value computation - iTensor implementation
// ============================================================================

// Helper: Build a Pauli operator ITensor for a single site
itensor::ITensor buildPauliOperator(const itensor::Index& site_index, char pauli_char) {
    using namespace itensor;

    auto s = site_index;
    auto sp = prime(s);
    auto op = ITensor(s, sp);

    if (pauli_char == 'I') {
        op.set(s(1), sp(1), 1.0);
        op.set(s(2), sp(2), 1.0);
    } else if (pauli_char == 'X') {
        op.set(s(1), sp(2), 1.0);
        op.set(s(2), sp(1), 1.0);
    } else if (pauli_char == 'Y') {
        op.set(s(1), sp(2), Cplx_i);
        op.set(s(2), sp(1), -Cplx_i);
    } else if (pauli_char == 'Z') {
        op.set(s(1), sp(1), 1.0);
        op.set(s(2), sp(2), -1.0);
    } else {
        throw std::runtime_error("Unknown Pauli operator: " + std::string(1, pauli_char));
    }

    return op;
}

// Build 4x4 Pauli ⊗ I operators for supersites (dimension 4)
// Basis ordering: |00⟩, |01⟩, |10⟩, |11⟩ where first qubit is physical, second is ancilla
itensor::ITensor buildPauliOperatorSupersite(const itensor::Index& site_index, char pauli_char) {
    using namespace itensor;

    auto s = site_index;
    auto sp = prime(s);
    auto op = ITensor(s, sp);

    if (pauli_char == 'I') {
        // Identity on both physical and ancilla
        op.set(s(1), sp(1), 1.0);
        op.set(s(2), sp(2), 1.0);
        op.set(s(3), sp(3), 1.0);
        op.set(s(4), sp(4), 1.0);
    } else if (pauli_char == 'X') {
        // X ⊗ I: acts on physical, identity on ancilla
        op.set(s(1), sp(3), 1.0);  // |00⟩ -> |10⟩
        op.set(s(2), sp(4), 1.0);  // |01⟩ -> |11⟩
        op.set(s(3), sp(1), 1.0);  // |10⟩ -> |00⟩
        op.set(s(4), sp(2), 1.0);  // |11⟩ -> |01⟩
    } else if (pauli_char == 'Y') {
        // Y ⊗ I: acts on physical with phase, identity on ancilla
        op.set(s(1), sp(3), -Cplx_i);  // |00⟩ -> -i|10⟩
        op.set(s(2), sp(4), -Cplx_i);  // |01⟩ -> -i|11⟩
        op.set(s(3), sp(1), Cplx_i);   // |10⟩ -> i|00⟩
        op.set(s(4), sp(2), Cplx_i);   // |11⟩ -> i|01⟩
    } else if (pauli_char == 'Z') {
        // Z ⊗ I: diagonal on physical, identity on ancilla
        op.set(s(1), sp(1), 1.0);   // |00⟩
        op.set(s(2), sp(2), 1.0);   // |01⟩
        op.set(s(3), sp(3), -1.0);  // |10⟩
        op.set(s(4), sp(4), -1.0);  // |11⟩
    } else {
        throw std::runtime_error("Unknown Pauli operator: " + std::string(1, pauli_char));
    }

    return op;
}

std::vector<double> computeExpectationsPureNaive(const itensor::MPS& psi,
                                                  const std::vector<std::string>& operators) {
    // Naive implementation: compute each expectation value independently
    std::vector<double> expectations;
    expectations.reserve(operators.size());

    for (const auto& op : operators) {
        expectations.push_back(computeExpectation(psi, op));
    }

    return expectations;
}

// OLD VERSION: Uses 2N-site representation
// Commented out in favor of V2 supersite version
/*
std::vector<double> computeExpectationsMixedNaiveOld(const itensor::MPS& psi,
                                                      const std::vector<std::string>& operators) {
    // Naive implementation: compute each expectation value independently
    // Note: psi is expected to be a purified mixed state (MPS with 2N sites)
    std::vector<double> expectations;
    expectations.reserve(operators.size());

    for (const auto& op : operators) {
        expectations.push_back(computeExpectationPurifiedOld(psi, op));
    }

    return expectations;
}
*/

std::vector<double> computeExpectationsMixedNaive(const itensor::MPS& psi,
                                                   const std::vector<std::string>& operators) {
    // Naive implementation for supersite representation
    // Note: psi is expected to be a purified mixed state with N supersites (dimension 4 each)
    std::vector<double> expectations;
    expectations.reserve(operators.size());

    for (const auto& op : operators) {
        expectations.push_back(computeExpectationPurified(psi, op));
    }

    return expectations;
}

std::vector<double> computeExpectationsMixed(const itensor::MPS& psi,
                                              const std::vector<std::string>& operators) {
    using namespace itensor;

    if (operators.empty()) {
        return std::vector<double>();
    }

    int N = length(psi);
    int num_ops = operators.size();
    std::vector<double> expectations(num_ops);

    // Get site indices
    auto sites = siteInds(psi);

    // Create psi_dag_prime: MPS with all indices primed and conjugated
    auto psi_dag_prime = psi;  // Copy the MPS
    for (int k = 1; k <= N; ++k) {
        psi_dag_prime.ref(k) = prime(conj(psi(k)));  // Conjugate and prime all indices
    }

    // Pre-compute R tensors (right-to-left sweep)
    std::vector<ITensor> R_tensors(N + 1);
    R_tensors[N] = ITensor(1.0); // Scalar tensor

    for (int k = N; k >= 1; --k) {
        auto psi_k = psi(k);
        auto psi_k_dag_prime = psi_dag_prime(k);

        // Unprime the site index so it contracts with psi_k
        // We need to find the primed site index from psi_k_dag_prime itself
        auto site_idx_prime = findIndex(inds(psi_k_dag_prime), "Site");
        psi_k_dag_prime.noPrime(site_idx_prime);  // Unprime site index (modifies in-place)

        // Contract: R[k] * psi(k) * conj(psi(k)) with site unprimed, links primed
        R_tensors[k-1] = R_tensors[k] * psi_k * psi_k_dag_prime;
    }

    // Initialize L tensors
    std::vector<ITensor> L_tensors(N + 1);
    L_tensors[0] = ITensor(1.0); // Scalar tensor

    // Track previous operator for DFS optimization
    std::string previous(N, '-');

    // Process each operator
    for (int i = 0; i < num_ops; ++i) {
        const auto& p = operators[i];

        if (static_cast<int>(p.length()) != N) {
            throw std::runtime_error("Operator string length doesn't match MPS length");
        }

        // Find first index that differs from previous operator
        int j = 0;
        for (j = 0; j < N; ++j) {
            if (p[j] != previous[j]) {
                break;
            }
        }

        // Find last non-trivial index (defaults to -1 if all identity)
        int last_nontriv = -1;
        for (int x = 0; x < N; ++x) {
            if (p[x] != 'I') {
                last_nontriv = x;
            }
        }

        int y = std::max(j, last_nontriv);

        // Update L tensors from j to y
        for (int k = j; k <= y; ++k) {
            auto L = L_tensors[k];
            auto s = sites(k + 1);

            // Build supersite operator for this site (Pauli ⊗ I, dimension 4x4)
            auto op = buildPauliOperatorSupersite(s, p[k]);

            // M(k) = psi(k) * O(k) * psi_dag_prime(k)
            auto M = psi(k + 1) * op * psi_dag_prime(k + 1);

            // L(k+1) = L(k) * M(k)
            L_tensors[k+1] = L * M;
        }

        // Compute final expectation value
        auto result = L_tensors[y+1] * R_tensors[y+1];
        expectations[i] = eltC(result).real();

        previous = p;
    }

    return expectations;
}

std::vector<double> computeExpectationsMixedParallel(const itensor::MPS& psi,
                                                      const std::vector<std::string>& operators,
                                                      int num_threads) {
    if (operators.empty()) {
        return std::vector<double>();
    }

    // Determine number of threads
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;  // Fallback
    }

    size_t total_ops = operators.size();

    // If too few operators, just use sequential version
    if (total_ops < static_cast<size_t>(num_threads * 10)) {
        return computeExpectationsMixed(psi, operators);
    }

    // Calculate chunk size
    size_t chunk_size = (total_ops + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    std::vector<std::vector<double>> results(num_threads);

    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, total_ops);

        if (start >= total_ops) break;

        // Create chunk (copy the relevant portion)
        std::vector<std::string> chunk(operators.begin() + start,
                                       operators.begin() + end);

        threads.emplace_back([&psi, &results, t, chunk]() {
            results[t] = computeExpectationsMixed(psi, chunk);
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Concatenate results
    std::vector<double> all_expectations;
    all_expectations.reserve(total_ops);
    for (const auto& result : results) {
        all_expectations.insert(all_expectations.end(), result.begin(), result.end());
    }

    return all_expectations;
}

std::vector<double> computeExpectationsPure(const itensor::MPS& psi,
                                             const std::vector<std::string>& operators) {
    using namespace itensor;

    if (operators.empty()) {
        return std::vector<double>();
    }

    int N = length(psi);
    int num_ops = operators.size();
    std::vector<double> expectations(num_ops);

    // Get site indices
    auto sites = siteInds(psi);

    // Create psi_dag_prime: MPS with all indices primed and conjugated
    auto psi_dag_prime = psi;  // Copy the MPS
    for (int k = 1; k <= N; ++k) {
        psi_dag_prime.ref(k) = prime(conj(psi(k)));  // Conjugate and prime all indices
    }

    // Pre-compute R tensors (right-to-left sweep)
    std::vector<ITensor> R_tensors(N + 1);
    R_tensors[N] = ITensor(1.0); // Scalar tensor

    for (int k = N; k >= 1; --k) {
        auto psi_k = psi(k);
        auto psi_k_dag_prime = psi_dag_prime(k);

        // Unprime the site index so it contracts with psi_k
        // We need to find the primed site index from psi_k_dag_prime itself
        auto site_idx_prime = findIndex(inds(psi_k_dag_prime), "Site");
        psi_k_dag_prime.noPrime(site_idx_prime);  // Unprime site index (modifies in-place)

        // Contract: R[k] * psi(k) * conj(psi(k)) with site unprimed, links primed
        R_tensors[k-1] = R_tensors[k] * psi_k * psi_k_dag_prime;
    }

    // Initialize L tensors
    std::vector<ITensor> L_tensors(N + 1);
    L_tensors[0] = ITensor(1.0); // Scalar tensor

    // Track previous operator for DFS optimization
    std::string previous(N, '-');

    // Process each operator
    for (int i = 0; i < num_ops; ++i) {
        const auto& p = operators[i];

        if (static_cast<int>(p.length()) != N) {
            throw std::runtime_error("Operator string length doesn't match MPS length");
        }

        // Find first index that differs from previous operator
        int j = 0;
        for (j = 0; j < N; ++j) {
            if (p[j] != previous[j]) {
                break;
            }
        }

        // Find last non-trivial index (defaults to -1 if all identity)
        int last_nontriv = -1;
        for (int x = 0; x < N; ++x) {
            if (p[x] != 'I') {
                last_nontriv = x;
            }
        }

        int y = std::max(j, last_nontriv);

        // Update L tensors from j to y
        for (int k = j; k <= y; ++k) {
            auto L = L_tensors[k];
            auto s = sites(k + 1);

            // Build operator for this site (has indices p and p')
            auto op = buildPauliOperator(s, p[k]);

            // M(k) = psi(k) * O(k) * psi_dag_prime(k)
            // psi(k+1) has [vL, p, vR]
            // op has [p, p']
            // psi_dag_prime(k+1) has [vL', p', vR']
            // Contractions: psi[p] with op[p], op[p'] with psi_dag_prime[p']
            // Result M has: psi[vL, vR] and psi_dag_prime[vL', vR']
            auto M = psi(k + 1) * op * psi_dag_prime(k + 1);

            // L(k+1) = L(k) * M(k)
            // L has [vR, vR'] from previous iteration
            // M has [vL, vR, vL', vR']
            // Contractions: L[vR] with M[vL], L[vR'] with M[vL']
            // Result has: M[vR, vR']
            L_tensors[k+1] = L * M;
        }

        // Compute final expectation value
        auto result = L_tensors[y+1] * R_tensors[y+1];
        expectations[i] = eltC(result).real();

        previous = p;
    }

    return expectations;
}

// OLD VERSION: Embedding helper for 2N-site representation
// Commented out in favor of V2 supersite version
/*
std::string embedOperatorInPurifiedSystem(const std::string& op) {
    std::string embedded;
    embedded.reserve(2 * op.length());
    for (char c : op) {
        embedded += c;   // Operator on physical site (odd 1-indexed: 1,3,5,...)
        embedded += 'I'; // Identity on ancilla site (even 1-indexed: 2,4,6,...)
    }
    return embedded;
}

std::vector<double> computeExpectationsMixedOld(const itensor::MPS& psi,
                                                 const std::vector<std::string>& operators) {
    // Embed operators into doubled purified system and use DFS-optimized pure state algorithm
    std::vector<std::string> embedded_operators;
    embedded_operators.reserve(operators.size());
    for (const auto& op : operators) {
        embedded_operators.push_back(embedOperatorInPurifiedSystem(op));
    }
    return computeExpectationsPure(psi, embedded_operators);
}
*/

std::vector<double> computeExpectationsED(const VectorXcd& psi,
                                          const std::vector<std::string>& operators) {
    std::vector<double> expectations;
    expectations.reserve(operators.size());
    
    MatrixXcd rho = psi * psi.adjoint();

    for (const auto& op : operators) {
        double exp_val = computeExpectation(op, rho);
        expectations.push_back(exp_val);
    }

    return expectations;
}

std::vector<double> computeExpectationsED(const MatrixXcd& rho,
                                          const std::vector<std::string>& operators) {
    std::vector<double> expectations;
    expectations.reserve(operators.size());

    for (const auto& op : operators) {
        double exp_val = computeExpectation(op, rho);
        expectations.push_back(exp_val);
    }

    return expectations;
}

// ============================================================================
// Helper for expectation values
// ============================================================================

// For ED pure states (VectorXcd)
double computeExpectation(const std::string& pauli_string, const VectorXcd& psi) {
    MatrixXcd P = pauliMatrix(pauli_string);
    Complex result = psi.adjoint() * P * psi;
    return result.real();
}

// For ED mixed states (MatrixXcd)
double computeExpectation(const std::string& pauli_string, const MatrixXcd& rho) {
    MatrixXcd P = pauliMatrix(pauli_string);
    Complex trace = (rho * P).trace();
    return trace.real();
}

// For MPS pure states
double computeExpectation(const itensor::MPS& psi,
                          const std::string& pauli_string) {
    using namespace itensor;

    int N = length(psi);

    if (static_cast<int>(pauli_string.length()) != N) {
        throw std::runtime_error("Pauli string length doesn't match number of sites");
    }

    // Build MPO for the Pauli operator
    auto sites = siteInds(psi);
    auto W = MPO(sites);

    for (int j = 0; j < N; ++j) {
        auto s = sites(j + 1);  // Sites are 1-indexed in iTensor
        W.ref(j + 1) = buildPauliOperator(s, pauli_string[j]);
    }

    // Compute <psi|W|psi> using innerC
    auto exp_val = innerC(psi, W, psi);
    return exp_val.real();
}

// ============================================================================
// Helper for purified MPS expectation values
// ============================================================================

// OLD VERSION: Uses 2N-site representation with separate physical and ancilla sites
// Commented out in favor of V2 supersite version
/*
double computeExpectationPurifiedOld(const itensor::MPS& psi,
                                      const std::string& pauli_string) {
    using namespace itensor;

    // Get number of physical sites (psi has 2N sites total)
    int total_sites = length(psi);
    int N = total_sites / 2;

    if (static_cast<int>(pauli_string.length()) != N) {
        throw std::runtime_error("Pauli string length doesn't match number of physical sites");
    }

    // Build MPO for O⊗I where O acts on physical sites (odd indices)
    // and I acts on ancilla sites (even indices)
    auto sites = siteInds(psi);
    auto W = MPO(sites);

    for (int j = 0; j < N; ++j) {
        int phys_site = 2 * j + 1;  // Physical sites at odd indices (1,3,5,...)
        int anc_site = 2 * j + 2;   // Ancilla sites at even indices (2,4,6,...)

        auto s_phys = sites(phys_site);
        auto s_anc = sites(anc_site);

        // Build operator for physical site and identity for ancilla site
        W.ref(phys_site) = buildPauliOperator(s_phys, pauli_string[j]);
        W.ref(anc_site) = buildPauliOperator(s_anc, 'I');
    }

    // Compute <psi|W|psi> using innerC
    auto exp_val = innerC(psi, W, psi);
    return exp_val.real();
}
*/

// Helper for purified MPS with supersites
double computeExpectationPurified(const itensor::MPS& psi,
                                     const std::string& pauli_string) {
    using namespace itensor;

    // psi has N supersites (dimension 4 each)
    int N = length(psi);

    if (static_cast<int>(pauli_string.length()) != N) {
        throw std::runtime_error("Pauli string length doesn't match number of supersites");
    }

    // Build MPO using Pauli ⊗ I operators for supersites
    auto sites = siteInds(psi);
    auto W = MPO(sites);

    for (int j = 0; j < N; ++j) {
        auto s = sites(j + 1);  // Sites are 1-indexed in iTensor
        W.ref(j + 1) = buildPauliOperatorSupersite(s, pauli_string[j]);
    }

    // Compute <psi|W|psi> using innerC
    auto exp_val = innerC(psi, W, psi);
    return exp_val.real();
}

} // namespace hamiltonian_learning
