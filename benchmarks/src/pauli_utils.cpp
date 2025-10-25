#include "pauli_utils.hpp"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <set>
#include <functional>
#include <omp.h>

namespace hamiltonian_learning {

// Pauli generators
static const std::map<char, MatrixXc> pauli_generators = {
    {'I', MatrixXc::Identity(2, 2)},
    {'X', (MatrixXc(2, 2) << 0, 1, 1, 0).finished()},
    {'Y', (MatrixXc(2, 2) << 0, Complex(0, -1), Complex(0, 1), 0).finished()},
    {'Z', (MatrixXc(2, 2) << 1, 0, 0, -1).finished()}
};

// Single Pauli multiplication table
static std::map<std::pair<char, char>, std::pair<char, Complex>> buildPauliMultTable() {
    std::map<std::pair<char, char>, std::pair<char, Complex>> table;
    table[{'I', 'I'}] = {'I', 1.0};
    table[{'I', 'X'}] = {'X', 1.0};
    table[{'I', 'Y'}] = {'Y', 1.0};
    table[{'I', 'Z'}] = {'Z', 1.0};

    table[{'X', 'I'}] = {'X', 1.0};
    table[{'X', 'X'}] = {'I', 1.0};
    table[{'X', 'Y'}] = {'Z', Complex(0, 1)};
    table[{'X', 'Z'}] = {'Y', Complex(0, -1)};

    table[{'Y', 'I'}] = {'Y', 1.0};
    table[{'Y', 'X'}] = {'Z', Complex(0, -1)};
    table[{'Y', 'Y'}] = {'I', 1.0};
    table[{'Y', 'Z'}] = {'X', Complex(0, 1)};

    table[{'Z', 'I'}] = {'Z', 1.0};
    table[{'Z', 'X'}] = {'Y', Complex(0, 1)};
    table[{'Z', 'Y'}] = {'X', Complex(0, -1)};
    table[{'Z', 'Z'}] = {'I', 1.0};

    return table;
}

static const auto single_pauli_mult_matrix = buildPauliMultTable();

// Pauli operations
std::string compressPauli(const std::string& pauli) {
    int n = pauli.length();
    auto supp = determineSupport(pauli);
    int w = 0;
    for (bool s : supp) w += s ? 1 : 0;

    std::ostringstream out;
    out << w;
    for (int i = 0; i < n; ++i) {
        if (supp[i]) {
            out << " " << pauli[i] << " " << i;
        }
    }
    return out.str();
}

std::string decompressPauli(const std::string& pauli_compressed, int n) {
    std::string result(n, 'I');
    std::istringstream iss(pauli_compressed);

    int k;
    iss >> k;

    for (int i = 0; i < k; ++i) {
        char pauli_char;
        int index;
        iss >> pauli_char >> index;
        result[index] = pauli_char;
    }

    return result;
}

std::vector<std::string> compressPauliToList(const std::string& pauli) {
    int n = pauli.length();
    auto supp = determineSupport(pauli);
    int w = 0;
    for (bool s : supp) w += s ? 1 : 0;

    std::vector<std::string> out;
    out.push_back(std::to_string(w));
    for (int i = 0; i < n; ++i) {
        if (supp[i]) {
            out.push_back(std::string(1, pauli[i]));
            out.push_back(std::to_string(i));
        }
    }
    return out;
}

std::string parseHumanReadablePauli(const std::string& op_str, int n) {
    // Initialize with all identity
    std::string result(n, 'I');

    // Handle empty string or whitespace-only
    if (op_str.empty() || op_str.find_first_not_of(" \t\n\r") == std::string::npos) {
        return result;
    }

    // Parse tokens: "Z_11 Z_15 X_14" or "Z11 Z15 X14"
    std::istringstream iss(op_str);
    std::string token;

    while (iss >> token) {
        // Token should be like "Z_11" or "Z11"
        if (token.empty()) continue;

        // First character is the Pauli type
        char pauli_type = token[0];
        if (pauli_type != 'I' && pauli_type != 'X' && pauli_type != 'Y' && pauli_type != 'Z') {
            throw std::runtime_error("Invalid Pauli type '" + std::string(1, pauli_type) + "' in token '" + token + "'");
        }

        // Skip identity operators
        if (pauli_type == 'I') continue;

        // Rest is the site index (may have underscore)
        std::string index_str = token.substr(1);

        // Remove underscore if present
        if (!index_str.empty() && index_str[0] == '_') {
            index_str = index_str.substr(1);
        }

        // Parse site index
        if (index_str.empty()) {
            throw std::runtime_error("No site index specified in token '" + token + "'");
        }

        int site_idx;
        try {
            site_idx = std::stoi(index_str);
        } catch (...) {
            throw std::runtime_error("Invalid site index in token '" + token + "'");
        }

        // Validate site index (0-indexed)
        if (site_idx < 0 || site_idx >= n) {
            throw std::runtime_error("Site index " + std::to_string(site_idx) + " out of range [0, " + std::to_string(n-1) + "]");
        }

        // Check for duplicate assignment
        if (result[site_idx] != 'I') {
            throw std::runtime_error("Site " + std::to_string(site_idx) + " specified multiple times");
        }

        // Place Pauli at specified site
        result[site_idx] = pauli_type;
    }

    return result;
}

std::string embedPauli(int n, const std::string& p, const std::vector<int>& region) {
    std::string out(n, 'I');
    for (size_t i = 0; i < region.size(); ++i) {
        out[region[i]] = p[i];
    }
    return out;
}

std::vector<std::string> restrictOperators(int n,
                                          const std::vector<std::string>& operators,
                                          const std::vector<int>& region) {
    // Build region complement
    std::set<int> region_set(region.begin(), region.end());
    std::vector<int> region_complement;
    for (int i = 0; i < n; ++i) {
        if (region_set.find(i) == region_set.end()) {
            region_complement.push_back(i);
        }
    }

    std::vector<std::string> out;
    for (const auto& p : operators) {
        auto support = determineSupport(p);

        // Check if operator has support outside region
        bool has_support_outside = false;
        for (int i : region_complement) {
            if (support[i]) {
                has_support_outside = true;
                break;
            }
        }

        if (!has_support_outside) {
            // Restrict to region
            std::string p_restricted;
            for (int i : region) {
                p_restricted += p[i];
            }
            out.push_back(p_restricted);
        }
    }

    return out;
}

std::pair<std::string, Complex> multiplyPaulis(const std::string& pauli1,
                                               const std::string& pauli2) {
    if (pauli1.length() != pauli2.length()) {
        throw std::runtime_error("Pauli strings must have same length");
    }

    std::string result;
    Complex phase = 1.0;

    for (size_t i = 0; i < pauli1.length(); ++i) {
        auto [W, z] = single_pauli_mult_matrix.at({pauli1[i], pauli2[i]});
        result += W;
        phase *= z;
    }

    return {result, phase};
}

bool checkCommute(const std::string& pauli1, const std::string& pauli2) {
    if (pauli1.length() != pauli2.length()) {
        throw std::runtime_error("Pauli strings must have same length");
    }

    int total = 0;
    for (size_t i = 0; i < pauli1.length(); ++i) {
        char a = pauli1[i];
        char b = pauli2[i];
        if (a != 'I' && b != 'I' && a != b) {
            total++;
        }
    }
    return total % 2 == 0;
}

bool checkOverlap(const std::string& pauli1, const std::string& pauli2) {
    if (pauli1.length() != pauli2.length()) {
        throw std::runtime_error("Pauli strings must have same length");
    }

    for (size_t i = 0; i < pauli1.length(); ++i) {
        if (pauli1[i] != 'I' && pauli2[i] != 'I') {
            return true;
        }
    }
    return false;
}

std::vector<bool> determineSupport(const std::string& pauli_string) {
    std::vector<bool> support;
    for (char c : pauli_string) {
        support.push_back(c != 'I');
    }
    return support;
}

int weight(const std::string& pauli_string) {
    int w = 0;
    for (char c : pauli_string) {
        if (c != 'I') w++;
    }
    return w;
}

std::vector<std::string> buildKLocalPaulis1D(int n, int k, bool periodic_bc) {
    if (n == 1) {
        if (k == 0) {
            return {"I"};
        } else {
            return {"I", "X", "Y", "Z"};
        }
    }

    std::set<std::string> out_set;
    const std::vector<char> paulis = {'X', 'Y', 'Z', 'I'};

    int max_i = periodic_bc ? n : (n - k + 1);
    for (int i = 0; i < max_i; ++i) {
        // Generate all combinations of k paulis
        std::function<void(int, std::string&)> generate;
        generate = [&](int depth, std::string& current) {
            if (depth == k) {
                out_set.insert(current);
                return;
            }
            for (char p : paulis) {
                int pos = periodic_bc ? (i + depth) % n : (i + depth);
                current[pos] = p;
                generate(depth + 1, current);
                current[pos] = 'I';
            }
        };

        std::string p(n, 'I');
        generate(0, p);
    }

    std::vector<std::string> result(out_set.begin(), out_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

MatrixXc pauliMatrix(const std::string& pauli_string) {
    MatrixXc result = MatrixXc::Identity(1, 1);

    for (char c : pauli_string) {
        MatrixXc current = pauli_generators.at(c);
        MatrixXc new_result(result.rows() * 2, result.cols() * 2);

        // Kronecker product
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                new_result.block(i * 2, j * 2, 2, 2) = result(i, j) * current;
            }
        }
        result = new_result;
    }

    return result;
}

// Note: complexToReal() and realToComplex() are now in sparse_tensor.cpp
// They are general linear algebra utilities, not Pauli-specific

// Note: tprint/tprintf are implemented in hamiltonian_learning.cpp

// Note: createSaveDirectory() commented out - not used
/*
std::string createSaveDirectory() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t_now);

    std::ostringstream oss;
    oss << "./runs/" << std::put_time(&tm, "%Y_%m_%d-%H_%M_%S");

    // In a real implementation, you'd create the directory here
    // For now, just return the path
    return oss.str();
}
*/

std::pair<SparseTensor, std::vector<std::string>>
buildMultiplicationTensor(const std::vector<std::string>& onebody_operators) {
    int R = onebody_operators.size();

    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;

    std::vector<std::string> twobody_operators;
    std::map<std::string, int> twobody_indices_dict;
    int l = 0;

    for (int i = 0; i < R; ++i) {
        for (int j = i; j < R; ++j) {
            auto [W, z] = multiplyPaulis(onebody_operators[i], onebody_operators[j]);

            // Add to twobody_operators if not seen before
            if (twobody_indices_dict.find(W) == twobody_indices_dict.end()) {
                twobody_operators.push_back(W);
                twobody_indices_dict[W] = l;
                l++;
            }

            // Add entry for (i,j)
            indices.push_back({i, j, twobody_indices_dict[W]});
            values.push_back(z);

            // Add symmetric entry for (j,i) if i != j
            if (j > i) {
                indices.push_back({j, i, twobody_indices_dict[W]});
                values.push_back(std::conj(z));
            }
        }
    }

    SparseTensor mult_tensor({R, R, l}, indices, values);
    return {mult_tensor, twobody_operators};
}

std::pair<SparseTensor, std::vector<std::string>>
buildMultiplicationTensorSingleThreaded(const std::vector<std::string>& onebody_operators) {
    int R = onebody_operators.size();

    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;

    std::vector<std::string> twobody_operators;
    std::map<std::string, int> twobody_indices_dict;
    int l = 0;

    for (int i = 0; i < R; ++i) {
        for (int j = i; j < R; ++j) {
            auto [W, z] = multiplyPaulis(onebody_operators[i], onebody_operators[j]);

            // Add to twobody_operators if not seen before
            if (twobody_indices_dict.find(W) == twobody_indices_dict.end()) {
                twobody_operators.push_back(W);
                twobody_indices_dict[W] = l;
                l++;
            }

            // Add entry for (i,j)
            indices.push_back({i, j, twobody_indices_dict[W]});
            values.push_back(z);

            // Add symmetric entry for (j,i) if i != j
            if (j > i) {
                indices.push_back({j, i, twobody_indices_dict[W]});
                values.push_back(std::conj(z));
            }
        }
    }

    SparseTensor mult_tensor({R, R, l}, indices, values);
    return {mult_tensor, twobody_operators};
}

// Helper function to convert Pauli string to integer array
static std::vector<uint8_t> pauliStringToIntArray(const std::string& p) {
    static const std::map<char, uint8_t> pauli_char_to_int = {
        {'I', 0}, {'X', 1}, {'Y', 2}, {'Z', 3}
    };

    std::vector<uint8_t> result;
    result.reserve(p.length());
    for (char c : p) {
        result.push_back(pauli_char_to_int.at(c));
    }
    return result;
}

// Helper function to convert integer array to Pauli string
static std::string intArrayToPauliString(const std::vector<uint8_t>& arr) {
    static const char int_to_pauli_char[] = {'I', 'X', 'Y', 'Z'};

    std::string result;
    result.reserve(arr.size());
    for (uint8_t val : arr) {
        result += int_to_pauli_char[val];
    }
    return result;
}

std::pair<SparseTensor, std::vector<std::string>>
buildTripleProductTensor(const std::vector<std::string>& onebody_operators,
                        const std::vector<std::string>& hamiltonian_terms) {
    int n = onebody_operators[0].length();
    int r = onebody_operators.size();
    int h = hamiltonian_terms.size();

    // Find noncommuting operators for each Hamiltonian term
    std::vector<std::vector<int>> noncommuting(h);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < r; ++j) {
            if (!checkCommute(hamiltonian_terms[i], onebody_operators[j])) {
                noncommuting[i].push_back(j);
            }
        }
    }

    // Convert to integer arrays for efficient operations
    std::vector<std::vector<uint8_t>> onebody_operators_intarray;
    onebody_operators_intarray.reserve(r);
    for (const auto& op : onebody_operators) {
        onebody_operators_intarray.push_back(pauliStringToIntArray(op));
    }

    std::vector<std::vector<uint8_t>> hamiltonian_terms_intarray;
    hamiltonian_terms_intarray.reserve(h);
    for (const auto& term : hamiltonian_terms) {
        hamiltonian_terms_intarray.push_back(pauliStringToIntArray(term));
    }

    // Phase table: phase_table[4*x+y] is the phase (mod 4) of sigma_x * sigma_y
    std::vector<uint8_t> phase_table(16, 0);
    phase_table[6] = 1;   // X*Y = iZ
    phase_table[11] = 1;  // Y*Z = iX
    phase_table[13] = 1;  // Z*X = iY
    phase_table[7] = 3;   // X*Z = -iY
    phase_table[9] = 3;   // Y*X = -iZ
    phase_table[14] = 3;  // Z*Y = -iX

    // Compute commutators [h_j, b_k] for all j,k pairs where they don't commute
    struct Commutator {
        int j, k;  // Hamiltonian term index j, onebody operator index k
        uint8_t log_phase;  // Phase as integer mod 4
        std::vector<uint8_t> pauli;  // Resulting Pauli string
    };

    std::vector<Commutator> commutators;
    for (int j = 0; j < h; ++j) {
        const auto& b = hamiltonian_terms_intarray[j];

        for (int k : noncommuting[j]) {
            const auto& c = onebody_operators_intarray[k];

            Commutator comm;
            comm.j = j;
            comm.k = k;

            // Compute phase
            uint8_t log_phase = 0;
            for (int i = 0; i < n; ++i) {
                log_phase += phase_table[4 * b[i] + c[i]];
            }
            comm.log_phase = log_phase % 4;

            // Compute Pauli string (XOR)
            comm.pauli.resize(n);
            for (int i = 0; i < n; ++i) {
                comm.pauli[i] = b[i] ^ c[i];
            }

            commutators.push_back(comm);
        }
    }

    // Build the tensor: for each (i, [j,k]) pair, compute b_i * [h_j, b_k]
    // The threebody_operators list is built simultaneously
    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;
    std::vector<std::string> threebody_operators;

    // Build index map for threebody operators
    std::map<std::string, int> threebody_operators_indices;

    static const Complex phase_exp_table[] = {1.0, Complex(0, 1), -1.0, Complex(0, -1)};

    for (int i = 0; i < r; ++i) {
        const auto& a = onebody_operators_intarray[i];

        for (const auto& comm : commutators) {
            // Compute a * comm.pauli
            std::vector<uint8_t> multiplied_pauli(n);
            uint8_t partial_log_phase = 0;

            for (int pos = 0; pos < n; ++pos) {
                multiplied_pauli[pos] = a[pos] ^ comm.pauli[pos];
                partial_log_phase += phase_table[4 * a[pos] + comm.pauli[pos]];
            }

            uint8_t total_log_phase = (partial_log_phase + comm.log_phase) % 4;
            Complex total_phase = 2.0 * phase_exp_table[total_log_phase];

            // Convert to string and find index in threebody_operators
            std::string pauli_string = intArrayToPauliString(multiplied_pauli);
            auto it = threebody_operators_indices.find(pauli_string);
            if (it != threebody_operators_indices.end()) {
                int l = it->second;
                indices.push_back({i, comm.j, comm.k, l});
                values.push_back(total_phase);
            }
            else {
                threebody_operators.push_back(pauli_string);
                int l = threebody_operators.size()-1;
                threebody_operators_indices[pauli_string] = l;
                indices.push_back({i, comm.j, comm.k, l});
                values.push_back(total_phase);
            }
        }
    }

    std::vector<int> shape = {r, h, r, static_cast<int>(threebody_operators.size())};
    return {SparseTensor(shape, indices, values), threebody_operators};
}

} // namespace hamiltonian_learning
