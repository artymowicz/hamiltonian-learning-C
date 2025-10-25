#include "pauli_utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <set>
#include <map>
#include <cmath>
#include <chrono>
#include <random>
#include <Eigen/Dense>

using namespace hamiltonian_learning;

// ============================================================================
// From test_contract.cpp
// ============================================================================

std::pair<bool, std::string> test_contractRight(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing contractRight ===" << std::endl;

    // Create a simple 3D tensor: shape [2, 3, 4]
    std::vector<int> shape = {2, 3, 4};
    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;

    // Add some entries (sparse)
    indices.push_back({0, 0, 0}); values.push_back(1.0);
    indices.push_back({0, 0, 1}); values.push_back(2.0);
    indices.push_back({0, 1, 2}); values.push_back(3.0);
    indices.push_back({0, 1, 3}); values.push_back(4.0);
    indices.push_back({1, 0, 0}); values.push_back(5.0);
    indices.push_back({1, 2, 1}); values.push_back(6.0);
    indices.push_back({1, 2, 2}); values.push_back(7.0);

    SparseTensor T(shape, indices, values, false);

    // Create a vector to contract with
    VectorXc v(4);
    v << 1.0, 2.0, 3.0, 4.0;

    log << "Original tensor shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
    log << "Number of nonzero entries: " << values.size() << std::endl;
    log << "Vector to contract: [";
    for (int i = 0; i < v.size(); ++i) {
        log << v[i];
        if (i < v.size() - 1) log << ", ";
    }
    log << "]" << std::endl << std::endl;

    // Contract along rightmost dimension
    auto result = T.contractRight(v);

    log << "Result shape: [" << result.shape[0] << ", " << result.shape[1] << "]" << std::endl;
    log << "Number of nonzero entries: " << result.values.size() << std::endl;

    // Expected values:
    // T[0,0,:] * v = 1*1 + 2*2 = 5
    // T[0,1,:] * v = 3*3 + 4*4 = 25
    // T[1,0,:] * v = 5*1 = 5
    // T[1,2,:] * v = 6*2 + 7*3 = 33
    std::map<std::vector<int>, Complex> expected;
    expected[{0, 0}] = 5.0;
    expected[{0, 1}] = 25.0;
    expected[{1, 0}] = 5.0;
    expected[{1, 2}] = 33.0;

    // Build map of actual results
    std::map<std::vector<int>, Complex> actual;
    for (size_t i = 0; i < result.values.size(); ++i) {
        actual[result.indices[i]] = result.values[i];
    }

    log << "Result entries:" << std::endl;
    for (size_t i = 0; i < result.values.size(); ++i) {
        log << "  [" << result.indices[i][0] << ", " << result.indices[i][1] << "] = "
                  << result.values[i] << std::endl;
    }
    log << std::endl;

    // Verify all expected entries are present and match
    log << "Verification:" << std::endl;
    for (const auto& [idx, exp_val] : expected) {
        if (actual.find(idx) == actual.end()) {
            log << "  [" << idx[0] << ", " << idx[1] << "] MISSING (expected " << exp_val << ")" << std::endl;
            passed = false;
        } else {
            Complex act_val = actual[idx];
            double diff = std::abs(exp_val - act_val);
            log << "  [" << idx[0] << ", " << idx[1] << "] expected=" << exp_val
                      << " actual=" << act_val << " diff=" << std::scientific << std::setprecision(2) << diff;
            if (diff > threshold) {
                log << " FAILED" << std::endl;
                passed = false;
            } else {
                log << " OK" << std::endl;
            }
        }
    }

    // Verify no unexpected entries exist (or they're zero within threshold)
    for (const auto& [idx, act_val] : actual) {
        if (expected.find(idx) == expected.end()) {
            double magnitude = std::abs(act_val);
            log << "  [" << idx[0] << ", " << idx[1] << "] UNEXPECTED (value=" << act_val << ")";
            if (magnitude > threshold) {
                log << " FAILED" << std::endl;
                passed = false;
            } else {
                log << " OK (within threshold)" << std::endl;
            }
        }
    }

    log << std::endl;
    log << "Threshold: " << std::scientific << std::setprecision(4) << threshold << std::endl;

    return {passed, log.str()};
}

std::pair<bool, std::string> test_contractLeft(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing contractLeft ===" << std::endl;

    // Create a simple 3D tensor: shape [3, 2, 4]
    std::vector<int> shape = {3, 2, 4};
    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;

    // Add some entries
    indices.push_back({0, 0, 0}); values.push_back(1.0);
    indices.push_back({0, 1, 2}); values.push_back(2.0);
    indices.push_back({1, 0, 1}); values.push_back(3.0);
    indices.push_back({1, 1, 0}); values.push_back(4.0);
    indices.push_back({2, 0, 3}); values.push_back(5.0);
    indices.push_back({2, 1, 1}); values.push_back(6.0);

    SparseTensor T(shape, indices, values, false);

    // Create a vector to contract with
    VectorXc v(3);
    v << 1.0, 2.0, 3.0;

    log << "Original tensor shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
    log << "Number of nonzero entries: " << values.size() << std::endl;
    log << "Vector to contract: [";
    for (int i = 0; i < v.size(); ++i) {
        log << v[i];
        if (i < v.size() - 1) log << ", ";
    }
    log << "]" << std::endl << std::endl;

    // Contract along leftmost dimension
    auto result = T.contractLeft(v);

    log << "Result shape: [" << result.shape[0] << ", " << result.shape[1] << "]" << std::endl;
    log << "Number of nonzero entries: " << result.values.size() << std::endl;

    // Expected values:
    // T[:,0,0] * v = 1*1 = 1
    // T[:,0,1] * v = 3*2 = 6
    // T[:,0,3] * v = 5*3 = 15
    // T[:,1,0] * v = 4*2 = 8
    // T[:,1,1] * v = 6*3 = 18
    // T[:,1,2] * v = 2*1 = 2
    std::map<std::vector<int>, Complex> expected;
    expected[{0, 0}] = 1.0;
    expected[{0, 1}] = 6.0;
    expected[{0, 3}] = 15.0;
    expected[{1, 0}] = 8.0;
    expected[{1, 1}] = 18.0;
    expected[{1, 2}] = 2.0;

    // Build map of actual results
    std::map<std::vector<int>, Complex> actual;
    for (size_t i = 0; i < result.values.size(); ++i) {
        actual[result.indices[i]] = result.values[i];
    }

    log << "Result entries:" << std::endl;
    for (size_t i = 0; i < result.values.size(); ++i) {
        log << "  [" << result.indices[i][0] << ", " << result.indices[i][1] << "] = "
                  << result.values[i] << std::endl;
    }
    log << std::endl;

    // Verify all expected entries are present and match
    log << "Verification:" << std::endl;
    for (const auto& [idx, exp_val] : expected) {
        if (actual.find(idx) == actual.end()) {
            log << "  [" << idx[0] << ", " << idx[1] << "] MISSING (expected " << exp_val << ")" << std::endl;
            passed = false;
        } else {
            Complex act_val = actual[idx];
            double diff = std::abs(exp_val - act_val);
            log << "  [" << idx[0] << ", " << idx[1] << "] expected=" << exp_val
                      << " actual=" << act_val << " diff=" << std::scientific << std::setprecision(2) << diff;
            if (diff > threshold) {
                log << " FAILED" << std::endl;
                passed = false;
            } else {
                log << " OK" << std::endl;
            }
        }
    }

    // Verify no unexpected entries exist (or they're zero within threshold)
    for (const auto& [idx, act_val] : actual) {
        if (expected.find(idx) == expected.end()) {
            double magnitude = std::abs(act_val);
            log << "  [" << idx[0] << ", " << idx[1] << "] UNEXPECTED (value=" << act_val << ")";
            if (magnitude > threshold) {
                log << " FAILED" << std::endl;
                passed = false;
            } else {
                log << " OK (within threshold)" << std::endl;
            }
        }
    }

    log << std::endl;
    log << "Threshold: " << std::scientific << std::setprecision(4) << threshold << std::endl;

    return {passed, log.str()};
}

std::pair<bool, std::string> test_contractRight_with_complex(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing contractRight with complex values ===" << std::endl;

    // Create a 2D tensor with complex values
    std::vector<int> shape = {3, 2};
    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;

    indices.push_back({0, 0}); values.push_back(Complex(1.0, 1.0));
    indices.push_back({0, 1}); values.push_back(Complex(2.0, -1.0));
    indices.push_back({1, 0}); values.push_back(Complex(0.0, 2.0));
    indices.push_back({2, 1}); values.push_back(Complex(3.0, 0.0));

    SparseTensor T(shape, indices, values, false);

    // Create a complex vector to contract with
    VectorXc v(2);
    v << Complex(1.0, 0.0), Complex(0.0, 1.0);

    log << "Original tensor shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
    log << "Vector to contract: [" << v[0] << ", " << v[1] << "]" << std::endl << std::endl;

    // Contract along rightmost dimension
    auto result = T.contractRight(v);

    log << "Result shape: [" << result.shape[0] << "]" << std::endl;
    log << "Number of nonzero entries: " << result.values.size() << std::endl;

    // Expected values:
    // T[0,:] * v = (1+i)*1 + (2-i)*i = (1+i) + (2i-i^2) = (1+i) + (2i+1) = (2+3i)
    // T[1,:] * v = 2i*1 = 2i
    // T[2,:] * v = 3*i = 3i
    std::map<std::vector<int>, Complex> expected;
    expected[{0}] = Complex(2.0, 3.0);
    expected[{1}] = Complex(0.0, 2.0);
    expected[{2}] = Complex(0.0, 3.0);

    // Build map of actual results
    std::map<std::vector<int>, Complex> actual;
    for (size_t i = 0; i < result.values.size(); ++i) {
        actual[result.indices[i]] = result.values[i];
    }

    log << "Result entries:" << std::endl;
    for (size_t i = 0; i < result.values.size(); ++i) {
        log << "  [" << result.indices[i][0] << "] = " << result.values[i] << std::endl;
    }
    log << std::endl;

    // Verify all expected entries are present and match
    log << "Verification:" << std::endl;
    for (const auto& [idx, exp_val] : expected) {
        if (actual.find(idx) == actual.end()) {
            log << "  [" << idx[0] << "] MISSING (expected " << exp_val << ")" << std::endl;
            passed = false;
        } else {
            Complex act_val = actual[idx];
            double diff = std::abs(exp_val - act_val);
            log << "  [" << idx[0] << "] expected=" << exp_val
                      << " actual=" << act_val << " diff=" << std::scientific << std::setprecision(2) << diff;
            if (diff > threshold) {
                log << " FAILED" << std::endl;
                passed = false;
            } else {
                log << " OK" << std::endl;
            }
        }
    }

    // Verify no unexpected entries exist (or they're zero within threshold)
    for (const auto& [idx, act_val] : actual) {
        if (expected.find(idx) == expected.end()) {
            double magnitude = std::abs(act_val);
            log << "  [" << idx[0] << "] UNEXPECTED (value=" << act_val << ")";
            if (magnitude > threshold) {
                log << " FAILED" << std::endl;
                passed = false;
            } else {
                log << " OK (within threshold)" << std::endl;
            }
        }
    }

    log << std::endl;
    log << "Threshold: " << std::scientific << std::setprecision(4) << threshold << std::endl;

    return {passed, log.str()};
}

// ============================================================================
// From test_utils_tensor_building.cpp
// ============================================================================

std::pair<bool, std::string> test_buildKLocalPaulis1D() {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing buildKLocalPaulis1D ===" << std::endl;
    log << std::endl;

    // Test 1: n=5, k=1, periodic=false
    log << "Test 1: n=5, k=1, periodic=false" << std::endl;
    auto result1 = buildKLocalPaulis1D(5, 1, false);
    std::set<std::string> result1_set(result1.begin(), result1.end());

    std::set<std::string> expected1 = {
        "IIIII",
        "XIIII", "YIIII", "ZIIII",
        "IXIII", "IYIII", "IZIII",
        "IIXII", "IIYII", "IIZII",
        "IIIXI", "IIIYI", "IIIZI",
        "IIIIX", "IIIIY", "IIIIZ"
    };

    log << "  Expected " << expected1.size() << " operators, got " << result1.size() << std::endl;

    if (result1_set != expected1) {
        passed = false;
        log << "  MISMATCH!" << std::endl;

        // Find missing operators
        std::set<std::string> missing;
        std::set_difference(expected1.begin(), expected1.end(),
                          result1_set.begin(), result1_set.end(),
                          std::inserter(missing, missing.begin()));
        if (!missing.empty()) {
            log << "  Missing operators: ";
            for (const auto& op : missing) log << op << " ";
            log << std::endl;
        }

        // Find unexpected operators
        std::set<std::string> unexpected;
        std::set_difference(result1_set.begin(), result1_set.end(),
                          expected1.begin(), expected1.end(),
                          std::inserter(unexpected, unexpected.begin()));
        if (!unexpected.empty()) {
            log << "  Unexpected operators: ";
            for (const auto& op : unexpected) log << op << " ";
            log << std::endl;
        }
    } else {
        log << "  OK" << std::endl;
    }
    log << std::endl;

    // Test 2: n=5, k=2, periodic=false
    log << "Test 2: n=5, k=2, periodic=false" << std::endl;
    auto result2 = buildKLocalPaulis1D(5, 2, false);
    std::set<std::string> result2_set(result2.begin(), result2.end());

    std::set<std::string> expected2 = {
        // k=0 operator (identity)
        "IIIII",
        // k=1 operators (single-site)
        "XIIII", "YIIII", "ZIIII",
        "IXIII", "IYIII", "IZIII",
        "IIXII", "IIYII", "IIZII",
        "IIIXI", "IIIYI", "IIIZI",
        "IIIIX", "IIIIY", "IIIIZ",
        // k=2 operators (nearest-neighbor)
        "XXIII", "XYIII", "XZIII", "YXIII", "YYIII", "YZIII", "ZXIII", "ZYIII", "ZZIII",
        "IXXII", "IXYII", "IXZII", "IYXII", "IYYII", "IYZII", "IZXII", "IZYII", "IZZII",
        "IIXXI", "IIXYI", "IIXZI", "IIYXI", "IIYYI", "IIYZI", "IIZXI", "IIZYI", "IIZZI",
        "IIIXX", "IIIXY", "IIIXZ", "IIIYX", "IIIYY", "IIIYZ", "IIIZX", "IIIZY", "IIIZZ"
    };

    log << "  Expected " << expected2.size() << " operators, got " << result2.size() << std::endl;

    if (result2_set != expected2) {
        passed = false;
        log << "  MISMATCH!" << std::endl;

        // Find missing operators
        std::set<std::string> missing;
        std::set_difference(expected2.begin(), expected2.end(),
                          result2_set.begin(), result2_set.end(),
                          std::inserter(missing, missing.begin()));
        if (!missing.empty()) {
            log << "  Missing operators (" << missing.size() << "): ";
            for (const auto& op : missing) log << op << " ";
            log << std::endl;
        }

        // Find unexpected operators
        std::set<std::string> unexpected;
        std::set_difference(result2_set.begin(), result2_set.end(),
                          expected2.begin(), expected2.end(),
                          std::inserter(unexpected, unexpected.begin()));
        if (!unexpected.empty()) {
            log << "  Unexpected operators (" << unexpected.size() << "): ";
            for (const auto& op : unexpected) log << op << " ";
            log << std::endl;
        }
    } else {
        log << "  OK" << std::endl;
    }
    log << std::endl;

    return {passed, log.str()};
}

std::pair<bool, std::string> test_buildMultiplicationTensor(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing buildMultiplicationTensor ===" << std::endl;
    log << std::endl;

    // Use n=10, k=2 as specified in the plan
    int n = 10;
    log << "Generating operators for n=" << n << ", k=2..." << std::endl;
    auto onebody_operators = buildKLocalPaulis1D(n, 2, false);
    log << "Number of operators: " << onebody_operators.size() << std::endl;
    log << std::endl;

    log << "Building multiplication tensor..." << std::endl;
    auto [mult_tensor, twobody_operators] = buildMultiplicationTensor(onebody_operators);
    log << "Multiplication tensor shape: [" << mult_tensor.shape[0] << ", "
              << mult_tensor.shape[1] << ", " << mult_tensor.shape[2] << "]" << std::endl;
    log << "Number of two-body operators: " << twobody_operators.size() << std::endl;
    log << "Number of nonzero entries: " << mult_tensor.values.size() << std::endl;
    log << std::endl;

    // Build a map for fast lookup: (i,j) -> (k, value)
    std::map<std::pair<int,int>, std::pair<int, Complex>> tensor_map;
    for (size_t idx = 0; idx < mult_tensor.values.size(); ++idx) {
        int i = mult_tensor.indices[idx][0];
        int j = mult_tensor.indices[idx][1];
        int k = mult_tensor.indices[idx][2];
        Complex v = mult_tensor.values[idx];
        tensor_map[{i, j}] = {k, v};
    }

    // Build a map for fast lookup of operator index: string -> index
    std::map<std::string, int> twobody_map;
    for (size_t i = 0; i < twobody_operators.size(); ++i) {
        twobody_map[twobody_operators[i]] = i;
    }

    log << "Verifying all Pauli multiplications..." << std::endl;
    int num_verified = 0;
    int num_errors = 0;
    int max_errors_to_print = 10;

    for (size_t i = 0; i < onebody_operators.size(); ++i) {
        for (size_t j = 0; j < onebody_operators.size(); ++j) {
            // Compute expected result using multiplyPaulis
            auto [product_string, phase] = multiplyPaulis(onebody_operators[i], onebody_operators[j]);

            // Find expected index in twobody_operators
            auto it = twobody_map.find(product_string);
            if (it == twobody_map.end()) {
                if (num_errors < max_errors_to_print) {
                    log << "  ERROR: Product " << product_string << " not found in twobody_operators" << std::endl;
                    log << "    From: " << onebody_operators[i] << " * " << onebody_operators[j] << std::endl;
                }
                num_errors++;
                passed = false;
                continue;
            }
            int k_expected = it->second;

            // Look up actual result in tensor
            auto tensor_it = tensor_map.find({static_cast<int>(i), static_cast<int>(j)});
            if (tensor_it == tensor_map.end()) {
                if (num_errors < max_errors_to_print) {
                    log << "  ERROR: Entry (" << i << "," << j << ") missing in tensor" << std::endl;
                    log << "    Operators: " << onebody_operators[i] << " * " << onebody_operators[j] << std::endl;
                }
                num_errors++;
                passed = false;
                continue;
            }

            int k_actual = tensor_it->second.first;
            Complex phase_actual = tensor_it->second.second;

            // Verify k matches
            if (k_actual != k_expected) {
                if (num_errors < max_errors_to_print) {
                    log << "  ERROR: Index mismatch for (" << i << "," << j << ")" << std::endl;
                    log << "    Operators: " << onebody_operators[i] << " * " << onebody_operators[j] << std::endl;
                    log << "    Expected k=" << k_expected << " (" << twobody_operators[k_expected] << ")" << std::endl;
                    log << "    Got k=" << k_actual << " (" << twobody_operators[k_actual] << ")" << std::endl;
                }
                num_errors++;
                passed = false;
                continue;
            }

            // Verify phase matches
            double phase_diff = std::abs(phase - phase_actual);
            if (phase_diff > threshold) {
                if (num_errors < max_errors_to_print) {
                    log << "  ERROR: Phase mismatch for (" << i << "," << j << ")" << std::endl;
                    log << "    Operators: " << onebody_operators[i] << " * " << onebody_operators[j] << std::endl;
                    log << "    Expected phase=" << phase << std::endl;
                    log << "    Got phase=" << phase_actual << std::endl;
                    log << "    Difference=" << std::scientific << phase_diff << std::endl;
                }
                num_errors++;
                passed = false;
                continue;
            }

            num_verified++;
        }
    }

    if (num_errors > max_errors_to_print) {
        log << "  ... and " << (num_errors - max_errors_to_print) << " more errors" << std::endl;
    }

    log << std::endl;
    log << "Verified " << num_verified << " multiplications" << std::endl;
    log << "Errors: " << num_errors << std::endl;
    log << "Threshold: " << std::scientific << std::setprecision(4) << threshold << std::endl;

    return {passed, log.str()};
}

std::pair<bool, std::string> test_buildTripleProductTensor(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing buildTripleProductTensor ===" << std::endl;
    log << std::endl;

    // Use n=7, k=2 as specified in the plan
    int n = 7;
    log << "Generating operators for n=" << n << ", k=2..." << std::endl;
    auto onebody_operators = buildKLocalPaulis1D(n, 2, false);
    log << "Number of operators: " << onebody_operators.size() << std::endl;
    log << std::endl;

    // Use onebody_operators for both B and C (as specified in plan)
    log << "Building triple product tensor..." << std::endl;
    auto [triple_product_tensor, threebody_operators] = buildTripleProductTensor(onebody_operators, onebody_operators);
    log << "Triple product tensor shape: [" << triple_product_tensor.shape[0] << ", "
              << triple_product_tensor.shape[1] << ", " << triple_product_tensor.shape[2]
              << ", " << triple_product_tensor.shape[3] << "]" << std::endl;
    log << "Number of three-body operators: " << threebody_operators.size() << std::endl;
    log << "Number of nonzero entries: " << triple_product_tensor.values.size() << std::endl;
    log << std::endl;

    // Build a map for fast lookup: (i,j,k) -> [(l, value), ...]
    std::map<std::tuple<int,int,int>, std::vector<std::pair<int, Complex>>> tensor_map;
    for (size_t idx = 0; idx < triple_product_tensor.values.size(); ++idx) {
        int i = triple_product_tensor.indices[idx][0];
        int j = triple_product_tensor.indices[idx][1];
        int k = triple_product_tensor.indices[idx][2];
        int l = triple_product_tensor.indices[idx][3];
        Complex v = triple_product_tensor.values[idx];
        tensor_map[{i, j, k}].push_back({l, v});
    }

    // Build a map for fast lookup of operator index: string -> index
    std::map<std::string, int> threebody_map;
    for (size_t i = 0; i < threebody_operators.size(); ++i) {
        threebody_map[threebody_operators[i]] = i;
    }

    log << "Verifying all triple products A[B,C] = ABC - ACB..." << std::endl;
    int num_verified = 0;
    int num_errors = 0;
    int max_errors_to_print = 10;

    for (size_t i = 0; i < onebody_operators.size(); ++i) {
        for (size_t j = 0; j < onebody_operators.size(); ++j) {
            for (size_t k = 0; k < onebody_operators.size(); ++k) {
                std::string A = onebody_operators[i];
                std::string B = onebody_operators[j];
                std::string C = onebody_operators[k];

                // Compute A[B,C] = ABC - ACB using multiplyPaulis
                auto [BC, phase_BC] = multiplyPaulis(B, C);
                auto [ABC, phase_ABC] = multiplyPaulis(A, BC);
                auto [AC, phase_AC] = multiplyPaulis(A, C);
                auto [ACB, phase_ACB] = multiplyPaulis(AC, B);

                // Sanity check: ABC and ACB Pauli strings should match
                if (ABC != ACB) {
                    if (num_errors < max_errors_to_print) {
                        log << "  ERROR: Pauli string mismatch for A[B,C]" << std::endl;
                        log << "    A=" << A << ", B=" << B << ", C=" << C << std::endl;
                        log << "    ABC=" << ABC << ", ACB=" << ACB << std::endl;
                    }
                    num_errors++;
                    passed = false;
                    continue;
                }

                // Compute commutator phase
                Complex comm_phase = phase_ABC * phase_BC - phase_ACB * phase_AC;
                double comm_magnitude = std::abs(comm_phase);

                // Look up tensor entry
                auto tensor_it = tensor_map.find({static_cast<int>(i), static_cast<int>(j), static_cast<int>(k)});

                if (comm_magnitude < threshold) {
                    // Result should be zero - no tensor entry should exist
                    if (tensor_it != tensor_map.end()) {
                        if (num_errors < max_errors_to_print) {
                            log << "  ERROR: Unexpected tensor entry for zero commutator" << std::endl;
                            log << "    A=" << A << ", B=" << B << ", C=" << C << std::endl;
                            log << "    Commutator magnitude=" << std::scientific << comm_magnitude << " (below threshold)" << std::endl;
                            log << "    But found " << tensor_it->second.size() << " tensor entries" << std::endl;
                        }
                        num_errors++;
                        passed = false;
                    }
                    num_verified++;
                } else {
                    // Result is non-zero - exactly one tensor entry should exist
                    if (tensor_it == tensor_map.end()) {
                        if (num_errors < max_errors_to_print) {
                            log << "  ERROR: Missing tensor entry for non-zero commutator" << std::endl;
                            log << "    A=" << A << ", B=" << B << ", C=" << C << std::endl;
                            log << "    Commutator magnitude=" << std::scientific << comm_magnitude << std::endl;
                            log << "    Expected phase=" << comm_phase << std::endl;
                        }
                        num_errors++;
                        passed = false;
                        continue;
                    }

                    if (tensor_it->second.size() != 1) {
                        if (num_errors < max_errors_to_print) {
                            log << "  ERROR: Multiple tensor entries for (" << i << "," << j << "," << k << ")" << std::endl;
                            log << "    A=" << A << ", B=" << B << ", C=" << C << std::endl;
                            log << "    Found " << tensor_it->second.size() << " entries (expected 1)" << std::endl;
                        }
                        num_errors++;
                        passed = false;
                        continue;
                    }

                    int l_actual = tensor_it->second[0].first;
                    Complex phase_actual = tensor_it->second[0].second;

                    // Verify the Pauli string matches
                    if (threebody_operators[l_actual] != ABC) {
                        if (num_errors < max_errors_to_print) {
                            log << "  ERROR: Pauli string mismatch in tensor" << std::endl;
                            log << "    A=" << A << ", B=" << B << ", C=" << C << std::endl;
                            log << "    Expected: " << ABC << std::endl;
                            log << "    Got: " << threebody_operators[l_actual] << std::endl;
                        }
                        num_errors++;
                        passed = false;
                        continue;
                    }

                    // Verify the phase matches
                    double phase_diff = std::abs(comm_phase - phase_actual);
                    if (phase_diff > threshold) {
                        if (num_errors < max_errors_to_print) {
                            log << "  ERROR: Phase mismatch for (" << i << "," << j << "," << k << ")" << std::endl;
                            log << "    A=" << A << ", B=" << B << ", C=" << C << std::endl;
                            log << "    Expected phase=" << comm_phase << std::endl;
                            log << "    Got phase=" << phase_actual << std::endl;
                            log << "    Difference=" << std::scientific << phase_diff << std::endl;
                        }
                        num_errors++;
                        passed = false;
                        continue;
                    }

                    num_verified++;
                }
            }
        }
    }

    if (num_errors > max_errors_to_print) {
        log << "  ... and " << (num_errors - max_errors_to_print) << " more errors" << std::endl;
    }

    log << std::endl;
    log << "Verified " << num_verified << " triple products" << std::endl;
    log << "Errors: " << num_errors << std::endl;
    log << "Threshold: " << std::scientific << std::setprecision(4) << threshold << std::endl;

    return {passed, log.str()};
}

// ============================================================================
// From test_parallelization.cpp
// ============================================================================

std::pair<bool, std::string> test_contractRight_parallel(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing contractRight parallelization ===" << std::endl;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> value_dis(-1.0, 1.0);

    // Create a test sparse tensor (rank-2, representing a sparse matrix)
    int m = 100;
    int n = 80;
    std::vector<int> shape = {m, n};

    // Create random sparse matrix with 60% sparsity
    std::vector<std::vector<int>> indices;
    std::vector<Complex> values;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // Roll a die: 40% chance of nonzero entry (60% sparsity)
            if (dis(gen) < 0.4) {
                indices.push_back({i, j});
                values.push_back(Complex(value_dis(gen), value_dis(gen)));
            }
        }
    }

    SparseTensor sparse_mat(shape, indices, values);
    log << "  Sparse matrix: " << m << " x " << n
        << " with " << sparse_mat.values.size() << " nonzeros" << std::endl;
    double actual_sparsity = 1.0 - (double)sparse_mat.values.size() / (m * n);
    log << "  Actual sparsity: " << (actual_sparsity * 100) << "%" << std::endl;

    // Create a dense matrix to contract with
    int k = 50;
    MatrixXd dense_mat = MatrixXd::Random(n, k);
    log << "  Dense matrix: " << n << " x " << k << std::endl;

    // Test single-threaded version
    auto start_st = std::chrono::high_resolution_clock::now();
    MatrixXd result_st = sparse_mat.contractRightSingleThreaded(dense_mat);
    auto end_st = std::chrono::high_resolution_clock::now();
    double time_st = std::chrono::duration<double>(end_st - start_st).count();

    // Test parallel version
    auto start_par = std::chrono::high_resolution_clock::now();
    MatrixXd result_par = sparse_mat.contractRight(dense_mat);
    auto end_par = std::chrono::high_resolution_clock::now();
    double time_par = std::chrono::duration<double>(end_par - start_par).count();

    // Compare results
    double max_diff = (result_st - result_par).cwiseAbs().maxCoeff();
    double rel_error = max_diff / (result_st.cwiseAbs().maxCoeff() + 1e-10);

    log << "  Single-threaded time: " << time_st << " s" << std::endl;
    log << "  Parallel time: " << time_par << " s" << std::endl;
    if (time_par > 0) {
        log << "  Speedup: " << time_st / time_par << "x" << std::endl;
    }
    log << "  Maximum difference: " << max_diff << std::endl;
    log << "  Relative error: " << rel_error << std::endl;

    if (rel_error < threshold) {
        log << "  Status: ✓ PASSED" << std::endl;
    } else {
        passed = false;
        log << "  Status: ✗ FAILED (rel_error: " << rel_error << " > threshold: " << threshold << ")" << std::endl;
    }
    log << std::endl;

    return {passed, log.str()};
}

std::pair<bool, std::string> test_buildMultiplicationTensor_parallel(double threshold) {
    std::stringstream log;
    bool passed = true;

    log << "=== Testing buildMultiplicationTensor parallelization ===" << std::endl;

    // Generate operators for n=5, k=2
    int n = 5;
    int k = 2;
    auto operators = buildKLocalPaulis1D(n, k, false);

    log << "  Testing with " << operators.size() << " operators" << std::endl;

    // Test single-threaded version
    auto start_st = std::chrono::high_resolution_clock::now();
    auto [tensor_st, ops_st] = buildMultiplicationTensorSingleThreaded(operators);
    auto end_st = std::chrono::high_resolution_clock::now();
    double time_st = std::chrono::duration<double>(end_st - start_st).count();

    // Test parallel version
    auto start_par = std::chrono::high_resolution_clock::now();
    auto [tensor_par, ops_par] = buildMultiplicationTensor(operators);
    auto end_par = std::chrono::high_resolution_clock::now();
    double time_par = std::chrono::duration<double>(end_par - start_par).count();

    log << "  Single-threaded time: " << time_st << " s" << std::endl;
    log << "  Parallel time: " << time_par << " s" << std::endl;
    if (time_par > 0) {
        log << "  Speedup: " << time_st / time_par << "x" << std::endl;
    }

    // Compare results
    // Check shape
    if (tensor_st.shape != tensor_par.shape) {
        log << "  ✗ Shape mismatch!" << std::endl;
        passed = false;
    }

    // Check number of nonzeros
    if (tensor_st.values.size() != tensor_par.values.size()) {
        log << "  ✗ Number of nonzeros mismatch: "
            << tensor_st.values.size() << " vs " << tensor_par.values.size() << std::endl;
        passed = false;
    }

    // Check operator lists
    if (ops_st != ops_par) {
        log << "  ✗ Operator lists mismatch!" << std::endl;
        passed = false;
    }

    if (passed) {
        // Convert to dense for comparison (order tensors first)
        tensor_st.order();
        tensor_par.order();

        // Check indices match
        bool indices_match = true;
        for (size_t i = 0; i < tensor_st.indices.size(); ++i) {
            if (tensor_st.indices[i] != tensor_par.indices[i]) {
                indices_match = false;
                break;
            }
        }

        if (!indices_match) {
            log << "  ✗ Indices don't match after ordering!" << std::endl;
            passed = false;
        } else {
            // Check values match
            double max_diff = 0.0;
            for (size_t i = 0; i < tensor_st.values.size(); ++i) {
                double diff = std::abs(tensor_st.values[i] - tensor_par.values[i]);
                max_diff = std::max(max_diff, diff);
            }

            log << "  Maximum value difference: " << max_diff << std::endl;

            if (max_diff > threshold) {
                log << "  ✗ Values differ by more than tolerance! (max_diff: " << max_diff << " > threshold: " << threshold << ")" << std::endl;
                passed = false;
            }
        }
    }

    log << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    log << std::endl;

    return {passed, log.str()};
}

// ============================================================================
// Main function to run all tests
// ============================================================================

int main(int argc, char* argv[]) {
    double threshold = 1e-10;
    if (argc > 1) {
        threshold = std::stod(argv[1]);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "        Utils Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Threshold: " << std::scientific << threshold << std::endl;
    std::cout << std::endl;

    int total = 0, passed_count = 0;

    // Test 1: contractRight
    auto [passed1, log1] = test_contractRight(threshold);
    total++;
    if (passed1) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 1: test_contractRight" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log1;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 2: contractLeft
    auto [passed2, log2] = test_contractLeft(threshold);
    total++;
    if (passed2) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 2: test_contractLeft" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log2;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 3: contractRight with complex values
    auto [passed3, log3] = test_contractRight_with_complex(threshold);
    total++;
    if (passed3) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 3: test_contractRight_with_complex" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log3;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 4: buildKLocalPaulis1D
    auto [passed4, log4] = test_buildKLocalPaulis1D();
    total++;
    if (passed4) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 4: test_buildKLocalPaulis1D" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log4;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 5: buildMultiplicationTensor
    auto [passed5, log5] = test_buildMultiplicationTensor(threshold);
    total++;
    if (passed5) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 5: test_buildMultiplicationTensor" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log5;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 6: buildTripleProductTensor
    auto [passed6, log6] = test_buildTripleProductTensor(threshold);
    total++;
    if (passed6) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 6: test_buildTripleProductTensor" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log6;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 7: contractRight parallelization
    auto [passed7, log7] = test_contractRight_parallel(threshold);
    total++;
    if (passed7) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 7: test_contractRight_parallel" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log7;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::endl;
    }

    // Test 8: buildMultiplicationTensor parallelization
    auto [passed8, log8] = test_buildMultiplicationTensor_parallel(threshold);
    total++;
    if (passed8) {
        passed_count++;
    } else {
        std::cout << "FAILED TEST 8: test_buildMultiplicationTensor_parallel" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << log8;
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
