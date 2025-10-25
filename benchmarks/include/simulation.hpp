#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <vector>
#include <string>
#include <complex>
#include <map>
#include <set>
#include <Eigen/Dense>
#include <itensor/all.h>

namespace hamiltonian_learning {

using Complex = std::complex<double>;
using VectorXd = Eigen::VectorXd;
using VectorXcd = Eigen::VectorXcd;
using MatrixXd = Eigen::MatrixXd;
using MatrixXcd = Eigen::MatrixXcd;

// Hamiltonian representation as Pauli strings and coefficients
struct Hamiltonian {
    int n;  // Number of qubits
    std::vector<std::string> terms;  // Pauli strings (e.g., "XXIYZ")
    std::vector<double> coefficients;

    // Constructors
    Hamiltonian(int n, const std::vector<std::string>& terms, const std::vector<double>& coeffs);

    // Load from YAML file
    static Hamiltonian loadFromYAML(int n, const std::string& filename,
                                     const std::map<std::string, double>& couplings = {});

    // Save to YAML file
    void saveToYAML(const std::string& filename) const;

    // Sort terms alphabetically
    void sort();

    // Print Hamiltonian
    void print(double threshold = 1e-8) const;

    // Normalize coefficients so identity term has coefficient 1
    double normalizeCoeffs(const std::map<std::string, double>& expectations_dict);

    // Get normalized coefficients without modifying the Hamiltonian
    std::vector<double> normalizedCoeffs(const std::map<std::string, double>& expectations_dict) const;

    // Restrict Hamiltonian to a subsystem region
    Hamiltonian restricted(const std::vector<int>& region) const;

    // Make Hamiltonian translation-invariant by adding translates
    void makeTranslationInvariant(bool periodic);

    // Add disorder to coefficients
    void addDisorder(double magnitude,
                    const std::vector<std::string>& disorder_terms = {},
                    const std::string& distribution = "normal");
};

// Parameters for DMRG computation
struct DMRGParams {
    int chi_max = 100;           // Maximum bond dimension
    double svd_min = 1e-10;      // SVD cutoff
    double max_E_err = 1e-10;    // Energy error tolerance
    int max_sweeps = 100;        // Maximum number of sweeps
    bool quiet = false;          // Suppress output
};

// Parameters for exact diagonalization
struct EDParams {
    bool compute_full_spectrum = false;  // Compute all eigenstates or just ground state
};

// Parameters for thermal state computation by purification
struct ThermalParams {
    double dt = 0.01;            // Time step for imaginary time evolution
    int order = 2;               // Trotter order (1 or 2)
    std::string approx = "II";   // Approximation method for MPO
    int chi_max = 100;           // Maximum bond dimension
    double svd_min = 1e-8;       // SVD cutoff
    bool quiet = false;          // Suppress output
    std::string dump_file = "";  // File to dump MPS tensors at each step (empty = no dump)
};

// Parameters for expectations computation
struct ExpectationsParams {
    bool naive = false;          // Use naive computation or DFS optimization
    bool quiet = false;          // Suppress output
};

// State computation functions

// Compute ground state using DMRG (returns MPS and energy)
std::pair<itensor::MPS, double> computeGroundState(const Hamiltonian& hamiltonian,
                                                     const DMRGParams& params = DMRGParams());

// Compute purified thermal state using supersites (d=4 per site)
itensor::MPS computePurifiedEquilibriumState(const Hamiltonian& hamiltonian,
                                              double beta,
                                              const ThermalParams& params = ThermalParams());

// OLD VERSION (commented out): Uses 2N sites with dimension 2 (physical + ancilla separate)
// itensor::MPS computePurifiedEquilibriumStateOld(const Hamiltonian& hamiltonian,
//                                                  double beta,
//                                                  const ThermalParams& params = ThermalParams());

// Compute ground state using exact diagonalization
VectorXcd computeGroundStateED(const Hamiltonian& hamiltonian,
                               const EDParams& params = EDParams());

// Compute thermal state using exact diagonalization
MatrixXcd computeEquilibriumStateED(const Hamiltonian& hamiltonian,
                                    double beta,
                                    const EDParams& params = EDParams());

// Expectation value computation functions

// Compute expectations for pure state (naive single-call method)
std::vector<double> computeExpectationsPureNaive(const itensor::MPS& psi,
                                                  const std::vector<std::string>& operators);

// Compute expectations for mixed state with supersites (naive single-call method)
std::vector<double> computeExpectationsMixedNaive(const itensor::MPS& psi,
                                                   const std::vector<std::string>& operators);

// OLD VERSION (commented out): Uses 2N-site representation
// std::vector<double> computeExpectationsMixedNaiveOld(const itensor::MPS& psi,
//                                                       const std::vector<std::string>& operators);

// Compute expectations for pure state (DFS optimized)
std::vector<double> computeExpectationsPure(const itensor::MPS& psi,
                                             const std::vector<std::string>& operators);

// Compute expectations for mixed state with supersites (DFS optimized)
std::vector<double> computeExpectationsMixed(const itensor::MPS& psi,
                                              const std::vector<std::string>& operators);

// Compute expectations for mixed state with supersites (parallel DFS optimized)
// Splits operators into chunks and processes them in parallel
std::vector<double> computeExpectationsMixedParallel(const itensor::MPS& psi,
                                                      const std::vector<std::string>& operators,
                                                      int num_threads = 0);

// OLD VERSION (commented out): Uses embedding approach for 2N-site representation
// std::vector<double> computeExpectationsMixedOld(const itensor::MPS& psi,
//                                                  const std::vector<std::string>& operators);

// Compute expectations using exact diagonalization (pure state)
std::vector<double> computeExpectationsED(const VectorXcd& psi,
                                          const std::vector<std::string>& operators);

// Compute expectations using exact diagonalization (mixed state)
std::vector<double> computeExpectationsED(const MatrixXcd& rho,
                                          const std::vector<std::string>& operators);

// Helper: compute single expectation value
double computeExpectation(const std::string& pauli_string, const VectorXcd& psi);
double computeExpectation(const std::string& pauli_string, const MatrixXcd& rho);
double computeExpectation(const itensor::MPS& psi, const std::string& pauli_string);

// Helper: compute expectation value for purified MPS with supersites
// Uses Pauli ⊗ I operators on supersites (dimension 4)
double computeExpectationPurified(const itensor::MPS& psi,
                                   const std::string& pauli_string);

// OLD VERSION (commented out): Uses 2N-site representation with embedding
// double computeExpectationPurifiedOld(const itensor::MPS& psi,
//                                       const std::string& pauli_string);

// OLD VERSION (commented out): Embedding helper for 2N-site representation
// std::string embedOperatorInPurifiedSystem(const std::string& op);

} // namespace hamiltonian_learning

#endif // SIMULATION_HPP
