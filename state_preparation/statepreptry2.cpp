#include <clang/Quantum/quintrinsics.h>
#include <clang/Quantum/qexpr.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <list>
#include <clang/Quantum/qlist.h>
#include <clang/Quantum/datalist.h>
#include <quantum_full_state_simulator_backend.h>
#include <qexpr_utils.h>
#include <bits/stdc++.h>



typedef size_t index_t;
using namespace std;
#define _EPS 1.0e-10




std::vector<std::vector<double>> createIdentityMatrix(int size) {
    std::vector<std::vector<double>> identityMatrix(size, std::vector<double>(size, 0.0));
    
    for (int i = 0; i < size; ++i) {
        identityMatrix[i][i] = 1.0;
    }
    
    return identityMatrix;
}

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<double>> kroneckerProduct(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int aRows = A.size();
    int aCols = A[0].size();
    int bRows = B.size();
    int bCols = B[0].size();
    
    int resultRows = aRows * bRows;
    int resultCols = aCols * bCols;
    
    std::vector<std::vector<double>> result(resultRows, std::vector<double>(resultCols, 0.0));
    
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < aCols; ++j) {
            for (int k = 0; k < bRows; ++k) {
                for (int l = 0; l < bCols; ++l) {
                    result[i * bRows + k][j * bCols + l] = A[i][j] * B[k][l];
                }
            }
        }
    }
    
    return result;
}

std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>>& matrix,
                                      const std::vector<double>& vector) {
    int m = matrix.size();      // Number of rows in matrix
    int n = matrix[0].size();   // Number of columns in matrix
    int vSize = vector.size();  // Size of the vector

    // Result vector will have size m (number of rows in matrix)
    std::vector<double> result(m, 0);

    // Perform matrix-vector multiplication
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

std::tuple<complex<double>, double, double> angles(const pair<complex<double>, complex<double>> &pair_of_complex)
{
    complex<double> a_complex = pair_of_complex.first;
    complex<double> b_complex = pair_of_complex.second;

    cout << "2 values passed to calculate angles: " << a_complex << " " << b_complex << endl;

    double mag_a = std::abs(a_complex);
    double final_r = std::sqrt(mag_a * mag_a + std::abs(b_complex) * std::abs(b_complex));
    double theta, phi, final_t;

    if (final_r < _EPS)
    {
        theta = 0;
        phi = 0;
        final_r = 0;
        final_t = 0;
    }
    else
    {
        theta = 2 * std::acos(mag_a / final_r);
        double a_arg = std::arg(a_complex);
        double b_arg = std::arg(b_complex);
        final_t = a_arg + b_arg;
        phi = b_arg - a_arg;
    }

    complex<double> remains = final_r * exp(complex<double>(0, final_t / 2));
    return {remains, theta, phi};
}

std::tuple<vector<complex<double>>, vector<double>, vector<double>> rotations(const vector<complex<double>> &local_param)
{
    vector<complex<double>> remaining_vector;
    size_t param_len = local_param.size();
    // double* thetas = new double[param_len / 2];
    // double* phis = new double[param_len / 2];

    vector<double> thetas;
    vector<double> phis;
    cout << "Length of the parameters passed: " << param_len << endl;

    for (size_t i = 0; i < param_len / 2; ++i)
    {
        cout << "Iteration in calc block angles: " << i << endl;
        cout << "Indexes i and i + 1 for loop: " << 2 * i << " , " << 2 * (i + 1) << endl;

        auto [remains, add_theta, add_phi] = angles({local_param[2 * i], local_param[2 * i + 1]});

        cout << "The theta to be added is: " << add_theta << endl;
        cout << "The phi to be added is: " << add_phi << endl;
        cout << "The remaining parameters are: " << remains << endl;

        remaining_vector.push_back(remains);

        // Rotations for all imaginary qubits of the full vector
        thetas.push_back(-add_theta);
        phis.push_back(-add_phi);
        // thetas[i] = -add_theta;
        // phis[i] = -add_phi;
    }

    return {remaining_vector, thetas, phis}; 
}

std::vector<double> correct_angles(vector<double> angles){

    int angles_len = angles.size();
    cout << "the length of angles : " << angles_len << endl;
    int local_qubits = log2(angles_len) + 1;
    cout << "The number of local qubits is : " << local_qubits << endl;

    std::vector<std::vector<double>> angle_weight = {
            {0.5, 0.5},
            {0.5, -0.5}
    }; 


    auto idmat = createIdentityMatrix(pow(2,local_qubits - 2));

    cout << "This is identity matrix" << endl;
    printMatrix(idmat);
    cout << "This is angle weight matrix" << endl;
    printMatrix(angle_weight);

    auto kronprod = kroneckerProduct(angle_weight, idmat);
    cout << "This is kronproduct matrix" << endl;
    printMatrix(kronprod);

    std::cout << "The list of angles before dot applied is : ";
    for (size_t i = 0; i < angles_len; ++i) {
        std::cout << angles[i] << " ";
    }

    std::cout << std::endl;

    double new_list_of_angles[angles_len];
    auto new_list_angles = matrixVectorMultiply(kronprod, angles);

    cout << "The list of angles after dot applied is : ";
    for (const auto& angle : new_list_angles) {
        cout << angle << " ";
    }
    cout << endl;

    return new_list_angles;
}

vector<double> calculate_angles(vector<complex<double>> statevector, int max_num_qubits){

    vector<complex<double>> remain_state = statevector;
    vector<double> thetas_vec;
    vector<double> phis_vec;
    vector<double> corrected;
    for (int i = 0; i < max_num_qubits-1; i++)
    {   std::cout << "--------------------------------------------------" << std::endl;
        cout << "Iteration number: " << i << endl;
        std::cout << "--------------------------------------------------" << std::endl;
        auto [new_remain_state, thetas_vec, phis_vec] = rotations(remain_state);

        std::cout << "Remaining vector: ";
        for (const auto &val : new_remain_state)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Thetas: ";
        for (const auto &val : thetas_vec)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Phis: ";
        for (const auto &val : phis_vec)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        remain_state = new_remain_state;
        
        corrected = correct_angles(thetas_vec);
    }

    return corrected;
}



QExpr multiplex(qbit &q, qbit &qs, vector<double> angles){
    
    return qexpr::_RY(q, -angles[0]) + qexpr::_CNOT(qs, q) + qexpr::_RY(q, -angles[1]) + qexpr::_CNOT(qs, q);
}

QExpr preparetozero(qlist::QList q){
    return qexpr::map(qexpr::_PrepZ, q);
}

int main(){
    iqsdk::IqsConfig iqs_config;
    iqsdk::FullStateSimulator iqs_device(iqs_config);
    iqsdk::QRT_ERROR_T status = iqs_device.ready();
    assert(status == iqsdk::QRT_ERROR_SUCCESS);

    constexpr int N = 2;
    qbit listable(qmem, N);

    vector<complex<double>> arbitrary_state = {0, 0.408248, 0.816497, 0.408248};

    vector<double> arbitrary_state_angles; 
    arbitrary_state_angles = calculate_angles(arbitrary_state, N);
    
    // vector<double> corrected;
    // corrected = correct_angles(arbitrary_state_angles);
    for (const auto& angle :arbitrary_state_angles) {
        cout << angle << " ";
    }
    cout << endl;

    qexpr::eval_hold(preparetozero(qmem));
    qexpr::eval_hold(multiplex(qmem[0], qmem[1], arbitrary_state_angles));

    std::vector<std::reference_wrapper<qbit>> qids;
	for (index_t qubit = 0; qubit < N; ++qubit)
	{
		qids.push_back(std::ref(qmem[qubit]));
	}

	std::cout << "------- " << N << " qubit states -------" << std::endl;
	std::vector<double> state_results = iqs_device.getProbabilities(qids);
    iqsdk::FullStateSimulator::displayProbabilities(state_results, qids);
    std::vector<std::complex<double>> amplitudes = iqs_device.getAmplitudes(qids);
    iqsdk::FullStateSimulator::displayAmplitudes(amplitudes, qids);
    return 0;

}