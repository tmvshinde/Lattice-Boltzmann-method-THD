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

vector<int> buildSequence(int level) {
    if(level == 0) {
        // Base case: for N=5
        return {4};
    }
    
    vector<int> prev = buildSequence(level - 1);
    
    
    vector<int> secondHalf = prev;
    
    secondHalf.back()++;
    
    
    vector<int> result = prev;
    result.insert(result.end(), secondHalf.begin(), secondHalf.end());
    
    return result;
}


vector<int> generatePattern(int N) {
    if (N < 5)
        throw invalid_argument("N must be at least 5.");
    
    int level = N - 5;  
    vector<int> auxSequence = buildSequence(level);
    
    vector<int> pattern;
    for (size_t i = 0; i < auxSequence.size(); i++) {
        pattern.push_back(3);
        pattern.push_back(auxSequence[i]);
    }
    return pattern;
}

vector<int> returnPattern(int N) {
    vector<int> pattern = generatePattern(N);
    vector<int> newpattern = {};
        for (size_t i = 0; i < pattern.size(); i++) {
            newpattern.push_back(pattern[i]);
        }
        pattern.insert(pattern.end(), newpattern.begin(), newpattern.end());
    return pattern;
}

float norm(vector<complex<double>> mat)
{
    double norm = 0.0;
    for (const auto &elem : mat)
    {
        norm += std::norm(elem); 
    }
    return std::sqrt(norm);
}

vector<complex<double>> normalize_statevector(vector<complex<double>> statevector)
{
    double vec_norm = norm(statevector);
    if (vec_norm == 0)
    {
        throw runtime_error("Cannot normalize a zero vector");
    }
    for (size_t i = 0; i < statevector.size(); i++)
    {
        statevector[i] = statevector[i] / vec_norm;
    }
    return statevector;
}

std::vector<std::vector<double>> createIdentityMatrix(int size)
{
    std::vector<std::vector<double>> identityMatrix(size, std::vector<double>(size, 0.0));

    for (int i = 0; i < size; ++i)
    {
        identityMatrix[i][i] = 1.0;
    }

    return identityMatrix;
}

void printMatrix(const std::vector<std::vector<double>> &matrix)
{
    for (const auto &row : matrix)
    {
        for (double val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void print_statevector(vector<complex<double>> statevector)
{
    for (const auto &arbitrary : statevector)
    {
        cout << arbitrary << endl;
    }
}

void print_angles_vec(vector<double> angles)
{
    for (const auto &val : angles)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void print_angles_vec1(vector<int> angles)
{
    for (const auto &val : angles)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void print2DVector(vector<vector<double>> &vec)
{
    for (const auto &row : vec)
    {
        for (const auto &elem : row)
        {
            std::cout << elem << " ";
        }
        std::cout << std::endl; // Move to the next line after each row
    }
}

std::vector<std::vector<double>> kroneckerProduct(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B)
{
    int aRows = A.size();
    int aCols = A[0].size();
    int bRows = B.size();
    int bCols = B[0].size();

    int resultRows = aRows * bRows;
    int resultCols = aCols * bCols;

    std::vector<std::vector<double>> result(resultRows, std::vector<double>(resultCols, 0.0));

    for (int i = 0; i < aRows; ++i)
    {
        for (int j = 0; j < aCols; ++j)
        {
            for (int k = 0; k < bRows; ++k)
            {
                for (int l = 0; l < bCols; ++l)
                {
                    result[i * bRows + k][j * bCols + l] = A[i][j] * B[k][l];
                }
            }
        }
    }

    return result;
}

std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>> &matrix,
                                         const std::vector<double> &vector)
{
    int m = matrix.size();     // Number of rows in matrix
    int n = matrix[0].size();  // Number of columns in matrix
    int vSize = vector.size(); // Size of the vector

    std::vector<double> result(m, 0);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
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

        thetas.push_back(-add_theta);
        phis.push_back(-add_phi);
    }

    return {remaining_vector, thetas, phis};
}


std::vector<std::vector<double>> mergeRows(const std::vector<std::vector<double>>& vec, int q) {
    int nrows = pow(2, q - 1) - 1; 
    if (vec.size() != nrows) {
        std::cerr << "Error: Input vector does not match expected row count: " << nrows << "\n";
        return {};
    }

    std::vector<std::vector<double>> newVec;
    int start = 0;
    int groupSize = pow(2, q - 2); 

    while (groupSize >= 1) {
        std::vector<double> mergedRow;
        
        for (int i = 0; i < groupSize && (start + i) < vec.size(); ++i) {
            mergedRow.insert(mergedRow.end(), vec[start + i].begin(), vec[start + i].end());
        }

        newVec.push_back(mergedRow);
        start += groupSize;
        groupSize /= 2; 
    }

    return newVec;
}


std::vector<double> correct_angles(vector<double> angles)
{

    int angles_len = angles.size();
    cout << "the length of angles : " << angles_len << endl;
    int local_qubits = log2(angles_len) + 1;
    cout << "The number of local qubits is : " << local_qubits << endl;

    std::vector<std::vector<double>> angle_weight = {
        {0.5, 0.5},
        {0.5, -0.5}};

    auto idmat = createIdentityMatrix(pow(2, local_qubits - 2));

    auto kronprod = kroneckerProduct(angle_weight, idmat);

    std::cout << "The list of angles before dot applied is : ";
    print_angles_vec(angles);
    
    std::cout << std::endl;

    double new_list_of_angles[angles_len];
    auto new_list_angles = matrixVectorMultiply(kronprod, angles);

    cout << "The list of angles after dot applied is : ";
    print_angles_vec(new_list_angles);
    
    return new_list_angles;
}

void processRow(const std::vector<double> &row, std::vector<std::vector<double>> &result)
{
    if (row.size() <= 2)
    {
        vector<double> row1 = correct_angles(row);
        result.push_back(row1);
    }
    else
    {
        vector<double> row1 = correct_angles(row);

        size_t mid = row1.size() / 2;
        std::vector<double> firstHalf(row1.begin(), row1.begin() + mid);
        std::vector<double> secondHalf(row1.begin() + mid, row1.end());

        cout << " First half " << endl;
        processRow(firstHalf, result);

        cout << " second half " << endl;
        processRow(secondHalf, result);
    }
}

std::tuple<vector<complex<double>>, vector<vector<double>>>
calculate_angles(vector<complex<double>> statevector, int max_num_qubits)
{

    vector<complex<double>> remain_state = statevector;
    vector<vector<double>> thetas_vec_full;
    vector<double> thetas_vec;
    vector<double> phis_vec;
    vector<double> corrected;
    vector<complex<double>> new_remain_state;
    vector<vector<double>> corrected_rows;
    vector<vector<double>> result;
    for (int i = 0; i < max_num_qubits - 1; i++)
    {
        std::cout << "--------------------------------------------------" << std::endl;
        cout << "Iteration number: " << i << endl;
        std::cout << "--------------------------------------------------" << std::endl;
        auto [new_remain_state, thetas_vec, phis_vec] = rotations(remain_state);

        std::cout << "Remaining vector: ";
        print_statevector(new_remain_state);
        std::cout << std::endl;

        std::cout << "Thetas: ";
        print_angles_vec(thetas_vec);

        std::cout << "Phis: ";
        print_angles_vec(phis_vec);

        remain_state = new_remain_state;

        processRow(thetas_vec, result);   
    }
    vector<vector<double>> new_result = mergeRows(result, max_num_qubits);
    
    return {remain_state, new_result};
}


QExpr multiplex(qbit &q, qbit &qs, vector<double> angles)
{

    return qexpr::_RY(q, -angles[0]) + qexpr::_CNOT(qs, q) + qexpr::_RY(q, -angles[1]) + qexpr::_CNOT(qs, q);
}

QExpr preparetozero(qlist::QList q)
{
    return qexpr::map(qexpr::_PrepZ, q);
}

QExpr only_one(qbit &q, double angle){
    return qexpr::_RY(q, -angle);
}

QExpr decom_two(qbit &qcontrol, qbit &qtarget, vector<double> angles){
    return only_one(qtarget, angles[0]) + qexpr::_CNOT(qcontrol, qtarget) + only_one(qtarget, angles[1]) + qexpr::_CNOT(qcontrol, qtarget) ;
}

QExpr decom_four(qlist::QList q, vector<double> angles){
    return decom_two(q[1], q[0], {angles[0], angles[1]}) + qexpr::_CNOT(q[2], q[0]) + decom_two(q[1], q[0], {angles[2], angles[3]}) + qexpr::_CNOT(q[2], q[0]);
}

int main()
{
    iqsdk::IqsConfig iqs_config;
    iqsdk::FullStateSimulator iqs_device(iqs_config);
    iqsdk::QRT_ERROR_T status = iqs_device.ready();
    assert(status == iqsdk::QRT_ERROR_SUCCESS);

    constexpr int N = 5;
    qbit listable(qmem, N);
    qexpr::eval_hold(preparetozero(qmem));

    vector<complex<double>> arbitrary_state = {0.25, 0.5, 1, 0.5, 0.25, 0.125, 0.065, 0,0.25, 0.5, 1, 0.5, 0.25, 0, 0, 0,0.25, 0.5, 1, 0.5, 0.25, 0, 0, 0,0.25, 0.5, 1, 0.5, 0.25, 0, 0, 0 };
    vector<complex<double>> normalized_state;
    normalized_state = normalize_statevector(arbitrary_state);
    print_statevector(normalized_state);
    auto [remain_state, arbitrary_state_angles] = calculate_angles(normalized_state, N);
    auto [new_remain_state, thetas_vec, phis_vec] = rotations(remain_state);
    
    arbitrary_state_angles.push_back(thetas_vec);
    print2DVector(arbitrary_state_angles);    
    
    for (size_t i = N; i -- >0;) {
        cout << "i" <<  i << endl;
                
        qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 1, only_one(qmem[N-1], arbitrary_state_angles[i][0])));
        qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 2,decom_two(qmem[N-1], qmem[N-2], arbitrary_state_angles[i])));
              
        qlist::QList last = qmem(N-3,N); 
        qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 4, decom_four(last, arbitrary_state_angles[i])));

        if (N > 3) {         
        qlist::QList last1 = qmem(N-4,N); 
        for(int j = 0; j < 2; j++){
            vector<double> new_angles{arbitrary_state_angles[i].begin() + j*4, arbitrary_state_angles[i].begin() + (j+1)*4};
            print_angles_vec(new_angles);
            qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 8, decom_four(last1, new_angles)));
            qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 8, qexpr::_CNOT(qmem[N-1], qmem[N-4])));
            }
        }
        if (N > 4){
        qlist::QList last1 = qmem(N-5,N);
        vector<int> pattern = returnPattern(N); 
        print_angles_vec1(pattern);
        for(int j = 0; j < 4; j++){
            vector<double> new_angles1{arbitrary_state_angles[i].begin() + j*4, arbitrary_state_angles[i].begin() + (j+1)*4};
            print_angles_vec(new_angles1);
            qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 16, decom_four(last1, new_angles1)));
            qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 16, qexpr::cIfTrue(j % 2 == 0, qexpr::_CNOT(qmem[3], qmem[0]))));
            qexpr::eval_hold(qexpr::cIfTrue(arbitrary_state_angles[i].size() == 16, qexpr::cIfTrue(j % 2 == 1, qexpr::_CNOT(qmem[4], qmem[0]))));                           
            }
        }
        
    }

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