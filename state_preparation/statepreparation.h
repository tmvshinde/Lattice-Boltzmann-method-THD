#ifndef STATEPREPARATION_H
#define STATEPREPARATION_H

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
#include "customgates.h"


typedef size_t index_t;
using namespace std;
#define _EPS 1.0e-10
// extern qbit qmem[];

class Arbitrarystateprep
{
private:
    vector<complex<double>> statevector;
    // list<double> thetas;
    // list<double> phis;
    const int num_qubits;
    bool normalize;

public:
    // vector<complex<double>> statevector1;
    // Arbitrarystateprep();
    // Arbitrarystateprep(vector<complex<double>> statevector);

    // ~Arbitrarystateprep();

    // vector<complex<double>> print_statevector();
    Arbitrarystateprep() : statevector(8), num_qubits(3), normalize(false)
    {

        cout << "The default constructor is called" << endl;
    }

    Arbitrarystateprep(vector<complex<double>> vec, int num_qubits, bool normalize) : statevector(vec), num_qubits(num_qubits), normalize(normalize)
    {
        cout << "The assignment constructor is called" << endl;
        if(normalize == true){
            normalize_state();
        }
    }

    Arbitrarystateprep(vector<complex<double>> vec, int num_qubits) : statevector(vec), num_qubits(num_qubits), normalize(false)
    {
        cout << "The default bool false constructor is called" << endl;
        double vec_norm = norm(statevector);
        if (vec_norm != 1.0){
            throw runtime_error("The sum of norm of the state is not 1, but is :" + to_string(vec_norm));
        }
        normalize_state();
    }

    
    void normalize_state(){
        
        double vec_norm = norm(statevector);
        // vector<complex<double>> normalized_vec(statevector.size());
        if (vec_norm == 0){
            throw runtime_error("Cannot normalize a zero vector");
        }
        for (size_t i=0;i< statevector.size(); i++)
        {
            statevector[i] = statevector[i] / vec_norm;

        }
    }

    void print_statevector()
    {
        cout << num_qubits << endl;
        for (const auto &arbitrary : statevector)
        {
            cout << arbitrary << endl;
        }
    }

    void create_circuit(qlist::QList qmem)
    {

        vector<complex<double>> remain_state = statevector;
        vector<double> thetas_vec;
        vector<double> phis_vec;
        for (int i = 0; i < num_qubits; i++)
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
            

            bool add_last_cnot = true;
            cout << "Norm phis :" << norm_angles(phis_vec) << endl;
            cout << "Norm thetas :" << norm_angles(thetas_vec) << endl;

            if (norm_angles(phis_vec) != 0 and norm_angles(thetas_vec)!=0){
                add_last_cnot = false;
                cout << "last cnot is : " << add_last_cnot << endl;
            }

            // if(norm_angles(phis_vec) != 0){
            //     state_prepare_gates(qmem, "RZ", phis_vec, add_last_cnot);
            // }
            
            if(norm_angles(thetas_vec) != 0){
                cout << "values in thetas vec : " << thetas_vec.size() << endl;
                state_prepare_gates(qmem, "RY", thetas_vec, add_last_cnot);

            }

            std::cout << "--------------------------------------------------" << std::endl;
        }
    }

    QExpr state_prepare_gates(qlist::QList qmem, string gate, vector<double> angles, bool last_cnot=true){

        
        int angles_len = angles.size();
        cout << "the length of angles : " << angles_len << endl;
        int local_qubits = log2(angles_len) + 1;
        cout << "The number of local qubits is : " << local_qubits << endl;

        const int lsb = 0;
        int msb = local_qubits - 1;
        datalist::DataList Y = "Y";
        datalist::DataList Z = "Z";
        datalist::DataList cx = "CNOT";


        if (local_qubits == 1){
            // RY(qmem[0], angles[0]);
            cout << "Local qubits : 1, applied final ry" << angles[0] << endl;
            cout << gate << endl;
            return singleQubitRotation(Y, qmem[1], angles[0]);
            
        }
        
        
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

        std::vector<double> firstHalf_new_list_angles(new_list_angles.begin(), new_list_angles.begin() + angles_len/2);

        state_prepare_gates(qmem, "RY", firstHalf_new_list_angles, last_cnot);

        // cout << gate << endl;
        // if(gate == "RY"){
            
        //     cout << "Apply RY" << new_list_angles[0] << endl;
        //     X_expr(qmem[0]);
        //     qexpr::eval_hold(singleQubitRotation(Y, 0, new_list_angles[0]));
        //     qexpr::printQuantumLogic(singleQubitRotation(Y, qmem[0], new_list_angles[0]));
        //     if (last_cnot == true){
        //         cout << " cx applied : " << lsb << " : " << msb<< endl;
        //         qexpr::eval_hold(CNOT_expr(cx, lsb, msb));
        //     }
        //     cout << "Applied RY" << new_list_angles[1] << endl;
        //     qexpr::eval_hold(singleQubitRotation(Y, qmem[0], new_list_angles[1]));
        // }
        
        // if(gate == "RZ"){
        //     qexpr::eval_hold(RotationZ(qmem, new_list_of_angles));
        // }

        
    }

    // void calc_angles(int angles_len, double* angles){
    //     std::vector<std::vector<double>> angle_weight = {
    //         {0.5, 0.5},
    //         {0.5, -0.5}
    //     };

    //     std::cout << "The list of angles before dot applied is : ";
    //     for (size_t i = 0; i < angles_len; ++i) {
    //         std::cout << angles[i] << " ";
    //     }
    //     std::cout << std::endl;


    //     double new_list_of_angles[angles_len];
    //     for (size_t i = 0; i < angles_len; ++i) {
    //         new_list_of_angles[i] = 0.0;
    //         for (size_t j = 0; j < angle_weight[i % 2].size(); ++j) {
    //             setprecision(5);
    //             new_list_of_angles[i] += angle_weight[i % 2][j] * angles[j + (i / 2) * 2];
    //             if (new_list_of_angles[i] < _EPS ){
    //                 new_list_of_angles[i] = 0;
    //             }
    //         }
    //     }

    //     cout << "The list of angles after dot applied is : ";
    //     for (const auto& angle : new_list_of_angles) {
    //         cout << angle << " ";
    //     }
    //     cout << endl;
    // }

    

    QExpr singleQubitRotation(datalist::DataList p, qbit q, double theta) {
        // cout << " This is rotation expression" << endl;
        return  qexpr::cIf(p == datalist::DataList("X"), qexpr::_RX(q,theta),
                qexpr::cIf(p == datalist::DataList("Y"), qexpr::_RY(q,theta),
                qexpr::cIf(p == datalist::DataList("Z"), qexpr::_RZ(q,theta),
                qexpr::exitAtCompile(
                    "singleQubitRotation: expected X, Y, or Z; got "
                    + p
                ))));
    }
    
    QExpr CNOT_expr(datalist::DataList p, qbit q, qbit t) {
        // cout << " This is CNOT expression" << endl;
        return  qexpr::cIfTrue(p == datalist::DataList("CNOT"), qexpr::_CNOT(q,t));
    }

    QExpr X_expr(qbit q) {
        // cout << " This is CNOT expression" << endl;
        return  qexpr::_X(q);
    }

    QExpr RotationY(datalist::DataList p, qlist::QList q, double* angle){

        return  qexpr::cIf(q.size() > 0,
                          singleQubitRotation(p, q[0], angle[0]) + RotationY(p, q +1, angle + 1),
                          qexpr::identity()
        );  
    }

    QExpr RotationZ(qlist::QList q, double* angle){
        return qexpr::cIf(sizeof(angle) > 0,
                          qexpr::_RZ(q[0], angle[0]) + RotationZ(q +1, angle + 1),
                          qexpr::exitAtCompile("singleQubitRotation: expected X, Y, or Z; got ")
        );
    }

    float norm(vector<complex<double>> mat)
    {
    
        // To store the sum of squares of the
        // elements of the given matrix
        double norm = 0.0;
        for (const auto& elem : mat) {
            norm += std::norm(elem); // std::norm returns the square of the magnitude
        }
        return std::sqrt(norm);
    }

    float norm_angles(vector<double> mat)
    {
    
        // To store the sum of squares of the
        // elements of the given matrix
        double norm = 0.0;
        for (const auto& elem : mat) {
            norm += std::norm(elem); // std::norm returns the square of the magnitude
        }
        return std::sqrt(norm);
    }


    double norm_angles(const double* arr) {
        double sum = 0;
        size_t len = sizeof(arr);
        for (size_t i = 0; i < len; ++i) {
            sum += arr[i] * arr[i];
        }
        return sqrt(sum);
    }

    std::tuple<complex<double>, double, double>
    angles(const pair<complex<double>, complex<double>> &pair_of_complex)
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

    std::tuple<vector<complex<double>>, vector<double>, vector<double>>
    rotations(const vector<complex<double>> &local_param)
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

    // static std::tuple<vector<complex<double>>, vector<double>, vector<double>>
    // rotations(const vector<complex<double>> &local_param)
    // {
    //     vector<complex<double>> remaining_vector;
    //     vector<double> thetas;
    //     vector<double> phis;

    //     size_t param_len = local_param.size();
    //     cout << "Length of the parameters passed: " << param_len << endl;

    //     for (size_t i = 0; i < param_len / 2; ++i)
    //     {
    //         cout << "Iteration in calc block angles: " << i << endl;
    //         cout << "Indexes i and i + 1 for loop: " << 2 * i << " , " << 2 * (i + 1) << endl;

    //         auto [remains, add_theta, add_phi] = angles({local_param[2 * i], local_param[2 * i + 1]});

    //         cout << "The theta to be added is: " << add_theta << endl;
    //         cout << "The phi to be added is: " << add_phi << endl;
    //         cout << "The remaining parameters are: " << remains << endl;

    //         remaining_vector.push_back(remains);

    //         // Rotations for all imaginary qubits of the full vector
    //         thetas.push_back(-add_theta);
    //         phis.push_back(-add_phi);
    //     }

    //     return {remaining_vector, thetas, phis};
    // }

    // static std::tuple<complex<double>, double, double> angles(const pair<complex<double>, complex<double>> &pair_of_complex)
    // {
    //     complex<double> a_complex = pair_of_complex.first;
    //     complex<double> b_complex = pair_of_complex.second;

    //     cout << "2 values passed to calculate angles: " << a_complex << " " << b_complex << endl;

    //     double mag_a = std::abs(a_complex);
    //     double final_r = std::sqrt(mag_a * mag_a + std::abs(b_complex) * std::abs(b_complex));
    //     double theta, phi, final_t;

    //     if (final_r < _EPS)
    //     {
    //         theta = 0;
    //         phi = 0;
    //         final_r = 0;
    //         final_t = 0;
    //     }
    //     else
    //     {
    //         theta = 2 * std::acos(mag_a / final_r);
    //         double a_arg = std::arg(a_complex);
    //         double b_arg = std::arg(b_complex);
    //         final_t = a_arg + b_arg;
    //         phi = b_arg - a_arg;
    //     }

    //     complex<double> remains = final_r * exp(complex<double>(0, final_t / 2));
    //     return {remains, theta, phi};
    // }

    // calculate the angles theta and phi

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

    QExpr prepare_qubits(qlist::QList qmem)
    {
        return qexpr::cIf(qmem.size() > 0,
                          qexpr::_PrepZ(qmem[0]) + prepare_qubits(qmem + 1),
                          qexpr::identity());
    }

    QExpr bell_state(qlist::QList qmem)
    {
        return qexpr::_H(qmem[0]) + qexpr::_CNOT(qmem[0], qmem[1]);
    }

    ~Arbitrarystateprep() {}
};

#endif