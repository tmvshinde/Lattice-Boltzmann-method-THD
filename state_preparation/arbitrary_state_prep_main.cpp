
#include <clang/Quantum/quintrinsics.h>
#include <clang/Quantum/qexpr.h>
#include <clang/Quantum/qlist.h>
#include <clang/Quantum/datalist.h>

#include <qexpr_utils.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <cstring>
#include <string>
#include <fstream>
#include <quantum_full_state_simulator_backend.h>

#include <cmath>
#include "statepreparation.h"

typedef size_t index_t;
using namespace std;
const int max_num_qubits = 4;
qbit listable(qmem, max_num_qubits);

int main()
{

    iqsdk::IqsConfig iqs_config;
    iqsdk::FullStateSimulator iqs_device(iqs_config);
    iqsdk::QRT_ERROR_T status = iqs_device.ready();
    assert(status == iqsdk::QRT_ERROR_SUCCESS);

    
    qbit listable(qmem, max_num_qubits);
    vector<complex<double>> arbitrary_state = {0, 0.5, 1.0, 0.5};

    Arbitrarystateprep trail_state1(arbitrary_state, max_num_qubits, true);
    trail_state1.print_statevector();



    qexpr::eval_hold(trail_state1.prepare_qubits(qmem));

    // qexpr::eval_hold(qexpr::invert(trail_state1.bell_state(q)));
    //qexpr::eval_hold(trail_state1.bell_state(q));
    trail_state1.create_circuit(qmem);
    // vector<double> angles_are = {-2.03444};
    // datalist::DataList X = "Y";
    // qexpr::eval_hold(
    // trail_state1.state_prepare_gates(qmem, "RY", angles_are, true);
    // qexpr::eval_hold(trail_state1.RotationY(X, qmem, angles_are));

    std::vector<std::reference_wrapper<qbit>> qids;
    for (index_t qubit = 0; qubit < max_num_qubits; ++qubit)
    {
        qids.push_back(std::ref(qmem[qubit]));
    }
    std::cout << "------- " << max_num_qubits << " qubit states -------" << std::endl;
    std::vector<double> state_results = iqs_device.getProbabilities(qids);
    iqsdk::FullStateSimulator::displayProbabilities(state_results, qids);
    std::vector<std::complex<double>> amplitudes = iqs_device.getAmplitudes(qids);
    iqsdk::FullStateSimulator::displayAmplitudes(amplitudes, qids);
    return 0;
}
