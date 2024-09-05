#include <clang/Quantum/quintrinsics.h>

#include <quantum.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

#include "encoding_x02.h"
#include "qlbmcircuits1D.h"
#include "customgates.h"

const int max_num_qubits = 15;
qbit qmem[max_num_qubits];
cbit cmem[max_num_qubits];

typedef size_t index_t;
using namespace std;
// state preparation
using namespace std::chrono;

template <int num_qubits, int num_qubits_work>
inline quantum_kernel void construct_general_circuit(double velocity)
{

	prepare_qubits<num_qubits>();
	QLBM1D::encoding_x02<num_qubits_work>();
	collision<num_qubits_work>(velocity);
	shift_right<num_qubits>();
	shift_left<num_qubits>();
	macros<num_qubits_work>();
}

template <int num_qubits, int num_qubits_work>
inline quantum_kernel void construct_qft(double velocity)
{

	prepare_qubits<num_qubits>();
	QLBM1D::encoding_x02<num_qubits_work>();
	optimized_collision<num_qubits_work>(velocity);
	propagation_QFT<num_qubits_work>();
	macros<num_qubits_work>();
}

template <int num_qubits, int num_qubits_work>
inline quantum_kernel void construct_parallel_shift(double velocity)
{

	prepare_qubits<num_qubits>();
	QLBM1D::encoding_x02<num_qubits_work>();
	// optimized_collision<num_qubits_work>(velocity);
	// parallel_shift_1d<num_qubits_work>();
	// macros<num_qubits_work>();
}

int main(int argc, char *argv[])
{
	auto start = high_resolution_clock::now();
	iqsdk::IqsConfig backend_config;
    iqsdk::FullStateSimulator iqs_device(backend_config);
    if (iqsdk::QRT_ERROR_SUCCESS != iqs_device.ready())
    {
        return 1;
    }
    iqs_device.printVerbose(false);

	//setprecision(5);
	constexpr double velocity = 0.2;
	constexpr int num_qubits_work = 5;						// 4 + 1 qubits - to encode both distribution functions
	constexpr int num_qubits_ancilla = num_qubits_work - 1; // ancilla qubits for the decomposition of multi controlled gates
	constexpr int num_qubits = num_qubits_work + num_qubits_ancilla;

	bool use_qft = false;
	// bool use_parallel_shift = false;

	if (use_qft)
	{
		construct_qft<num_qubits, num_qubits_work>(velocity);
	}
	else
	{
		construct_parallel_shift<num_qubits, num_qubits_work>(velocity);
	}

	// construct_general_circuit<num_qubits, num_qubits_work>(velocity);

	std::vector<std::reference_wrapper<qbit>> qids;
	for (index_t qubit = 0; qubit < num_qubits; ++qubit)
	{
		qids.push_back(std::ref(qmem[qubit]));
	}

	std::cout << "------- " << num_qubits << " qubit states -------" << std::endl;
	std::vector<double> state_results = iqs_device.getProbabilities(qids);
	// std::vector<std::complex<double>> amplitudes = iqs_device.getAmplitudes(qids);
	//  iqsdk::FullStateSimulator::displayProbabilities(state_results, qids);
	//  iqsdk::FullStateSimulator::displayAmplitudes(amplitudes, qids);

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(stop - start);

	cout << "Time taken by function: "
		 << duration.count() << " milliseconds" << endl;

	std::ofstream outFile("qlbm-states-timestep.txt"); // saving the state vector
	for (index_t k = 0; k < state_results.size(); ++k)
	{
		outFile //<< k << ", "
			<< state_results[k] << "\n";
	}
	outFile.close();

	return 0;
}
