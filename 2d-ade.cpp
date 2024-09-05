
/*
2D-ADE using Lattice Boltzmann Method
*/

// including the quantum header files
#include <clang/Quantum/quintrinsics.h>
#include <quantum.hpp>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <chrono>

#include "customgates.h"
#include "qlbmcircuits2D.h"
#include "encoding_x02.h"

const int max_num_qubits = 15;
qbit qmem[max_num_qubits];
cbit cmem[max_num_qubits];

typedef size_t index_t;
using namespace std;
using namespace std::chrono;

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

    cout << argc << endl;
    for (int i = 0; i < argc; ++i)
    {
        cout << argv[i] << "\n";
    }
    const int iter = stoi(argv[1]);
    const double x_vel = stod(argv[2]);
    const double y_vel = stod(argv[3]);

    /*
    solving 16 by 16 grid problem - 4 + 4 = 8 qubits for the grid and 2 Dist func
                                        + 3 for other 3 dist func .
        lattice | number_qubits_work |  num_qubits_lattice_side | Total qubits
        16*16   |       11           |           4              |      15
        32*32   |       13           |           5              |      18
        64*64   |       15           |           6              |      21
        128*128 |       17           |           7              |      24
        256*256 |       19           |           8              |      27
        512*512 |       21           |           9              |      30  
    */
    // for changing the size of lattice, change num_qubits_work and num_qubits_lattice_side
    constexpr int num_qubits_lattice_side = 4;
    constexpr int num_qubits_work = 11;
    constexpr int total_qubits = num_qubits_work + num_qubits_lattice_side;

    prepare_qubits<max_num_qubits>();

    /*
    if (iter == 0)
    {
        initial_encoding<num_qubits_work>();
    }
    */
    // else
    QLBM2D::encoding_x02<num_qubits_work>();
    //}

    optimize_collision<num_qubits_work>(x_vel, y_vel);
    // qft_propagation<num_qubits_work, num_qubits_lattice_side>();
    parallel_shift_2d<num_qubits_work, num_qubits_lattice_side>();
    macros<num_qubits_work>();

    // display and write probabilities to the txt file
    std::vector<std::reference_wrapper<qbit>> qids;
    for (index_t qubit = 0; qubit < total_qubits; ++qubit)
    {
        qids.push_back(std::ref(qmem[qubit]));
    }

    std::cout << "-----------" << total_qubits << " qubit states ----------" << std::endl;
    std::vector<double> state_results = iqs_device.getProbabilities(qids);
    // iqsdk::FullStateSimulator::displayProbabilities(state_results, qids);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(stop - start);

    cout << "Time taken by function: "
         << duration.count() << " seconds" << endl;

    std::ofstream outFile("qlbm-2d-states.txt");
    for (index_t k = 0; k < state_results.size(); ++k)
    {
        outFile << state_results[k] << "\n";
    }
    outFile.close();

    /*
        // sampling
        std::vector<std::vector<bool>> measurement_samples;
        unsigned total_samples = 10000;
        measurement_samples = iqs_device.getSamples(total_samples, qids);

        // build a distribution of states
        iqsdk::QssMap<unsigned int> distribution =
            iqsdk::FullStateSimulator::samplesToHistogram(measurement_samples);

        // print out the results
        std::cout << "Using " << total_samples
                  << " samples, the distribution of states is:" << std::endl;
        for (const auto &entry : distribution)
        {
            double weight = entry.second;
            weight = weight / total_samples;

            std::cout << entry.first << " : " << weight << std::endl;
        }
    */
    return 0;
}
