#ifndef QLBMCIRCUITS2D_H
#define QLBMCIRCUITS2D_H

#include <clang/Quantum/quintrinsics.h>
#include <quantum.hpp>
#include <cmath>

static const double PI = std::acos(-1.0);
typedef size_t index_t;
extern qbit qmem[];

template <int num_qubits_work>
inline quantum_kernel void initial_encoding()
{
    const int q_anc = num_qubits_work;
    const int q_ctrl0 = num_qubits_work - 3;
    const int q_ctrl1 = num_qubits_work - 2;
    const int q_ctrl2 = num_qubits_work - 1; // last working qubit - last distribution

    X(qmem[0]);
    X(qmem[1]);
    X(qmem[4]);
    X(qmem[5]);
    // X(qmem[6]);

    // Hadamards to duplicate the state to distribution functions
    H(qmem[q_ctrl0]);
    H(qmem[q_ctrl1]);
    H(qmem[q_ctrl2]);
}

template <int num_qubits_work>
inline quantum_kernel void macros()
{
    const int q_anc = num_qubits_work;
    const int q_ctrl0 = num_qubits_work - 3;
    const int q_ctrl1 = num_qubits_work - 2;
    const int q_ctrl2 = num_qubits_work - 1; // last working qubit - last distribution

    const int q_anc1 = num_qubits_work + 1;
    const int q_anc2 = num_qubits_work + 2;

    SWAP(qmem[q_ctrl0], qmem[q_anc]);
    MCX_111(q_ctrl1, q_ctrl2, q_anc, q_ctrl0, q_anc1, q_anc2);
    H(qmem[q_anc]);

    SWAP(qmem[q_ctrl1], qmem[q_anc]);
    MCX_110(q_anc, q_ctrl2, q_ctrl1, q_ctrl0, q_anc1, q_anc2);
    H(qmem[q_anc]);

    SWAP(qmem[q_ctrl2], qmem[q_anc]);
    H(qmem[q_anc]);
}

template <int num_qubits_work>
inline quantum_kernel void optimize_collision(const double x_vel, const double y_vel)
{   
    double u_u = x_vel * x_vel + y_vel * y_vel ;

    // const double angle1 = 2 * acos(0.1667 * (1 + 3 * x_vel + 4.5 * x_vel * x_vel - 1.5 * u_u));
    // const double angle2 = 2 * asin(0.1667 * (1 - 3 * x_vel + 4.5 * x_vel * x_vel - 1.5 * u_u));
    // const double angle3 = 2 * acos(0.1667 * (1 + 3 * y_vel + 4.5 * y_vel * y_vel - 1.5 * u_u));
    // const double angle4 = 2 * asin(0.1667 * (1 - 3 * y_vel + 4.5 * y_vel * y_vel - 1.5 * u_u));
    // const double angle0 = 2 * acos(0.3332 * (1 - 1.5 * u_u));

    const double angle1 = 2 * acos(0.1667 * (1 + 3 * x_vel));
    const double angle2 = 2 * asin(0.1667 * (1 - 3 * x_vel));
    const double angle3 = 2 * acos(0.1667 * (1 + 3 * y_vel));
    const double angle4 = 2 * asin(0.1667 * (1 - 3 * y_vel));
    const double angle0 = 2 * acos(0.3332);

    const int q_anc = num_qubits_work;
    const int q_ctrl0 = num_qubits_work - 3;
    const int q_ctrl1 = num_qubits_work - 2;
    const int q_ctrl2 = num_qubits_work - 1; // last working qubit - last distribution
    const int ancilla1 = num_qubits_work + 1;
    const int ancilla2 = num_qubits_work + 2;

    CNOT(qmem[q_ctrl0], qmem[q_anc]);

    X(qmem[q_ctrl1]);
    X(qmem[q_ctrl2]);
    X(qmem[q_anc]);

    C3_RY(q_ctrl1, q_ctrl2, q_anc, q_ctrl0, angle1, ancilla1, ancilla2);

    X(qmem[q_anc]);

    C3_RY(q_ctrl1, q_ctrl2, q_anc, q_ctrl0, -angle2, ancilla1, ancilla2);

    X(qmem[q_ctrl1]);
    X(qmem[q_anc]);

    C3_RY(q_ctrl1, q_ctrl2, q_anc, q_ctrl0, angle3, ancilla1, ancilla2);

    X(qmem[q_anc]);

    C3_RY(q_ctrl1, q_ctrl2, q_anc, q_ctrl0, -angle4, ancilla1, ancilla2);

    X(qmem[q_ctrl1]);
    X(qmem[q_ctrl2]);
    X(qmem[q_anc]);

    C3_RY(q_ctrl1, q_ctrl2, q_anc, q_ctrl0, angle0, ancilla1, ancilla2);

    X(qmem[q_ctrl1]);
    X(qmem[q_anc]);

    SWAP(qmem[q_ctrl0], qmem[q_anc]);
}

template <int num_qubits_work, int num_qubits_lattice_side>
inline quantum_kernel void qft_propagation()
{
    const int q_anc = num_qubits_work;
    const int q_ctrl0 = num_qubits_work - 3;
    const int q_ctrl1 = num_qubits_work - 2;
    const int q_ctrl2 = num_qubits_work - 1; // last working qubit - last distribution

    const int lattice_end = num_qubits_lattice_side;

    // Controlled swaps
    for (index_t i = 0; i < lattice_end; ++i)
    {
        C_SWAP(q_ctrl1, i, lattice_end + i);
    }

    X(qmem[q_ctrl2]);
    CNOT(qmem[q_ctrl2], qmem[q_anc]);
    X(qmem[q_ctrl2]);

    // Toffoli
    for (index_t i = 0; i < lattice_end; ++i)
    {
        Toffoli(qmem[q_anc], qmem[q_ctrl0], qmem[i]);
    }

    // Phase cascade
    for (index_t j = 0; j < lattice_end; ++j)
    {
        const int q_targ = lattice_end - j - 1;
        CH(q_anc, q_targ);

        for (index_t m = 0; m < q_targ; ++m)
        {
            double angle = 2.0 * PI / std::pow(2.0, (static_cast<double>(m) + 2.0));
            CCPHASE(q_targ, q_anc, lattice_end - j - 2 - m, angle);
        }
    }

    for (index_t i = 0; i < lattice_end; ++i)
    {
        double angle = 2.0 * PI / std::pow(2.0, (static_cast<double>(i) + 1.0));
        CPhase(qmem[q_anc], qmem[i], angle);
    }

    // Reverse phase
    for (index_t i = 0; i < lattice_end - 1; ++i)
    {
        int q_targ = i;
        CH(q_anc, q_targ);
        q_targ = i + 1;
        for (index_t m = 0; m < q_targ; ++m)
        {
            double angle = 2.0 * PI / std::pow(2.0, (static_cast<double>(i - m) + 2.0));
            CCPHASE(m, q_anc, q_targ, -angle);
        }
    }

    CH(q_anc, lattice_end - 1);

    // Reverse cnot
    for (index_t i = lattice_end; i > 0; --i)
    {
        Toffoli(qmem[q_anc], qmem[q_ctrl0], qmem[i - 1]);
    }

    X(qmem[q_ctrl2]);
    CNOT(qmem[q_ctrl2], qmem[q_anc]);
    X(qmem[q_ctrl2]);

    for (index_t i = 0; i < lattice_end; ++i)
    {
        const int swap1 = lattice_end - 1 - i;
        C_SWAP(q_ctrl1, swap1, swap1 + lattice_end);
    }
}

template <int num_qubits_work, int num_qubits_lattice_side>
inline quantum_kernel void parallel_shift_2d()
{
    const int q_anc = num_qubits_work;
    const int q_ctrl0 = num_qubits_work - 3;
    const int q_ctrl1 = num_qubits_work - 2;
    const int q_ctrl2 = num_qubits_work - 1; // last working qubit - last distribution
    const int ancilla1 = num_qubits_work + 1;
    const int ancilla2 = num_qubits_work + 2;

    const int lattice_end = num_qubits_lattice_side;

    // Controlled swaps
    for (index_t i = 0; i < lattice_end; ++i)
    {
        C_SWAP(q_ctrl1, i, lattice_end + i);
    }

    Toffoli_10(q_ctrl2, q_anc, q_ctrl0);
    Toffoli_10(0, q_ctrl2, q_anc);
    Toffoli_10(q_ctrl0, q_ctrl2, q_anc);

    CNOT(qmem[q_anc], qmem[1]);
    CNOT_0(q_ctrl2, 0);

    Toffoli_10(q_anc, q_ctrl0, 1);
    Toffoli(qmem[q_anc], qmem[q_ctrl0], qmem[2]);
    MCX_111(1, 2, q_anc, 3, ancilla1, ancilla2);
    Toffoli(qmem[q_anc], qmem[q_ctrl0], qmem[2]);
    Toffoli_10(q_anc, q_ctrl0, 1);

    MCX_100(q_anc, q_ctrl0, 1, 2, ancilla1, ancilla2);

    MCX_111(q_anc, q_ctrl0, 1, 2, ancilla1, ancilla2);

    if (num_qubits_work > 11)
    {
        constexpr int index_first_variable_work = 4;

        X(qmem[q_ctrl0]);
        for (index_t i = 0; i < index_first_variable_work; ++i)
        {
            CNOT(qmem[q_ctrl0], qmem[i]);
        }

        Toffoli(qmem[1], qmem[q_anc], qmem[ancilla1]);
        Toffoli(qmem[2], qmem[ancilla1], qmem[ancilla2]);
        Toffoli(qmem[3], qmem[ancilla2], qmem[ancilla2 + 1]);
        CNOT(qmem[ancilla2 + 1], qmem[index_first_variable_work]);

        for (index_t i = index_first_variable_work; i < num_qubits_lattice_side - 1; ++i)
        {
            CNOT(qmem[q_ctrl0], qmem[i]);
            Toffoli(qmem[i], qmem[q_anc + i - 1], qmem[q_anc + i]);
            CNOT(qmem[q_anc + i], qmem[i + 1]);
        }

        for (index_t i = num_qubits_lattice_side - 2; i > 1; --i)
        {
            Toffoli(qmem[i], qmem[q_anc + i - 1], qmem[q_anc + i]);
        }

        Toffoli(qmem[1], qmem[q_anc], qmem[ancilla1]);

        for (index_t i = 0; i < num_qubits_lattice_side - 1; ++i)
        {
            CNOT(qmem[q_ctrl0], qmem[i]);
        }
        X(qmem[q_ctrl0]);
    }

    Toffoli_10(q_ctrl0, q_ctrl2, q_anc);
    Toffoli_10(0, q_ctrl2, q_anc);
    CNOT_0(q_ctrl2, q_anc);
    Toffoli_10(q_ctrl2, q_anc, q_ctrl0);

    for (index_t i = 0; i < lattice_end; ++i)
    {
        const int swap1 = lattice_end - 1 - i;
        C_SWAP(q_ctrl1, swap1, swap1 + lattice_end);
    }
}

template <int num_qubits_work, int total_qubits>
inline quantum_kernel void reinitialize_01()
{

    const int q_anc = num_qubits_work;
    const int q_ctrl0 = num_qubits_work - 3;
    const int q_ctrl1 = num_qubits_work - 2;
    const int q_ctrl2 = num_qubits_work - 1; // last working qubit - last distribution
    const int ancilla1 = num_qubits_work + 1;
    const int ancilla2 = num_qubits_work + 2;

    MCX_0000(q_ctrl0, q_ctrl1, q_ctrl2, q_anc, ancilla1, ancilla2, ancilla2 + 1, ancilla2 + 2);

    CH(ancilla1, q_ctrl0);
    CH(ancilla1, q_ctrl1);
    CH(ancilla1, q_ctrl2);
}

template <int num_qubits_work, int total_qubits>
inline quantum_kernel void reinitialize_02()
{

    const int q_anc = num_qubits_work;
    const int q_ctrl0 = num_qubits_work - 3;
    const int q_ctrl1 = num_qubits_work - 2;
    const int q_ctrl2 = num_qubits_work - 1; // last working qubit - last distribution
    const int ancilla1 = num_qubits_work + 1;
    const int ancilla2 = num_qubits_work + 2;

    MCX_00001(q_ctrl0, q_ctrl1, q_ctrl2, q_anc, ancilla1, ancilla2, ancilla2 + 1, ancilla2 + 2, ancilla2 + 3, ancilla2 + 4);

    CH(ancilla2, q_ctrl0);
    CH(ancilla2, q_ctrl1);
    CH(ancilla2, q_ctrl2);
}

#endif
