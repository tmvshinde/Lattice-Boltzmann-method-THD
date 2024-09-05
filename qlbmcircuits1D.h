#ifndef QLBMCIRCUITS1D_H
#define QLBMCIRCUITS1D_H

#include <clang/Quantum/quintrinsics.h>
#include <quantum.hpp>
#include <cmath>

#include "customgates.h"

static const double PI = std::acos(-1.0);
typedef size_t index_t;
extern qbit qmem[];

template <int num_qubits_work>
inline quantum_kernel void collision(const double velocity)
{

  const double angle1 = 0.92;
  const double angle2 = 1.15;

  const int q_max = num_qubits_work - 1; // last working qubit
  const int q_anc = num_qubits_work;
  const int q_target = 0;

  X(qmem[q_max]); // to get control 0

  H(qmem[q_anc]); // Hadamard on the ancilla qubit
  X(qmem[q_anc]); // control 0

  CCPHASE(q_max, q_anc, q_target, angle1); // defined 2 qubit controlled phase gate
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);
  CCPHASE(q_max, q_anc, q_target, angle1);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);

  X(qmem[q_max]);

  CCPHASE(q_max, q_anc, q_target, angle2);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);
  CCPHASE(q_max, q_anc, q_target, angle2);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);

  X(qmem[q_anc]);

  X(qmem[q_max]);

  CCPHASE(q_max, q_anc, q_target, -angle1);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);
  CCPHASE(q_max, q_anc, q_target, -angle1);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);

  X(qmem[q_max]);

  CCPHASE(q_max, q_anc, q_target, -angle2);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);
  CCPHASE(q_max, q_anc, q_target, -angle2);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[q_target]);

  H(qmem[q_anc]);
}

// the right shift - not generalized
template <int num_qubits>
inline quantum_kernel void shift_right()
{

  const int q_max = 5;

  X(qmem[q_max]); // control on 0 state

  // parameters - c0,c1,c2,c3,target,anc1,anc2,anc3
  MCX_5(0, 1, 2, 3, q_max, 4, 7, 8, 9, 10);
  MCX_4(0, 1, 2, q_max, 3, 7, 8, 9); 
  MCX_111(0, 1, q_max, 2, 7, 8);     //
  CNOT(qmem[q_max], qmem[0]);
  CNOT(qmem[q_max], qmem[1]);
  Toffoli(qmem[0], qmem[q_max], qmem[1]);

  X(qmem[q_max]);
}

// left shift - not generalized
template <int num_qubits>
inline quantum_kernel void shift_left()
{
  const int q_max = 5;

  CNOT(qmem[q_max], qmem[0]);
  Toffoli(qmem[0], qmem[q_max], qmem[1]);

  CNOT(qmem[q_max], qmem[2]);
  MCX_111(0, 1, q_max, 2, 7, 8);
  CNOT(qmem[q_max], qmem[2]);

  CNOT(qmem[q_max], qmem[3]);
  MCX_4(0, 1, 2, q_max, 3, 7, 8, 9);
  CNOT(qmem[q_max], qmem[3]);

  CNOT(qmem[q_max], qmem[4]);
  MCX_5(0, 1, 2, 3, q_max, 4, 7, 8, 9, 10);
  CNOT(qmem[q_max], qmem[4]);
}

// macros calculation
template <int num_qubits_work>
inline quantum_kernel void macros()
{

  const int q_max = num_qubits_work - 1;
  const int q_anc = num_qubits_work;

  SWAP(qmem[q_anc], qmem[q_max]);
  H(qmem[q_anc]);
}

template <int num_qubits_work>
inline quantum_kernel void optimized_collision(const double velocity)
{
  const double angle1 = acos(0.5 * (1 + velocity));
  const double angle2 = asin(0.5 * (1 - velocity));

  // Last working qubit, and ancilla qubit
  const int q_max = num_qubits_work - 1;
  const int q_anc = num_qubits_work;

  SWAP(qmem[q_anc], qmem[q_max]);
  X(qmem[q_anc]);
  RY(qmem[q_max], angle1);
  CNOT(qmem[q_anc], qmem[q_max]);
  RY(qmem[q_max], -angle1);
  CNOT(qmem[q_anc], qmem[q_max]);
  X(qmem[q_anc]);
  RY(qmem[q_max], angle2);
  CNOT(qmem[q_anc], qmem[q_max]);
  RY(qmem[q_max], -angle2);
  SWAP(qmem[q_anc], qmem[q_max]);
}

template <int num_qubits_work>
inline quantum_kernel void propagation_QFT()
{
  // Index of the last working qubit
  const int q_max = num_qubits_work - 1;

  // CNOT cascade
  int q_ctrl = q_max;
  int q_target = 0;
  for (index_t i = 0; i < q_max; ++i)
  {
    CNOT(qmem[q_ctrl], qmem[i]);
  }

  // Phase cascade
  for (index_t i = 0; i < q_max; ++i)
  {
    q_target = num_qubits_work - i - 2;
    H(qmem[q_target]);
    for (index_t k = 0; k < q_target; ++k)
    {
      q_ctrl = q_target - k - 1;
      double angle = 2.0 * PI / std::pow(2.0, (static_cast<double>(k) + 2.0));
      // Note: CPhase is inverted compared to the usual definition
      CPhase(qmem[q_ctrl], qmem[q_target], -angle);
    }
  }

  for (index_t i = 0; i < q_max; ++i)
  {
    double angle = 2.0 * PI / std::pow(2.0, (static_cast<double>(i) + 1.0));
    // Note that in Qiskit this would be P-gate. Global phase factor is omitted.
    RZ(qmem[i], angle);
  }

  // Reverse phase cascade
  for (index_t i = 0; i < q_max - 1; ++i)
  {
    q_target = i;
    H(qmem[q_target]);
    q_target = i + 1;
    for (index_t k = 0; k < q_target; ++k)
    {
      q_ctrl = k;
      double angle =
          -2.0 * PI / std::pow(2.0, (static_cast<double>(i - k) + 2.0));
      CPhase(qmem[q_ctrl], qmem[q_target], -angle);
    }
  }
  H(qmem[q_max - 1]);

  // Reverse CNOT cascade
  q_ctrl = q_max;
  for (index_t i = q_max; i > 0; --i)
  {
    CNOT(qmem[q_ctrl], qmem[i - 1]);
  }
}

template <int num_qubits_work>
inline quantum_kernel void parallel_shift_1d()
{

  const int q_max = num_qubits_work - 1;
  const int q_anc = num_qubits_work;

  const int q_anc1 = num_qubits_work + 1;
  const int q_anc2 = num_qubits_work + 2;
  const int q_anc3 = num_qubits_work + 3;

  CNOT(qmem[0], qmem[q_anc]);
  CNOT(qmem[q_max], qmem[q_anc]);
  CNOT(qmem[q_anc], qmem[1]);

  X(qmem[0]);

  Toffoli_10(q_anc, q_max, 1);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[2]);
  MCX_111(1, 2, q_anc, 3, q_anc1, q_anc2);
  Toffoli(qmem[q_anc], qmem[q_max], qmem[2]);
  Toffoli_10(q_anc, q_max, 1);

  MCX_100(q_anc, 1, q_max, 2, q_anc1, q_anc2);
  MCX_111(1, q_max, q_anc, 2, q_anc1, q_anc2);

  // if condition - variable part
  if (num_qubits_work > 5)
  {
    constexpr int num_qubits_work_variable = num_qubits_work - 6;
    constexpr int index_first_variable_work = 4;

    X(qmem[q_max]);
    for (index_t id = 0; id < index_first_variable_work; ++id)
    {
      CNOT(qmem[q_max], qmem[id]);
    }

    Toffoli(qmem[0], qmem[1], qmem[q_anc1]);
    Toffoli(qmem[2], qmem[q_anc1], qmem[q_anc2]);
    Toffoli(qmem[3], qmem[q_anc2], qmem[q_anc3]);

    CNOT(qmem[q_anc3], qmem[index_first_variable_work]);

    for (index_t i = index_first_variable_work; i < q_max - 1; ++i)
    {
      CNOT(qmem[q_max], qmem[i]);
      Toffoli(qmem[i], qmem[q_anc + i - 1], qmem[q_anc + i]);
      CNOT(qmem[q_anc + i], qmem[i + 1]);
    }

    for (index_t i = q_max - 2; i > 1; --i)
    {
      Toffoli(qmem[i], qmem[q_anc + i - 1], qmem[q_anc + i]);
    }

    Toffoli(qmem[0], qmem[1], qmem[q_anc1]);

    for (index_t i = 0; i < q_max - 1; ++i)
    {
      CNOT(qmem[q_max], qmem[i]);
    }
    X(qmem[q_max]);
  }

  CNOT(qmem[q_max], qmem[q_anc]);
  CNOT(qmem[0], qmem[q_anc]);

  X(qmem[q_anc]);
}
#endif
