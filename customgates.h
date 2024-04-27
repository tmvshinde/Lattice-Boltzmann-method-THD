#ifndef CUSTOMGATES_H
#define CUSTOMGATES_H

#include <clang/Quantum/quintrinsics.h>
#include <quantum.hpp>

extern qbit qmem[];
typedef size_t index_t;

template <int N>
inline quantum_kernel void prepare_qubits()
{
	for (int i = 0; i < N; ++i)
	{
		PrepZ(qmem[i]);
	}
}

inline quantum_kernel void CNOT_0(int control0_1, int target)
{
	X(qmem[control0_1]);
	CNOT(qmem[control0_1], qmem[target]);
	X(qmem[control0_1]);
}

inline quantum_kernel void Toffoli_10(const int control0, const int control0_1,
									  const int target)
{
	X(qmem[control0_1]);
	Toffoli(qmem[control0], qmem[control0_1], qmem[target]);
	X(qmem[control0_1]);
}

inline quantum_kernel void MCX_111(int control1, int control2,
								   int control3, int target, int ancilla1, int ancilla2)
{

	Toffoli(qmem[control1], qmem[control2], qmem[ancilla1]);
	Toffoli(qmem[control3], qmem[ancilla1], qmem[ancilla2]);
	CNOT(qmem[ancilla2], qmem[target]);
	Toffoli(qmem[control3], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control1], qmem[control2], qmem[ancilla1]);
}

inline quantum_kernel void MCX_110(const int control1, const int control2,
								   const int control0_3, const int target,
								   const int ancilla1, const int ancilla2)
{

	// Flip 0 controls
	X(qmem[control0_3]);

	// Decompose with Toffolis over ancilla
	Toffoli(qmem[control1], qmem[control2], qmem[ancilla1]);
	Toffoli(qmem[control0_3], qmem[ancilla1], qmem[ancilla2]);
	CNOT(qmem[ancilla2], qmem[target]);
	Toffoli(qmem[control0_3], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control1], qmem[control2], qmem[ancilla1]);

	// Flip 0 controls
	X(qmem[control0_3]);
}

inline quantum_kernel void MCX_100(const int control1, const int control0_2,
								   const int control0_3, const int target,
								   const int ancilla1, const int ancilla2)
{

	// Flip 0 controls
	X(qmem[control0_2]);
	X(qmem[control0_3]);

	// Decompose with Toffolis over ancilla
	Toffoli(qmem[control1], qmem[control0_2], qmem[ancilla1]);
	Toffoli(qmem[control0_3], qmem[ancilla1], qmem[ancilla2]);
	CNOT(qmem[ancilla2], qmem[target]);
	Toffoli(qmem[control0_3], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control1], qmem[control0_2], qmem[ancilla1]);

	// Flip 0 controls
	X(qmem[control0_3]);
	X(qmem[control0_2]);
}

// Controlled Rotation Y gate decomposition
inline quantum_kernel void CRY(int q_ctrl, int q_targ, double angle)
{
	RY(qmem[q_targ], angle);
	CNOT(qmem[q_ctrl], qmem[q_targ]);
	RY(qmem[q_targ], -angle);
}

// 3-qubit controlled rotation Y gate decomposition
inline quantum_kernel void C3_RY(int q_ctrl1, int q_ctrl2, int q_anc, int q_ctrl0, double angle, int ancilla1, int ancilla2)
{
	Toffoli(qmem[q_ctrl1], qmem[q_ctrl2], qmem[ancilla1]);
	Toffoli(qmem[q_anc], qmem[ancilla1], qmem[ancilla2]);

	RY(qmem[q_ctrl0], angle / 2);
	CNOT(qmem[ancilla2], qmem[q_ctrl0]);
	RY(qmem[q_ctrl0], -angle / 2);
	CNOT(qmem[ancilla2], qmem[q_ctrl0]);

	Toffoli(qmem[q_anc], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[q_ctrl1], qmem[q_ctrl2], qmem[ancilla1]);
}

inline quantum_kernel void C_SWAP(int q_ctrl, int q_swap1, int q_swap2)
{
	CNOT(qmem[q_swap2], qmem[q_swap1]);
	Toffoli(qmem[q_ctrl], qmem[q_swap1], qmem[q_swap2]);
	CNOT(qmem[q_swap2], qmem[q_swap1]);
}

inline quantum_kernel void CH(int q_ctrl, int q_targ)
{
	S(qmem[q_targ]);
	H(qmem[q_targ]);
	T(qmem[q_targ]);
	CNOT(qmem[q_ctrl], qmem[q_targ]);
	Tdag(qmem[q_targ]);
	H(qmem[q_targ]);
	Sdag(qmem[q_targ]);
}

inline quantum_kernel void CCPHASE(int q_m, int q_a, int q_t, double angle)
{
	// angle divided by 2, condition needs to be satisfied V^2 = X
	CPhase(qmem[q_m], qmem[q_t], angle / 2);
	CNOT(qmem[q_a], qmem[q_m]);
	CPhase(qmem[q_m], qmem[q_t], -angle / 2);
	CNOT(qmem[q_a], qmem[q_m]);
	CPhase(qmem[q_a], qmem[q_t], angle / 2);
}

/*
inline quantum_kernel void c4_phase()
{
	Toffoli(qmem[q_ctrl1], qmem[q_ctrl2], qmem[ancilla1]);
	Toffoli(qmem[q_anc], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[q_ctrl1], qmem[q_ctrl2], qmem[ancilla1]);
	CPhase(qmem[q_m], qmem[q_t], angle / 2); // angle divided by 2, condition needs to be satisfied V^2 = X
	CNOT(qmem[q_a], qmem[q_m]);
	CPhase(qmem[q_m], qmem[q_t], -angle / 2);
	CNOT(qmem[q_a], qmem[q_m]);
	CPhase(qmem[q_a], qmem[q_t], angle / 2);
	Toffoli(qmem[q_anc], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[q_ctrl1], qmem[q_ctrl2], qmem[ancilla1]);
	Toffoli(qmem[q_anc], qmem[ancilla1], qmem[ancilla2]);
}
*/

// decomposition of 4 qubit controlled CNOT
inline quantum_kernel void MCX_4(const int control0, const int control1, const int control2,
								 const int control3, const int target, const int ancilla1, const int ancilla2,
								 const int ancilla3)
{

	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	CNOT(qmem[ancilla3], qmem[target]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
}

inline quantum_kernel void MCX_0000(const int control0, const int control1, const int control2,
									const int control3, const int target, const int ancilla1, const int ancilla2,
									const int ancilla3)
{
	X(qmem[control0]);
	X(qmem[control1]);
	X(qmem[control2]);
	X(qmem[control3]);
	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	CNOT(qmem[ancilla3], qmem[target]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
	X(qmem[control0]);
	X(qmem[control1]);
	X(qmem[control2]);
	X(qmem[control3]);
}

inline quantum_kernel void MCX_5(const int control0, const int control1, const int control2,
								 const int control3, const int control4, const int target, const int ancilla1, const int ancilla2,
								 const int ancilla3, const int ancilla4)
{

	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	Toffoli(qmem[control4], qmem[ancilla3], qmem[ancilla4]);
	CNOT(qmem[ancilla4], qmem[target]);
	Toffoli(qmem[control4], qmem[ancilla3], qmem[ancilla4]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
}

inline quantum_kernel void MCX_00001(const int control0, const int control1, const int control2,
									 const int control3, const int control4, const int target, const int ancilla1, const int ancilla2,
									 const int ancilla3, const int ancilla4)
{
	X(qmem[control0]);
	X(qmem[control1]);
	X(qmem[control2]);
	X(qmem[control3]);
	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	Toffoli(qmem[control4], qmem[ancilla3], qmem[ancilla4]);
	CNOT(qmem[ancilla4], qmem[target]);
	Toffoli(qmem[control4], qmem[ancilla3], qmem[ancilla4]);
	Toffoli(qmem[control3], qmem[ancilla2], qmem[ancilla3]);
	Toffoli(qmem[control2], qmem[ancilla1], qmem[ancilla2]);
	Toffoli(qmem[control0], qmem[control1], qmem[ancilla1]);
	X(qmem[control0]);
	X(qmem[control1]);
	X(qmem[control2]);
	X(qmem[control3]);
}

inline quantum_kernel void multi_controlled_X(int num_qubits_work, int num_controls, int target)
{
	int q_ancilla1 = num_qubits_work + 1;
	// First 2 controls separated to loop through controls and ancilla qubit properly
	Toffoli(qmem[0], qmem[1], qmem[q_ancilla1]);

	// For loop for Remaining controls and ancilla
	for (index_t i = 0; i < num_controls - 2; i++)
	{

		int q_ancilla = q_ancilla1 + i;
		Toffoli(qmem[i + 2], qmem[q_ancilla], qmem[q_ancilla + 1]);

		if (i == num_controls - 3)
		{
			// ancilla to target qubit CNOT
			CNOT(qmem[q_ancilla + 1], qmem[target]);
		}
	}

	// Reversing to uncompute the ancillas
	for (int i = 0; i < num_controls - 2; ++i)
	{

		int q_ancillary = num_qubits_work + num_controls - i - 2;
		Toffoli(qmem[num_controls - i - 1], qmem[q_ancillary], qmem[q_ancillary + 1]);
	}

	Toffoli(qmem[0], qmem[1], qmem[q_ancilla1]);
}

#endif