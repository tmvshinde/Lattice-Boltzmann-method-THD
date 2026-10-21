"""Check that an optimised circuit implements the same operation as the original.

The optimised circuit must hold the original's qubits first, in the same order, followed by
any ancillas (``optimize`` always produces this layout).  Ancillas start in |0> and must be
returned to |0> for the check to pass.

    <= max_direct qubits   random-statevector check: a random state on all original qubits,
                           ancillas in |0>; fidelity |<orig psi| P_anc0 opt psi>|^2.
    <= max_mirror qubits   mirror check: brickwork random-state prep S, then
                           S ; optimised ; original^dagger ; S^dagger, reading one amplitude.
                           One statevector in memory; threshold 1e-3 (single precision above
                           28 qubits reads ~1 - 1e-5 on a correct circuit).

Circuits with measurements, resets, classical control or unbound parameters are not
unitaries and raise ``NotVerifiable``.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate


class NotVerifiable(Exception):
    """The pair of circuits cannot be compared by statevector simulation."""


def _check_unitary(qc, label):
    if qc.parameters:
        raise NotVerifiable(f"{label} circuit has unbound parameters")
    for inst in qc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        if inst.clbits or not isinstance(op, Gate):
            raise NotVerifiable(f"{label} circuit contains non-unitary operation '{op.name}'")


def _flat(qc):
    return transpile(qc, basis_gates=["cx", "u"], optimization_level=0,
                     qubits_initially_zero=False)


def _simulator(precision="double"):
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        return None
    return AerSimulator(method="statevector", precision=precision)


def _run_state(qc, vec, sim):
    if sim is None:
        from qiskit.quantum_info import Statevector
        return Statevector(vec).evolve(_flat(qc)).data
    c = QuantumCircuit(qc.num_qubits)
    c.set_statevector(vec)
    c.compose(_flat(qc), inplace=True)
    c.save_statevector()
    return np.asarray(sim.run(c).result().get_statevector())


def _brickwork(n, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for layer in range(n):
        for q in range(n):
            qc.u(*rng.uniform(0, 2 * np.pi, 3), q)
        for q in range(layer % 2, n - 1, 2):
            qc.cx(q, q + 1)
    return qc


def verify(original, optimized, *, seed=11, max_direct=26, max_mirror=28):
    """Return {'passed', 'fidelity', 'method', 'leak'}; raises NotVerifiable if impossible."""
    _check_unitary(original, "original")
    _check_unitary(optimized, "optimised")
    n, m = original.num_qubits, optimized.num_qubits
    if m < n:
        raise NotVerifiable("optimised circuit has fewer qubits than the original")

    sim = _simulator()
    if sim is None and m > 20:
        raise NotVerifiable(f"{m} qubits needs qiskit-aer (pip install qiskit-aer)")

    if m <= max_direct:
        from qiskit.quantum_info import random_statevector
        psi = random_statevector(2 ** n, seed=seed).data
        out_o = _run_state(original, psi, sim)
        full = np.zeros(2 ** m, dtype=complex)
        full[: 2 ** n] = psi
        sub = _run_state(optimized, full, sim)[: 2 ** n]
        fid = float(abs(np.vdot(out_o, sub)) ** 2)
        leak = float(max(0.0, 1.0 - np.vdot(sub, sub).real))
        return {"passed": fid > 1 - 1e-9, "fidelity": fid, "method": "statevector", "leak": leak}

    if m <= max_mirror and sim is not None:
        precision = "double" if m <= 28 else "single"
        s = _brickwork(n, seed)
        c = QuantumCircuit(m)
        c.compose(s, range(n), inplace=True)
        c.compose(_flat(optimized), range(m), inplace=True)
        c.compose(_flat(original).inverse(), range(n), inplace=True)
        c.compose(s.inverse(), range(n), inplace=True)
        c.save_amplitudes([0])
        amp = _simulator(precision).run(c).result().data(0)["amplitudes"][0]
        fid = float(abs(amp) ** 2)
        return {"passed": fid > 1 - 1e-3, "fidelity": fid, "method": f"mirror/{precision}",
                "leak": None}

    raise NotVerifiable(f"{m} qubits is above max_mirror={max_mirror}")
