from qiskit.circuit.random import random_circuit
from qiskit import transpile, QuantumCircuit
from ctc_optimizer import ancilla_tradeoff, optimize, verify
import numpy as np
import os
from qiskit.transpiler.passes.synthesis import HLSConfig
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit.circuit.library import CXGate, XGate, HGate

from qiskit.transpiler.passes import (
    InverseCancellation, CommutativeCancellation,
    Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure,
)

from qiskit_aer import AerSimulator

sim = AerSimulator(method="statevector")

path = "/home/tmvshinde/E-Folder/PHD/Codes/Resource estimates/ctc_optimizer/examples/test_circuits/"

n_qubits = 6
qc = QuantumCircuit(n_qubits)
qc.x(0)
qc.x(1)
for n_controls in range(1, n_qubits - 1):
    qc.mcx(
        list(range(n_controls)),
        n_controls + 1
    )

tqc = transpile(
    qc,
    basis_gates=("cx", "rz", "sx", "x"),
    optimization_level=3
)

print("Original:")
print(qc.count_ops())
qc.draw("mpl", filename=path + "original_qc.png")



print("\nTranspiled:")
print(tqc.count_ops())
tqc.draw("mpl", filename=path + "transpiled_qc.png")

print("\nTranspiled depth:", tqc.depth())

optimized = optimize(qc, optimization_level=3, basis_gates=["cx", "rz", "sx", "x"])
optimized.draw("mpl", filename=path + "optimized_qc.png")

pm = generate_preset_pass_manager(
    optimization_level=3,
    basis_gates=["cx", "rz", "sx", "x"],
    hls_config=HLSConfig(mcx=["n_clean_m15"]),
)

pm.pre_layout = PassManager([
    InverseCancellation([CXGate(), XGate(), HGate()]),
    CommutativeCancellation(),
    Optimize1qGatesDecomposition(basis=["rz", "sx", "x"]),
    RemoveDiagonalGatesBeforeMeasure(),
])

out = pm.run(qc)
print("-------------------------------------")
print("\nPass manager:")
print("Gate counts:", dict(out.count_ops()))
print("Total depth:", out.depth())
print("CX count:", out.count_ops().get("cx", 0))
print("CX depth:", out.depth(lambda inst: inst.operation.name == "cx"))
print("-------------------------------------")
# --- optional: see what happened, check it, explore ancillas -------------------
optimized, report = optimize(qc, return_report=True)
print(report)
print("\nverification:", verify(qc, optimized))
print("\nqubits:", qc.num_qubits, "->", optimized.num_qubits,
      "| registers:", [r.name for r in optimized.qregs])
print("optimized depth:", optimized.depth())
print("optimized gate counts:", optimized.count_ops())
qc_statevector = qc.save_statevector()
optimized_statevector = optimized.save_statevector()
result1 = sim.run(qc).result()
result2 = sim.run(optimized).result()

qc_statevector = result1.get_statevector(qc)
optimized_statevector = result2.get_statevector(optimized)

qc_statevector = np.array(qc_statevector)
optimized_statevector = np.array(optimized_statevector).round(5)
len_qc = len(qc_statevector)
with open(os.path.join(path, "qc_statevector.txt"), "w") as f:
    for i, amplitude in enumerate(qc_statevector):
        f.write(
            f"{i:>8}: "
            f"{amplitude.real:+.15e} "
            f"{amplitude.imag:+.15e}j\n"
        )

with open(os.path.join(path, "optimized_statevector.txt"), "w") as f:
    for i, amplitude in enumerate(optimized_statevector[:len_qc]):
        f.write(
            f"{i:>8}: "
            f"{amplitude.real:+.15e} "
            f"{amplitude.imag:+.15e}j\n"
        )

print(f"Saved statevectors to: {path}")



# print(
#     "equal statevector:",
#     np.allclose(qc_statevector, optimized_statevector)
# )
