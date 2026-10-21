# ctc-optimizer

**Control-Tree Compilation** for Qiskit: fewer two-qubit gates for circuits built from
multi-controlled gates.

```python
from ctc_optimizer import optimize

optimized = optimize(my_circuit)
```

That's the whole interface. The result is exact: it implements the same unitary as your
circuit, with any added ancillas starting and ending in |0⟩. By default it never has more
two-qubit gates than transpiling your circuit directly.

On the curl-curl solver it was developed for, one full timestep at N=8 goes from
**2 458 to 530 CX** on the same 7 ancillas (4.6×). The details are in
`resource_estimation/CTC_REPORT.pdf`.

---

## When it helps

CTC targets circuits that are really **quantum if-statements**, where many gates share
some of their control qubits:

- selector- or oracle-controlled blocks, e.g. "apply this ladder only when register `s` = 5"
- controlled incrementers, decrementers and shift operators (quantum walks, lattice
  Boltzmann)
- controlled register permutations and controlled swaps
- multiplexed rotations (the same target rotated under different control values)
- circuits with redundant multi-controlled pairs, or `Z X Z X`-style groups

Qiskit synthesises each multi-controlled gate independently, so shared controls are
recomputed for every gate. A circuit with no shared multi-controlled logic (e.g. plain
CX + rotation layers) will see little change, and `optimize` then simply returns the best
of the variants it tried, which may be the ordinary transpiled circuit.

## Install

```bash
pip install -e path/to/ctc_optimizer            # core: qiskit>=2.0, numpy
pip install -e "path/to/ctc_optimizer[verify]"  # + qiskit-aer for verify() on >20 qubits
pip install -e "path/to/ctc_optimizer[all]"     # + pytket for the optional polish pass
```

In this repository it is already installed (editable) into `venv/`.

## Usage

```python
from ctc_optimizer import optimize, verify, ancilla_tradeoff

out = optimize(qc)                                 # transpiled to cx, rz, sx, x
out, report = optimize(qc, return_report=True)     # + what happened
print(report)

out = optimize(qc, max_ancillas=3)                 # cap the extra qubits
out = optimize(qc, basis_gates=["cz", "rz", "sx", "x"])  # another basis
out = optimize(qc, basis_gates=None)               # keep high-level gates (rccx, ccx, cswap, ...)
out = optimize(qc, steps=["A1", "A2", "B2"])       # choose steps
out = optimize(qc, verify=True)                    # simulate and raise if not equivalent
out = optimize(qc, tket=True)                      # finish with pytket FullPeepholeOptimise

verify(qc, out)             # {'passed': True, 'fidelity': 1.0, 'method': 'statevector', ...}
ancilla_tradeoff(qc)        # two-qubit count at each ancilla budget, with the Pareto front
```

### `optimize` arguments

| argument | default | meaning |
|---|---|---|
| `max_ancillas` | `None` | Clean ancillas it may add; `None` means as many as help. |
| `steps` | all | Any subset of the steps below, by name or alias. |
| `basis_gates` | `("cx","rz","sx","x")` | Transpile target; `None` returns untranspiled gates. |
| `optimization_level` | `3` | Passed to `qiskit.transpile`. |
| `seed_transpiler` | `42` | Passed to `qiskit.transpile`. |
| `search` | `True` | Try several variants (see below) and keep the best. |
| `tket` | `False` | pytket polish, mapped back to your qubits by register; kept only if it lowers the count. |
| `verify` | `False` | Simulate and raise `RuntimeError` on any difference. |
| `return_report` | `False` | Return `(circuit, OptimizationReport)`. |

The report is also stored in `out.metadata["ctc"]`.

**Command line:**

```bash
python -m ctc_optimizer circuit.qpy -o optimized.qpy --verify
python -m ctc_optimizer circuit.qasm -o optimized.qasm --max-ancillas 4
```

## The steps

All steps are exact: each is an algebraic identity, not an approximation.

### Stage A — rewrites on the circuit, before synthesis

| step | alias | what it does |
|---|---|---|
| `cancel` | A1 | Removes pairs of identical controlled gates when everything between them commutes with them. Also merges rotations. |
| `pauli` | A2 | Multiplies X/Z gates under the same controls into one Pauli times a phase. `ZXZX` = −I becomes a phase on the controls alone. |
| `merge` | A3 | Merges the same operation under two control patterns that differ in one bit, dropping that control. |
| `multiplex` | A4 | Turns controlled Ry gates on one target into one uniformly controlled rotation (a multiplexor). |

### Stage B — synthesis with shared control

| step | alias | what it does |
|---|---|---|
| `order` | B1 | Regroups gates whose shared "selector" controls take different values: they act on orthogonal subspaces, so they commute. Groups with similar values are placed next to each other. |
| `and_cache` | B2 | Computes each shared AND of controls once onto a clean ancilla with relative-phase Toffolis (3 CX each) and keeps it while later gates need it. This is the core step. |
| `flip` | B3 | Turns a cached K∧l into K∧¬l with one CX plus a diagonal phase fix (2 CX), instead of uncomputing and recomputing (6 CX). |

Any multi-controlled X left over uses Maslov's clean-ancilla v-chain, or the Khattar–Gidney
1-ancilla construction when ancillas are scarce.

### How `search` picks the result

With `search=True` it transpiles these candidates and keeps the one with the fewest
two-qubit gates. Ties go to fewer ancillas, then fewer gates.

1. your circuit, untouched
2. Stage A only
3. full CTC
4. CTC with `flip`, `order` or `multiplex` switched off, one at a time
5. CTC holding one ancilla in reserve, if `max_ancillas` ≥ 1

With `search=False` it runs your chosen steps once.

## The output circuit

- **Qubits.** Your qubits come first, in order, with your registers and names. Ancillas
  follow in an `AncillaRegister` named **`ctc_anc`**. Ancillas must start in |0⟩ and are
  returned to |0⟩.
- **Kept verbatim.** Measurements, resets, classical control flow, unbound parameters and
  unrecognised gates are kept as they are. They act as walls that rewrites don't move past.
  Barriers are removed.
- **Transpile settings.** Transpilation uses `transpile_exact`: `qubits_initially_zero=False`,
  and Qiskit's `ElidePermutations` pass is removed. At optimisation levels 2–3 that pass
  deletes SWAP gates and hides the resulting qubit permutation in `layout.final_layout`. The
  returned circuit therefore acts on your qubits in their original order, on every input
  state. If you transpile the output again yourself,
  keep that setting unless your circuit truly starts from |0…0⟩. With `True`, Qiskit may
  borrow qubits as clean ancillas when they aren't.

## Verification

`verify(original, optimized)` compares the two by simulation. It needs `qiskit-aer`
above 20 qubits.

- **Up to 26 qubits:** a random state on all original qubits, ancillas in |0⟩. The
  fidelity is 1 only if the circuits agree *and* every ancilla returns to |0⟩.
- **27–28 qubits:** a mirror check. It prepares a random entangled state S, runs
  S; optimised; original†; S†, and reads one amplitude.
- **Not verifiable:** circuits with measurements or unbound parameters raise
  `NotVerifiable`.

The test suite checks exactness on randomised circuits covering these cases:
open controls, shared selectors, controlled swaps, multiplexed rotations, CP/CRZ/CS/CY/CH,
redundant pairs and `ZXZX` groups. Every candidate variant is checked, not just the one
chosen.

```bash
python tests/test_ctc.py                    # or: pytest tests/
python examples/quickstart.py
python examples/curl_curl_timestep.py 4 8   # reproduces the report's numbers
```

## Limits

- It **optimises two-qubit gate count.** T-count often drops too, but is not targeted.
- **Ancillas are real qubits.** Use `max_ancillas` or `ancilla_tradeoff` to pick a
  budget. With `max_ancillas=0` the gains are usually small.
- **Stage B is greedy** with a CX cost model. It is not optimal, and more ancillas do not
  always mean fewer gates (`search` and `ancilla_tradeoff` cover this).
- **Runtime:** `search=True` transpiles about six variants. For very large circuits use
  `search=False`.

## References

1. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application to
   multiple control Toffoli optimization*, PRA 93, 022311 (2016). arXiv:1508.03273
2. C. Yuan, M. Carbin, *The T-Complexity Costs of Error Correction for Control Flow in
   Quantum Computation*, PLDI 2024. arXiv:2311.12772
3. T. Khattar, C. Gidney, *Rise of conditionally clean ancillae for optimizing quantum
   circuits*, Quantum 9, 1752 (2025). arXiv:2407.17966
4. C. Gidney, *Halving the cost of quantum addition*, Quantum 2, 74 (2018). arXiv:1709.06648
5. D. Maslov, G. W. Dueck, D. M. Miller, *Toffoli network synthesis with templates*,
   IEEE TCAD 24(6) (2005)
6. M. M. Rahman, J. E. Rice, *Templates for positive and negative control Toffoli
   networks*, RC 2014, LNCS 8507
7. M. Möttönen et al., *Quantum circuits for general multiqubit gates*, PRL 93, 130502
   (2004). quant-ph/0404089
8. V. V. Shende, S. S. Bullock, I. L. Markov, *Synthesis of quantum logic circuits*,
   IEEE TCAD 25(6) (2006). quant-ph/0406176
