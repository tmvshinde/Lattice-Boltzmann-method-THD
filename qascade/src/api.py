"""The one-call interface: ``optimize(circuit)``."""

import warnings
from dataclasses import asdict, dataclass, field

from qiskit.transpiler import PassManager, generate_preset_pass_manager

from .factor import lower_ctc, order_runs
from .ir import attach, lift, lower_plain
from .passes import simplify

DEFAULT_BASIS = ("cx", "rz", "sx", "x")

STAGE_A_STEPS = ("cancel", "pauli", "merge", "multiplex")
STAGE_B_STEPS = ("order", "and_cache", "flip")
ALL_STEPS = STAGE_A_STEPS + STAGE_B_STEPS
STEP_ALIASES = {"A1": "cancel", "A2": "pauli", "A3": "merge", "A4": "multiplex",
                "B1": "order", "B2": "and_cache", "B3": "flip"}


@dataclass
class OptimizationReport:
    """What ``optimize`` did.  Two-qubit counts are after transpiling with the same basis,
    optimisation level and seed, with qubits_initially_zero=False."""

    two_qubit_before: int
    two_qubit_after: int
    ancillas: int
    chosen: str
    steps: tuple
    candidates: dict = field(default_factory=dict)
    stage_a: dict = field(default_factory=dict)
    stage_b: dict = field(default_factory=dict)
    verification: dict = None

    @property
    def reduction(self):
        """two_qubit_before / two_qubit_after (1.0 if nothing to reduce)."""
        if not self.two_qubit_after:
            return float("inf") if self.two_qubit_before else 1.0
        return self.two_qubit_before / self.two_qubit_after

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        lines = [
            f"two-qubit gates: {self.two_qubit_before} -> {self.two_qubit_after} "
            f"({self.reduction:.2f}x), ancillas: {self.ancillas}, chosen: {self.chosen}",
            "candidates:",
        ]
        for name, c in self.candidates.items():
            mark = "*" if name == self.chosen else " "
            lines.append(f"  {mark} {name:28s} two-qubit {c['two_qubit']:6d}  ancillas {c['ancillas']}")
        if self.stage_a:
            lines.append("stage A: " + ", ".join(f"{k}={v}" for k, v in self.stage_a.items()))
        if self.stage_b:
            lines.append("stage B: " + ", ".join(f"{k}={v}" for k, v in self.stage_b.items()))
        if self.verification:
            lines.append(f"verification: {self.verification}")
        return "\n".join(lines)


_PASS_MANAGERS = {}


def transpile_exact(circuit, basis_gates=DEFAULT_BASIS, optimization_level=3, seed_transpiler=42):
    """``qiskit.transpile`` that preserves the operator *and* the qubit order.

    Two differences from a plain ``transpile(..., optimization_level=3)``:
      * qubits_initially_zero=False -- no qubit is assumed to start in |0>;
      * no ElidePermutations -- at levels 2-3 Qiskit deletes SWAP gates and records the
        permutation only in ``circuit.layout.final_layout``, so the returned circuit alone
        would implement a different operator (e.g. a QFT without its final swaps).
    """
    key = (tuple(basis_gates), optimization_level, seed_transpiler)
    pm = _PASS_MANAGERS.get(key)
    if pm is None:
        pm = generate_preset_pass_manager(optimization_level=optimization_level,
                                          basis_gates=list(basis_gates),
                                          seed_transpiler=seed_transpiler,
                                          qubits_initially_zero=False)
        if pm.init is not None:
            tasks = [t for t in pm.init.to_flow_controller().tasks
                     if type(t).__name__ != "ElidePermutations"]
            pm.init = PassManager(tasks)
        _PASS_MANAGERS[key] = pm
    return pm.run(circuit)


def _normalise_steps(steps):
    out = []
    for s in steps:
        s = STEP_ALIASES.get(s, s)
        if s not in ALL_STEPS:
            raise ValueError(f"unknown step {s!r}; choose from {ALL_STEPS} or {tuple(STEP_ALIASES)}")
        if s not in out:
            out.append(s)
    return tuple(out)


def _count(tqc):
    two = sum(1 for inst in tqc.data if len(inst.qubits) >= 2 and inst.operation.name != "barrier")
    return two, len(tqc.data)


def optimize(circuit, *, max_ancillas=None, steps=ALL_STEPS, basis_gates=DEFAULT_BASIS,
             optimization_level=3, seed_transpiler=42, search=True, tket=False,
             verify=False, return_report=False):
    """Optimise ``circuit`` for two-qubit gate count with Control-Tree Compilation.

    Parameters
    ----------
    circuit : QuantumCircuit
        Any circuit.  Operations the optimiser does not model (measurements, resets,
        classical control flow, unbound parameters, custom multi-qubit gates) are kept
        verbatim and only block rewrites that would have to move past them.
    max_ancillas : int or None
        Clean ancillas the optimiser may add (None = as many as help).  They are appended
        after the circuit's qubits in an AncillaRegister named 'ctc_anc', must start in
        |0>, and are returned to |0>.
    steps : iterable of str
        Which steps to run (default: all).  Stage A: 'cancel' (A1), 'pauli' (A2),
        'merge' (A3), 'multiplex' (A4).  Stage B: 'order' (B1), 'and_cache' (B2),
        'flip' (B3).  The A1..B3 aliases are accepted.
    basis_gates : sequence of str or None
        Transpile the result to this basis (default cx, rz, sx, x).  None returns the
        un-transpiled circuit (RCCX, CCX, CSWAP, ... gates); candidates are still scored
        in the default basis.
    optimization_level, seed_transpiler : int
        Passed to the transpiler (see ``transpile_exact``: qubits_initially_zero=False and no
        SWAP elision, so the output acts on your qubits in their original order).
    search : bool
        True (default): also try the original circuit, Stage A alone, and CTC variants with
        individual steps switched off, and keep the one with the fewest two-qubit gates,
        so the result is never worse than transpiling the input.  False: run the selected
        steps once.
    tket : bool
        Finish with pytket FullPeepholeOptimise (needs pytket, pytket-qiskit); kept only
        if it lowers the two-qubit count.
    verify : bool
        Check the result against the input by simulation (see ctc_optimizer.verify) and
        raise RuntimeError if it differs.  Circuits that cannot be simulated are skipped
        with a warning.
    return_report : bool
        Return (circuit, OptimizationReport) instead of just the circuit.

    Returns
    -------
    QuantumCircuit, or (QuantumCircuit, OptimizationReport).  The report is also stored in
    ``result.metadata['ctc']``.
    """
    steps = _normalise_steps(steps)
    score_basis = list(basis_gates) if basis_gates is not None else list(DEFAULT_BASIS)

    def tr(qc):
        return transpile_exact(qc, score_basis, optimization_level, seed_transpiler)

    n = circuit.num_qubits
    a_steps = tuple(s for s in STAGE_A_STEPS if s in steps)
    gates0 = lift(circuit)
    stage_a_stats, stage_b_stats = {}, {}
    candidates = {}                  # name -> dict(circuit, transpiled, two_qubit, ancillas, gates)

    def add(name, qc, n_anc, stats=None):
        t = tr(qc)
        two, total = _count(t)
        candidates[name] = {"circuit": qc, "transpiled": t, "two_qubit": two,
                            "ancillas": n_anc, "gates": total, "stats": stats or {}}

    # baseline: the input itself
    add("original", circuit, 0)

    simplified = {}

    def stage_a(a):
        if a not in simplified:
            simplified[a] = simplify(gates0, a) if a else (gates0, {})
        return simplified[a]

    gA, statsA = stage_a(a_steps)
    stage_a_stats = statsA
    if a_steps and (search or "and_cache" not in steps):
        ops, ph = lower_plain(gA)
        add("stage_a", attach(circuit, ops, 0, ph), 0)

    def ctc(name, a, order, flips, reserve=0):
        g, _ = stage_a(a)
        if order:
            g, _ = order_runs(g)
        ops, used, ph, st = lower_ctc(g, n, max_anc=max_ancillas, reserve=reserve, flips=flips)
        add(name, attach(circuit, ops, used, ph), used, st)

    if "and_cache" in steps:
        use_order, use_flip = "order" in steps, "flip" in steps
        ctc("ctc", a_steps, use_order, use_flip)
        if search:
            if max_ancillas is not None and max_ancillas >= 1:
                ctc("ctc, reserve 1 ancilla", a_steps, use_order, use_flip, reserve=1)
            if use_flip:
                ctc("ctc without flip", a_steps, use_order, False)
            if use_order:
                ctc("ctc without order", a_steps, False, use_flip)
            if "multiplex" in a_steps:
                ctc("ctc without multiplex", tuple(s for s in a_steps if s != "multiplex"),
                    use_order, use_flip)

    pool = candidates if search else {k: v for k, v in candidates.items() if k != "original"} or candidates
    chosen = min(pool, key=lambda k: (pool[k]["two_qubit"], pool[k]["ancillas"], pool[k]["gates"]))
    best = candidates[chosen]
    result = best["transpiled"] if basis_gates is not None else best["circuit"]
    two_after = best["two_qubit"]
    stage_b_stats = best["stats"]

    if tket and basis_gates is not None:
        polished = _tket_polish(best["transpiled"], tr)
        if polished is not None:
            two, total = _count(polished)
            if two < two_after:
                result, two_after, chosen = polished, two, chosen + " + tket"

    report = OptimizationReport(
        two_qubit_before=candidates["original"]["two_qubit"],
        two_qubit_after=two_after,
        ancillas=best["ancillas"],
        chosen=chosen,
        steps=steps,
        candidates={k: {"two_qubit": v["two_qubit"], "ancillas": v["ancillas"]}
                    for k, v in candidates.items()},
        stage_a=stage_a_stats,
        stage_b=stage_b_stats,
    )

    if verify:
        from .verify import NotVerifiable, verify as _verify
        try:
            report.verification = _verify(circuit, result)
        except NotVerifiable as exc:
            report.verification = {"passed": None, "method": f"not verified: {exc}"}
            warnings.warn(f"ctc_optimizer: result not verified ({exc})", stacklevel=2)
        else:
            if not report.verification["passed"]:
                raise RuntimeError(f"ctc_optimizer produced a non-equivalent circuit: "
                                   f"{report.verification}")

    if result is circuit:
        result = circuit.copy()              # never mutate the caller's circuit
    result.metadata = dict(circuit.metadata or {}, ctc=report.as_dict())
    return (result, report) if return_report else result


def _tket_polish(tqc, tr):
    try:
        from pytket import OpType
        from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
        from pytket.passes import FullPeepholeOptimise
    except ImportError:
        warnings.warn("tket=True needs pytket and pytket-qiskit; skipped", stacklevel=3)
        return None
    try:
        tk = qiskit_to_tk(tqc)
        FullPeepholeOptimise(allow_swaps=False, target_2qb_gate=OpType.CX).apply(tk)
        back = tk_to_qiskit(tk)
    except Exception as exc:  # noqa: BLE001 - unsupported instructions etc.
        warnings.warn(f"tket polish skipped: {exc}", stacklevel=3)
        return None
    out = map_by_register(tqc, back)
    return None if out is None else tr(out)


def map_by_register(template, converted):
    """Rebuild ``converted`` onto ``template``'s bits, matching (register name, index).

    pytket's tk_to_qiskit orders registers by name, so bit *positions* change for circuits
    with several registers (e.g. 't' and 's' come back as 's', 't').  Matching by register
    name and index restores the original qubit order.  Returns None if a bit cannot be
    matched (e.g. bits that belong to no register).
    """
    if converted.num_qubits != template.num_qubits or converted.num_clbits != template.num_clbits:
        return None
    lookup = {}
    for reg in list(template.qregs) + list(template.cregs):
        for i, bit in enumerate(reg):
            lookup[(reg.name, i, "q" if reg in template.qregs else "c")] = bit
    out = template.copy_empty_like()

    def find(bit, kind):
        loc = converted.find_bit(bit)
        if not loc.registers:
            return None
        reg, idx = loc.registers[0]
        return lookup.get((reg.name, idx, kind))

    for inst in converted.data:
        qs = [find(q, "q") for q in inst.qubits]
        cs = [find(c, "c") for c in inst.clbits]
        if None in qs or None in cs:
            return None
        out.append(inst.operation, qs, cs)
    out.global_phase = converted.global_phase
    return out


def ancilla_tradeoff(circuit, budgets=None, **kwargs):
    """Two-qubit count against the ancilla budget.

    Returns a list of dicts {'max_ancillas', 'ancillas', 'two_qubit', 'pareto'} for each
    budget (default: 0 up to what the unrestricted optimisation uses).  Extra keyword
    arguments go to ``optimize``.
    """
    kwargs.pop("return_report", None)
    kwargs.pop("max_ancillas", None)
    _, top = optimize(circuit, return_report=True, **kwargs)
    if budgets is None:
        budgets = list(range(0, top.ancillas + 1))
    rows = []
    for b in budgets:
        _, rep = optimize(circuit, max_ancillas=b, return_report=True, **kwargs)
        rows.append({"max_ancillas": b, "ancillas": rep.ancillas, "two_qubit": rep.two_qubit_after})
    best = None
    for r in sorted(rows, key=lambda r: (r["ancillas"], r["two_qubit"])):
        r["pareto"] = best is None or r["two_qubit"] < best
        if r["pareto"]:
            best = r["two_qubit"]
    return rows
