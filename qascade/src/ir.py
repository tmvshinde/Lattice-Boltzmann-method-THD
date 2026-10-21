"""Intermediate representation: a circuit as a list of guarded operations.

Each gate ``G`` is  (cube, kind, targets, ...):

    cube     tuple of (qubit, polarity) literals, sorted by qubit.  The op fires iff every
             listed qubit equals its polarity.
    kind     'x' | 'z' | 'h' | 'ry' | 'swap'   recognised cores
             'gate1'   any other single-qubit gate with a known matrix (s, t, rz, p, y, ...)
             'phase'   exp(i*param) on the cube's subspace (no targets)
             'ucry'    uniformly controlled Ry (made by the multiplexor pass)
             'raw'     anything else, kept verbatim: measurements, resets, classical
                       control flow, unbound parameters, multi-qubit custom gates, ...
    order    control qubits in the order the source circuit listed them (not part of
             equality).  Lowering keeps this order, because multi-controlled synthesis ANDs
             controls in list order and consecutive gates sharing a leading prefix are
             what Qiskit's optimiser can cancel.

``lift`` turns a QuantumCircuit into a gate list.  Qubits and clbits are referred to by
index, so the result can be re-attached to the source circuit's registers.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Gate
from qiskit.circuit.library import (
    HGate,
    MCXGate,
    XGate,
    PhaseGate,
    RYGate,
    SwapGate,
    UCRYGate,
    ZGate,
)

DIAG_TOL = 1e-12
SELF_INVERSE_KINDS = ("x", "z", "swap", "h")
_MATRIX_CHECK_MAX_CONTROLS = 6


@dataclass(frozen=True)
class G:
    cube: tuple
    kind: str
    targets: tuple = ()
    param: object = None
    selects: tuple = ()
    op: object = field(default=None, compare=False)       # raw / gate1: the qiskit op
    order: tuple = field(default=(), compare=False)
    clbits: tuple = field(default=(), compare=False)      # raw only
    matrix: object = field(default=None, compare=False)   # gate1 only

    @property
    def ctrl_qubits(self):
        return tuple(q for q, _ in self.cube)

    @property
    def ordered_cube(self):
        rank = {q: i for i, q in enumerate(self.order)}
        return tuple(sorted(self.cube, key=lambda l: (rank.get(l[0], len(rank)), l[0])))

    @property
    def diagonal(self):
        if self.kind in ("z", "phase"):
            return True
        if self.kind == "gate1":
            m = self.matrix
            return abs(m[0, 1]) < DIAG_TOL and abs(m[1, 0]) < DIAG_TOL
        return False

    @property
    def writes(self):
        """Qubits whose computational-basis value this gate can change."""
        if self.diagonal:
            return ()
        if self.kind == "raw":
            return self.ctrl_qubits + self.targets
        return self.targets

    @property
    def support(self):
        return set(self.ctrl_qubits) | set(self.targets) | set(self.selects)

    def with_cube(self, cube):
        return G(tuple(sorted(cube)), self.kind, self.targets, self.param, self.selects,
                 self.op, self.order, self.clbits, self.matrix)


def lit_state(cube):
    """ctrl_state integer for the cube's qubits in cube order."""
    return sum(p << i for i, (_, p) in enumerate(cube))


# ---------------------------------------------------------------------------
# lift
# ---------------------------------------------------------------------------

def _full_matrix(gate):
    """gate.to_matrix(), falling back to Operator(gate) (built from the definition), or None."""
    try:
        return np.asarray(gate.to_matrix(), dtype=complex)
    except Exception:  # noqa: BLE001 - many gates (MCXGate, ControlledGate) define no to_matrix
        pass
    try:
        from qiskit.quantum_info import Operator
        return np.asarray(Operator(gate).data, dtype=complex)
    except Exception:  # noqa: BLE001 - unbound parameters, opaque gates, ...
        return None


def _matrix(gate):
    m = _full_matrix(gate)
    return m if m is not None and m.shape == (2, 2) else None


def _controlled_semantics_ok(op, base_matrix, nc):
    """True if ``op`` really is |ctrl_state><ctrl_state| (x) base + rest (x) I.

    Qiskit's CUGate, for example, carries a phase that its base_gate does not, so the base
    alone does not describe it.  Checked by matrix for up to a few controls; beyond that
    only gate classes known to be built as plain controls are accepted.
    """
    if nc > _MATRIX_CHECK_MAX_CONTROLS:
        return type(op).__name__ in ("ControlledGate", "MCPhaseGate", "MCU1Gate")
    full = _full_matrix(op)
    if full is None:
        return False
    dim = 2 ** (nc + 1)
    ref = np.eye(dim, dtype=complex)
    s = op.ctrl_state
    for t_in in (0, 1):
        for t_out in (0, 1):
            ref[s | t_out << nc, s | t_in << nc] = base_matrix[t_out, t_in]
    return full.shape == (dim, dim) and np.allclose(full, ref, atol=1e-10)


def _raw(op, qs, cs):
    return G((), "raw", tuple(qs), op=op, clbits=tuple(cs))


def lift(qc, drop_barriers=True):
    gates = []
    for inst in qc.data:
        op = inst.operation
        qs = [qc.find_bit(q).index for q in inst.qubits]
        cs = [qc.find_bit(c).index for c in inst.clbits]
        if op.name == "barrier":
            if not drop_barriers:
                gates.append(_raw(op, qs, cs))
            continue
        if cs or not isinstance(op, Gate):
            gates.append(_raw(op, qs, cs))
            continue

        if isinstance(op, ControlledGate):
            nc = op.num_ctrl_qubits
            base = op.base_gate
            if op.num_qubits != nc + base.num_qubits:          # e.g. MCX with ancillas
                gates.append(_raw(op, qs, cs))
                continue
            cube = tuple(sorted((qs[i], (op.ctrl_state >> i) & 1) for i in range(nc)))
            tg, order = tuple(qs[nc:]), tuple(qs[:nc])
        else:
            base, nc, cube, tg, order = op, 0, (), tuple(qs), ()

        name = base.name
        # isinstance, not class names: Qiskit 2 gates are singleton subclasses (_SingletonXGate)
        if isinstance(base, (XGate, ZGate, HGate)) and len(tg) == 1:
            gates.append(G(cube, name, tg, order=order))
            continue
        if isinstance(base, SwapGate) and len(tg) == 2:
            gates.append(G(cube, "swap", tg, order=order))
            continue
        if base.num_qubits == 1:
            m = _matrix(base)
            if m is None or (nc and not _controlled_semantics_ok(op, m, nc)):
                gates.append(_raw(op, qs, cs))
                continue
            if name == "ry":
                gates.append(G(cube, "ry", tg, float(base.params[0]), order=order))
                continue
            # keyed by the matrix, not the params: robust for UnitaryGate and aliases
            key = ("u1q", tuple(complex(round(v.real, 12), round(v.imag, 12)) for v in m.flatten()))
            gates.append(G(cube, "gate1", tg, key, op=base, order=order, matrix=m))
            continue
        gates.append(_raw(op, qs, cs))
    return gates


# ---------------------------------------------------------------------------
# lower (Stage-A-only path): IR -> multi-controlled qiskit gates
# ---------------------------------------------------------------------------

def _controlled(base, cube):
    if not cube:
        return base
    return base.control(len(cube), ctrl_state=lit_state(cube))


def append_gate(ops, g):
    """Append one IR gate to ``ops`` (a list of (instruction, qubits, clbits))."""
    cube = g.ordered_cube
    cq = [q for q, _ in cube]
    if g.kind == "x":
        if cube:
            ops.append((MCXGate(len(cube), ctrl_state=lit_state(cube)), cq + list(g.targets), []))
        else:
            ops.append((XGate(), list(g.targets), []))
    elif g.kind == "z":
        ops.append((_controlled(ZGate(), cube), cq + list(g.targets), []))
    elif g.kind == "h":
        ops.append((_controlled(HGate(), cube), cq + list(g.targets), []))
    elif g.kind == "swap":
        ops.append((_controlled(SwapGate(), cube), cq + list(g.targets), []))
    elif g.kind == "ry":
        ops.append((_controlled(RYGate(g.param), cube), cq + list(g.targets), []))
    elif g.kind == "gate1":
        ops.append((_controlled(g.op, cube), cq + list(g.targets), []))
    elif g.kind == "phase":
        if not cube:
            return g.param                       # global phase
        (tq, tp), rest = cube[-1], cube[:-1]
        if tp == 0:
            ops.append((XGate(), [tq], []))
        if abs(math.remainder(g.param - math.pi, 2 * math.pi)) < 1e-12 and rest:
            ops.append((HGate(), [tq], []))
            ops.append((MCXGate(len(rest), ctrl_state=lit_state(rest)), [q for q, _ in rest] + [tq], []))
            ops.append((HGate(), [tq], []))
        else:
            ops.append((_controlled(PhaseGate(g.param), rest), [q for q, _ in rest] + [tq], []))
        if tp == 0:
            ops.append((XGate(), [tq], []))
    elif g.kind == "ucry":
        gate = UCRYGate(list(g.param))
        if cube:
            gate = gate.control(len(cube), ctrl_state=lit_state(cube))
        ops.append((gate, cq + list(g.targets) + list(g.selects), []))
    elif g.kind == "raw":
        ops.append((g.op, list(g.targets), list(g.clbits)))
    else:
        raise ValueError(g.kind)
    return 0.0


def lower_plain(gates):
    """Stage-A-only lowering. Returns (ops, extra_global_phase)."""
    ops, phase = [], 0.0
    for g in gates:
        phase += append_gate(ops, g)
    return ops, phase


def attach(source, ops, n_anc, extra_phase=0.0, name=None):
    """Build the output circuit on the source circuit's own registers + ancillas.

    ``ops`` refer to qubits/clbits by index; indices >= source.num_qubits are ancillas,
    placed in an AncillaRegister named 'ctc_anc' after the source qubits.
    """
    from qiskit.circuit import AncillaRegister

    out = source.copy_empty_like(name=name or source.name)
    if n_anc:
        out.add_register(AncillaRegister(n_anc, "ctc_anc"))
    for inst, qs, cs in ops:
        out.append(inst, [out.qubits[i] for i in qs], [out.clbits[j] for j in cs], copy=False)
    out.global_phase = source.global_phase + extra_phase
    return out


# ---------------------------------------------------------------------------
# small exact matrices for commutation tests
# ---------------------------------------------------------------------------

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def core_matrix(g, wires):
    """Matrix of the core op (controls stripped) on ``wires`` (little-endian), or None."""
    n = len(wires)
    pos = {q: i for i, q in enumerate(wires)}

    def one(m, q):
        full = np.array([[1]], dtype=complex)
        for i in reversed(range(n)):
            full = np.kron(full, m if i == pos[q] else np.eye(2))
        return full

    if g.kind == "x":
        return one(_X, g.targets[0])
    if g.kind == "z":
        return one(_Z, g.targets[0])
    if g.kind == "h":
        return one(_H, g.targets[0])
    if g.kind == "ry":
        return one(_ry(g.param), g.targets[0])
    if g.kind == "gate1":
        return one(g.matrix, g.targets[0])
    if g.kind == "swap":
        a, b = pos[g.targets[0]], pos[g.targets[1]]
        dim = 2 ** n
        m = np.zeros((dim, dim), dtype=complex)
        for s in range(dim):
            ba, bb = (s >> a) & 1, (s >> b) & 1
            t = s & ~(1 << a) & ~(1 << b) | (bb << a) | (ba << b)
            m[t, s] = 1
        return m
    return None
