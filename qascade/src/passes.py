"""Stage A -- exact simplification on the IR, before any synthesis.

A1  cancel     commutation-aware cancellation / merging of identical guarded gates
A2  pauli      X/Z products under one cube -> one Pauli times a cube phase (ZXZX = -I)
A3  merge      the same op under two cubes differing in one literal -> drop the literal
A4  multiplex  Ry gates on one target under cubes over a few qubits -> one UCRY

Every rewrite relies only on ``commute``, whose rules are each an exact algebraic fact.
"""

import itertools
import math

import numpy as np

from .ir import SELF_INVERSE_KINDS, G, core_matrix

TOL = 1e-12
_CORE_KINDS = ("x", "z", "h", "ry", "swap", "gate1")


# ---------------------------------------------------------------------------
# commutation
# ---------------------------------------------------------------------------

def _core_commute(g, h):
    tg, th = set(g.targets), set(h.targets)
    if not tg & th:
        return True
    if g.kind not in _CORE_KINDS or h.kind not in _CORE_KINDS:
        return False
    wires = sorted(tg | th)
    if len(wires) > 4:
        return False
    a, b = core_matrix(g, wires), core_matrix(h, wires)
    return np.allclose(a @ b, b @ a, atol=1e-10)


def commute(g, h):
    """Sufficient (sound, not complete) test that gates g and h commute."""
    if g.kind == "raw" or h.kind == "raw":
        return not (g.support & h.support)
    # 1. disjoint supports
    if not (g.support & h.support):
        return True
    # 2. two diagonal gates
    if g.diagonal and h.diagonal:
        return True
    # 3. a diagonal gate commutes with anything that does not change its qubits' values:
    #    the other gate is block-diagonal in the diagonal gate's computational basis.
    if g.diagonal and not (g.support & set(h.writes)):
        return True
    if h.diagonal and not (h.support & set(g.writes)):
        return True
    # 4. orthogonal cubes on a qubit neither gate writes: the space splits into invariant
    #    subspaces on each of which at most one of the two gates acts.
    hc = dict(h.cube)
    written = set(g.writes) | set(h.writes)
    for q, p in g.cube:
        if q in hc and hc[q] != p and q not in written:
            return True
    # 5. controls untouched by the other gate's targets, and core ops that commute
    if g.kind == "ucry" or h.kind == "ucry":
        return False
    if set(g.ctrl_qubits) & set(h.targets) or set(h.ctrl_qubits) & set(g.targets):
        return False
    return _core_commute(g, h)


def _commutes_with_all(g, gates):
    return all(commute(g, x) for x in gates)


# ---------------------------------------------------------------------------
# A1 -- cancellation / merging
# ---------------------------------------------------------------------------

def _merge(g, h):
    """(merged_gate_or_None, ok).  ok=False if g and h do not merge."""
    if g.cube != h.cube or g.kind != h.kind or g.kind in ("raw", "ucry"):
        return None, False
    if g.kind in SELF_INVERSE_KINDS:
        same = set(g.targets) == set(h.targets) if g.kind == "swap" else g.targets == h.targets
        return (None, True) if same else (None, False)
    if g.kind == "ry" and g.targets == h.targets:
        t = g.param + h.param
        if abs(math.remainder(t, 4 * math.pi)) < TOL:
            return None, True
        if abs(math.remainder(t - 2 * math.pi, 4 * math.pi)) < TOL:     # Ry(2pi) = -I
            return G(g.cube, "phase", (), math.pi, order=g.order), True
        return G(g.cube, "ry", g.targets, t, order=g.order), True
    if g.kind == "gate1" and g.targets == h.targets:
        prod = h.matrix @ g.matrix
        w = prod[0, 0]
        if abs(abs(w) - 1) < 1e-10 and np.allclose(prod, w * np.eye(2), atol=1e-10):
            ang = float(np.angle(w))
            if abs(ang) < 1e-12:
                return None, True
            return G(g.cube, "phase", (), ang, order=g.order), True
        return None, False
    if g.kind == "phase":
        t = g.param + h.param
        if abs(math.remainder(t, 2 * math.pi)) < TOL:
            return None, True
        return G(g.cube, "phase", (), t, order=g.order), True
    return None, False


def cancel(gates):
    """A1. Repeated forward sweeps until nothing cancels."""
    gates = list(gates)
    removed = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(gates):
            g = gates[i]
            between = []
            hit = None
            for j in range(i + 1, len(gates)):
                h = gates[j]
                merged, ok = _merge(g, h)
                if ok and _commutes_with_all(h, between):
                    hit = (j, merged)
                    break
                if not commute(g, h):
                    break
                between.append(h)
            if hit is None:
                i += 1
                continue
            j, merged = hit
            del gates[j]
            if merged is None:
                del gates[i]
                removed += 2
            else:
                gates[i] = merged
                removed += 1
            changed = True
    return gates, removed


# ---------------------------------------------------------------------------
# A2 -- Pauli-group collapse
# ---------------------------------------------------------------------------

_PAULI = {"x": np.array([[0, 1], [1, 0]], dtype=complex),
          "z": np.diag([1, -1]).astype(complex)}


def _decompose_pauli(u):
    """u == w * X^a Z^b  ->  (a, b, w)."""
    for a, b in itertools.product((0, 1), repeat=2):
        base = np.linalg.matrix_power(_PAULI["x"], a) @ np.linalg.matrix_power(_PAULI["z"], b)
        w = np.trace(base.conj().T @ u) / 2
        if abs(abs(w) - 1) < 1e-9 and np.allclose(u, w * base, atol=1e-9):
            return a, b, w
    raise ValueError("not a Pauli")


def pauli_collapse(gates):
    """A2. Contiguous X/Z windows under one cube on one target."""
    out, i, collapsed = [], 0, 0
    while i < len(gates):
        g = gates[i]
        if g.kind in ("x", "z"):
            j = i
            while (j + 1 < len(gates) and gates[j + 1].kind in ("x", "z")
                   and gates[j + 1].cube == g.cube and gates[j + 1].targets == g.targets):
                j += 1
            if j > i:
                u = np.eye(2, dtype=complex)
                for h in gates[i:j + 1]:
                    u = _PAULI[h.kind] @ u
                a, b, w = _decompose_pauli(u)
                new = []
                if b:
                    new.append(G(g.cube, "z", g.targets, order=g.order))
                if a:
                    new.append(G(g.cube, "x", g.targets, order=g.order))
                ang = float(np.angle(w))
                if abs(ang) > 1e-9:
                    new.append(G(g.cube, "phase", (), ang, order=g.order))
                if len(new) < j - i + 1:          # only when it shrinks the window
                    out.extend(new)
                    collapsed += 1
                    i = j + 1
                    continue
        out.append(g)
        i += 1
    return out, collapsed


# ---------------------------------------------------------------------------
# A3 -- one-literal cube merging
# ---------------------------------------------------------------------------

def _one_literal_apart(c1, c2):
    if len(c1) != len(c2) or [q for q, _ in c1] != [q for q, _ in c2]:
        return None
    diff = [q for (q, p1), (_, p2) in zip(c1, c2) if p1 != p2]
    return diff[0] if len(diff) == 1 else None


def cube_merge(gates):
    """A3. Repeated forward sweeps until nothing merges."""
    gates = list(gates)
    merged = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(gates):
            g = gates[i]
            if g.kind in ("raw", "ucry") or not g.cube:
                i += 1
                continue
            between, hit = [], None
            for j in range(i + 1, len(gates)):
                h = gates[j]
                same_op = (h.kind == g.kind and h.param == g.param
                           and (set(h.targets) == set(g.targets) if g.kind == "swap"
                                else h.targets == g.targets))
                q = _one_literal_apart(g.cube, h.cube) if same_op else None
                if q is not None and _commutes_with_all(h, between):
                    hit = (j, q)
                    break
                if not commute(g, h):
                    break
                between.append(h)
            if hit is None:
                i += 1
                continue
            j, q = hit
            gates[i] = g.with_cube(tuple(l for l in g.cube if l[0] != q))
            del gates[j]
            merged += 1
            changed = True
    return gates, merged


# ---------------------------------------------------------------------------
# A4 -- multiplexor detection
# ---------------------------------------------------------------------------

def multiplex(gates, max_selects=4):
    """A4. Ry gates on one target (adjacent modulo commutation) -> one UCRY.

    Literals shared by every cube in the group stay as the UCRY's control cube; the other
    control qubits become select qubits.  The angle table is, for each select state, the
    sum of the angles of the group's gates whose cube covers it (commuting Ry on one
    target add).
    """
    gates = list(gates)
    made = 0
    i = 0
    while i < len(gates):
        g = gates[i]
        if g.kind != "ry" or not g.cube:
            i += 1
            continue
        group, between, idx = [g], [], [i]
        for j in range(i + 1, len(gates)):
            h = gates[j]
            if (h.kind == "ry" and h.targets == g.targets and h.cube
                    and _commutes_with_all(h, between)):
                group.append(h)
                idx.append(j)
                continue
            if not _commutes_with_all(h, group):
                break
            between.append(h)
        if len(group) < 2:
            i += 1
            continue
        common = set(group[0].cube)
        for h in group[1:]:
            common &= set(h.cube)
        common_q = {q for q, _ in common}
        selects = sorted({q for h in group for q in h.ctrl_qubits} - common_q)
        if not selects or len(selects) > max_selects:
            i += 1
            continue
        table = [0.0] * (2 ** len(selects))
        for s in range(len(table)):
            bits = {q: (s >> k) & 1 for k, q in enumerate(selects)}
            for h in group:
                if all(bits.get(q, p) == p for q, p in h.cube if q not in common_q):
                    table[s] += h.param
        ucry = G(tuple(sorted(common)), "ucry", g.targets, tuple(table), tuple(selects),
                 order=g.order)
        for j in reversed(idx[1:]):
            del gates[j]
        gates[i] = ucry
        made += 1
        i += 1
    return gates, made


# ---------------------------------------------------------------------------

STAGE_A = {"cancel": cancel, "pauli": pauli_collapse, "merge": cube_merge, "multiplex": multiplex}
# Pauli collapse runs first so a ZXZX group is not broken up by cancelling one of its X's.
STAGE_A_ORDER = ("pauli", "cancel", "merge", "multiplex")


def simplify(gates, steps=STAGE_A_ORDER):
    """Run the selected Stage-A steps to a fixed point.  Returns (gates, stats)."""
    stats = {"cancel": 0, "pauli": 0, "merge": 0, "multiplex": 0}
    ordered = [s for s in STAGE_A_ORDER if s in steps]
    while True:
        before = (len(gates), dict(stats))
        for s in ordered:
            gates, n = STAGE_A[s](gates)
            stats[s] += n
        if (len(gates), stats) == before:
            return gates, stats
