"""Stage B -- selector factoring.

B1  order     ``order_runs``: inside maximal segments whose gates all read a common set Q of
              >= 2 control qubits that nothing in the segment writes, gates with different
              literal values on Q act on orthogonal invariant subspaces and commute.  They
              are regrouped by value and the groups ordered by a Held-Karp tour minimising
              Hamming distance between consecutive values.

B2  and_cache ``lower_ctc``: an AND-cache.  A node is a set of literals whose conjunction
              is held on a clean ancilla, computed from a parent node plus one literal by
              a relative-phase Toffoli (RCCX, 3 CX); singletons are free.  Each gate is
              emitted under the deepest cached node inside its cube; a node is extended
              while the CX it saves on current and future gates pays for it, and
              uncomputed when nothing later can use it, before anything writes its
              qubits, or on budget pressure (farthest next use first).

B3  flip      sibling flips: a leaf node holding K^l becomes K^(not l) with one CX
              (K^l XOR K = K^(not l)) plus a diagonal phase correction on (K, l) derived
              numerically for the installed Qiskit's RCCX.  Typically 2 CX instead of 6.

Exactness is Maslov's relative-phase argument applied across many gates: between a node's
compute and its uncompute nothing writes its literal qubits, and its ancilla is only read
(as a control) or hit by diagonal gates, so the RCCX phases cancel.
"""

import bisect
import functools
import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CCXGate,
    CPhaseGate,
    CSwapGate,
    CXGate,
    CZGate,
    HGate,
    MCXGate,
    PhaseGate,
    RCCXGate,
    RYGate,
    SwapGate,
    UCRYGate,
    XGate,
    ZGate,
)
from qiskit.quantum_info import Operator
from qiskit.synthesis import synth_mcx_1_clean_kg24, synth_mcx_n_clean_m15

from .passes import commute

NODE_COST = 6          # RCCX compute + RCCX uncompute
FLIP_COST = 2          # CX + at most one CZ phase correction


# ---------------------------------------------------------------------------
# B1 -- run ordering
# ---------------------------------------------------------------------------

def _tour(values):
    """Order of ``values`` (bit tuples) minimising total Hamming distance (path)."""
    n = len(values)
    if n <= 2:
        return list(range(n))
    d = [[sum(a != b for a, b in zip(values[i], values[j])) for j in range(n)] for i in range(n)]
    if n > 12:                                   # nearest neighbour from the first group
        order, left = [0], set(range(1, n))
        while left:
            nxt = min(left, key=lambda j: (d[order[-1]][j], j))
            order.append(nxt)
            left.remove(nxt)
        return order
    best = {(1 << j, j): (0, None) for j in range(n)}
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = sum(1 << j for j in subset)
            for j in subset:
                prev = mask ^ (1 << j)
                best[(mask, j)] = min(((best[(prev, i)][0] + d[i][j], i)
                                       for i in subset if i != j))
    full = (1 << n) - 1
    j = min(range(n), key=lambda j: (best[(full, j)][0], j))
    order, mask = [], full
    while j is not None:
        order.append(j)
        mask, j = mask ^ (1 << j), best[(mask, j)][1]
    return order[::-1]


def order_runs(gates, min_selector=2):
    out, i, moved = [], 0, 0
    n = len(gates)
    while i < n:
        if gates[i].kind == "raw":
            out.append(gates[i])
            i += 1
            continue
        # grow the segment while the common unwritten control set stays >= min_selector
        common = set(gates[i].ctrl_qubits)
        written = set(gates[i].writes)
        j = i
        while j + 1 < n and gates[j + 1].kind != "raw":
            nxt = gates[j + 1]
            c2, w2 = common & set(nxt.ctrl_qubits), written | set(nxt.writes)
            if len(c2 - w2) < min_selector:
                break
            common, written, j = c2, w2, j + 1
        seg = gates[i:j + 1]
        q = sorted(common - written)
        if len(seg) < 2 or len(q) < min_selector:
            out.append(gates[i])
            i += 1
            continue
        groups, keys = {}, []
        for g in seg:
            c = dict(g.cube)
            key = tuple(c[x] for x in q)
            if key not in groups:
                groups[key] = []
                keys.append(key)
            groups[key].append(g)
        # every pair of gates from different groups must commute; if the (sufficient)
        # commutation test cannot show it, the segment is left as it was
        safe = all(commute(ga, gb) for a, b in itertools.combinations(keys, 2)
                   for ga in groups[a] for gb in groups[b])
        if safe and len(keys) > 1:
            new = [g for t in _tour(keys) for g in groups[keys[t]]]
            moved += sum(1 for x, y in zip(new, seg) if x is not y)
            out.extend(new)
        else:
            out.extend(seg)
        i = j + 1
    return out, moved


# ---------------------------------------------------------------------------
# cost model (CX) for a gate with m control literals
# ---------------------------------------------------------------------------

def _cx_mcx(m):
    return (0, 1, 6)[m] if m < 3 else 6 * m - 6


def gate_cost(g, m):
    k = g.kind
    if k in ("x", "z"):
        return _cx_mcx(m)
    if k == "phase":
        if m <= 1:
            return 0
        if abs(math.remainder(g.param - math.pi, 2 * math.pi)) < 1e-12:
            return _cx_mcx(m - 1)
        return 2 if m == 2 else 6 * m
    if k == "swap":
        return 3 if m == 0 else (7 if m == 1 else 2 + _cx_mcx(m + 1))
    if k in ("ry", "h", "gate1"):
        return 0 if m == 0 else (2 if m == 1 else 6 * (m - 1) + 2)
    if k == "ucry":
        s = len(g.selects)
        return 2 ** s if m == 0 else (2 ** (s + 1) if m == 1 else 2 ** (s + 1) + 6 * (m - 1))
    return 10 ** 6


# ---------------------------------------------------------------------------
# sibling-flip phase correction
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _flip_correction(pa, p_old):
    """Phases theta[(a_bit, l_bit)] such that, on every state the ancilla can hold,
    RCCX_new . RCCX_old == diag(theta) . CX(a -> anc), for parent literal (a, pa) and node
    literal (l, p_old) -> (l, 1-p_old)."""
    def rccx_lits(qc, pl):
        flips = [q for q, p in ((0, pa), (1, pl)) if p == 0]
        for q in flips:
            qc.x(q)
        qc.rccx(0, 1, 2)
        for q in flips:
            qc.x(q)

    e = QuantumCircuit(3)
    rccx_lits(e, p_old)
    rccx_lits(e, 1 - p_old)
    u = Operator(e).data
    theta = {}
    for a_bit in (0, 1):
        for l_bit in (0, 1):
            old = int(a_bit == pa and l_bit == p_old)
            new = int(a_bit == pa and l_bit != p_old)
            amp = u[a_bit | l_bit << 1 | new << 2, a_bit | l_bit << 1 | old << 2]
            if abs(abs(amp) - 1) > 1e-9:
                raise RuntimeError("sibling-flip identity does not hold for this Qiskit's RCCX")
            theta[(a_bit, l_bit)] = float(np.angle(amp))
    return tuple(sorted(theta.items()))


# ---------------------------------------------------------------------------
# emitter: ops as (instruction, qubit indices, clbit indices)
# ---------------------------------------------------------------------------

class _Emitter:
    def __init__(self, n, max_anc, reserve=0):
        self.n = n
        self.max_anc = max_anc
        self.reserve = reserve if max_anc is not None else 0
        self.ops = []
        self.global_phase = 0.0
        self.busy = set()
        self.used = 0

    def free_ancillas(self):
        cap = self.max_anc if self.max_anc is not None else self.used + 64
        return [self.n + i for i in range(cap) if self.n + i not in self.busy]

    def take(self):
        free = self.free_ancillas()
        if len(free) <= self.reserve:
            return None
        a = free[0]
        self.busy.add(a)
        self.used = max(self.used, a - self.n + 1)
        return a

    def give(self, a):
        self.busy.discard(a)

    def note_temp(self, ancs):
        if ancs:
            self.used = max(self.used, max(ancs) - self.n + 1)

    def op(self, inst, qubits, clbits=()):
        self.ops.append((inst, list(qubits), list(clbits)))

    def _xconj(self, lits):
        for q, p in lits:
            if p == 0:
                self.op(XGate(), [q])

    def rccx(self, a, b, anc):
        self._xconj([a, b])
        self.op(RCCXGate(), [a[0], b[0], anc])
        self._xconj([a, b])

    def flip(self, a, lit_old, anc):
        (qa, pa), (ql, pl) = a, lit_old
        theta = dict(_flip_correction(pa, pl))
        self.op(CXGate(ctrl_state=pa), [qa, anc])
        t00 = theta[(0, 0)]
        alpha = theta[(1, 0)] - t00
        beta = theta[(0, 1)] - t00
        gamma = theta[(1, 1)] - theta[(1, 0)] - theta[(0, 1)] + t00
        self.global_phase += t00
        for q, ang in ((qa, alpha), (ql, beta)):
            if abs(math.remainder(ang, 2 * math.pi)) > 1e-12:
                self.op(PhaseGate(ang), [q])
        g = math.remainder(gamma, 2 * math.pi)
        if abs(g) < 1e-12:
            return
        if abs(abs(g) - math.pi) < 1e-12:
            self.op(CZGate(), [qa, ql])
        else:
            self.op(CPhaseGate(g), [qa, ql])

    def x(self, ctrls, t):
        m = len(ctrls)
        if m == 0:
            self.op(XGate(), [t])
        elif m == 1:
            self.op(CXGate(ctrl_state=ctrls[0][1]), [ctrls[0][0], t])
        elif m == 2:
            self.op(CCXGate(ctrl_state=ctrls[0][1] | ctrls[1][1] << 1),
                    [ctrls[0][0], ctrls[1][0], t])
        else:
            free = self.free_ancillas()
            qs = [q for q, _ in ctrls]
            self._xconj(ctrls)
            if len(free) >= m - 2:
                anc = free[: m - 2]
                self.note_temp(anc)
                self.op(synth_mcx_n_clean_m15(m).to_gate(label=f"mcx{m}_clean"), qs + [t] + anc)
            elif free:
                self.note_temp(free[:1])
                self.op(synth_mcx_1_clean_kg24(m).to_gate(label=f"mcx{m}_kg24"), qs + [t] + free[:1])
            else:
                self.op(MCXGate(m), qs + [t])
            self._xconj(ctrls)

    def z(self, ctrls, t):
        m = len(ctrls)
        if m == 0:
            self.op(ZGate(), [t])
        elif m == 1:
            self.op(CZGate(ctrl_state=ctrls[0][1]), [ctrls[0][0], t])
        else:
            self.op(HGate(), [t])
            self.x(ctrls, t)
            self.op(HGate(), [t])

    def phase(self, ctrls, phi):
        m = len(ctrls)
        if m == 0:
            self.global_phase += phi
            return
        (q, p), rest = ctrls[-1], ctrls[:-1]
        self._xconj([(q, p)])
        if abs(math.remainder(phi - math.pi, 2 * math.pi)) < 1e-12:
            self.z(rest, q)
        elif m == 1:
            self.op(PhaseGate(phi), [q])
        elif m == 2:
            self.op(CPhaseGate(phi, ctrl_state=rest[0][1]), [rest[0][0], q])
        else:
            st = sum(pp << i for i, (_, pp) in enumerate(rest))
            self.op(PhaseGate(phi).control(m - 1, ctrl_state=st), [r for r, _ in rest] + [q])
        self._xconj([(q, p)])

    def swap(self, ctrls, a, b):
        m = len(ctrls)
        if m == 0:
            self.op(SwapGate(), [a, b])
        elif m == 1:
            self.op(CSwapGate(ctrl_state=ctrls[0][1]), [ctrls[0][0], a, b])
        else:
            self.op(CXGate(), [b, a])
            self.x(list(ctrls) + [(a, 1)], b)
            self.op(CXGate(), [b, a])

    def single(self, base, ctrls, t):
        m = len(ctrls)
        if m == 0:
            self.op(base, [t])
        else:
            st = sum(p << i for i, (_, p) in enumerate(ctrls))
            self.op(base.control(m, ctrl_state=st), [q for q, _ in ctrls] + [t])

    def ucry(self, ctrls, table, selects, t):
        if len(ctrls) == 1:                    # fold the one control in as an extra select
            q, p = ctrls[0]
            s = len(selects)
            table = [table[i & ((1 << s) - 1)] if ((i >> s) & 1) == p else 0.0
                     for i in range(2 ** (s + 1))]
            selects, ctrls = list(selects) + [q], []
        gate = UCRYGate(list(table))
        if ctrls:
            st = sum(p << i for i, (_, p) in enumerate(ctrls))
            gate = gate.control(len(ctrls), ctrl_state=st)
        self.op(gate, [q for q, _ in ctrls] + [t] + list(selects))


# ---------------------------------------------------------------------------
# AND-cache
# ---------------------------------------------------------------------------

class _Cache:
    """AND nodes: key frozenset(literals) -> (ancilla, parent_key, literal)."""

    def __init__(self, em, flips=True):
        self.em = em
        self.flips = flips
        self.nodes = {}
        self.stats = {"nodes_built": 0, "node_reuses": 0, "sibling_flips": 0}

    def control_of(self, key):
        if not key:
            return None
        if len(key) == 1:
            return next(iter(key))
        return (self.nodes[key][0], 1)

    def children(self, key):
        return [k for k, v in self.nodes.items() if v[1] == key]

    def _parent_control(self, parent):
        return next(iter(parent)) if len(parent) == 1 else (self.nodes[parent][0], 1)

    def build(self, parent, lit):
        anc = self.em.take()
        if anc is None:
            return None
        self.em.rccx(self._parent_control(parent), lit, anc)
        key = parent | {lit}
        self.nodes[key] = (anc, parent, lit)
        self.stats["nodes_built"] += 1
        return key

    def sibling(self, key):
        _, parent, (q, p) = self.nodes[key]
        return parent | {(q, 1 - p)}

    def flippable(self, parent, lit):
        """Cached leaf node parent+{not lit} that could be flipped into parent+{lit}.

        The node must have been *built* as parent + (not lit): a node with the same literal
        set but a different construction order holds a different last literal, and flipping
        it would toggle the wrong one."""
        sib = parent | {(lit[0], 1 - lit[1])}
        if self.flips and parent and sib in self.nodes and not self.children(sib):
            _, built_parent, built_lit = self.nodes[sib]
            if built_parent == parent and built_lit == (lit[0], 1 - lit[1]):
                return sib
        return None

    def flip(self, sib, lit):
        anc, parent, old = self.nodes.pop(sib)
        self.em.flip(self._parent_control(parent), old, anc)
        key = parent | {lit}
        self.nodes[key] = (anc, parent, lit)
        self.stats["sibling_flips"] += 1
        return key

    def drop(self, key):
        for c in self.children(key):
            self.drop(c)
        anc, parent, lit = self.nodes.pop(key)
        self.em.rccx(self._parent_control(parent), lit, anc)      # RCCX is self-inverse
        self.em.give(anc)

    def invalidate(self, qubits):
        for key in list(self.nodes):
            if key in self.nodes and {q for q, _ in key} & qubits:
                self.drop(key)


class _Lookahead:
    """Which gates from i on could use AND node ``key`` (stopping at the first write)."""

    def __init__(self, gates):
        self.n = len(gates)
        self.by_lit, self.by_write = {}, {}
        for j, g in enumerate(gates):
            for lit in g.cube:
                self.by_lit.setdefault(lit, []).append(j)
            w = g.support if g.kind == "raw" else g.writes
            for q in set(w):
                self.by_write.setdefault(q, []).append(j)
        self.memo = {}

    def uses(self, i, key):
        mk = (i, key)
        if mk in self.memo:
            return self.memo[mk]
        stop = self.n
        for q, _ in key:
            w = self.by_write.get(q, [])
            pos = bisect.bisect_left(w, i)
            if pos < len(w):
                stop = min(stop, w[pos])
        lists = sorted((self.by_lit.get(lit, []) for lit in key), key=len)
        if not lists:
            out = []
        else:
            first = lists[0]
            lo, hi = bisect.bisect_left(first, i), bisect.bisect_left(first, stop)
            rest = [set(l) for l in lists[1:]]
            out = [j for j in first[lo:hi] if all(j in r for r in rest)]
        self.memo[mk] = out
        return out


def _residual_m(g, key):
    return len(set(g.cube) - key) + (1 if key else 0)


def lower_ctc(gates, n, max_anc=None, reserve=0, flips=True):
    """AND-cache lowering.  Returns (ops, ancillas_used, extra_global_phase, stats)."""
    em = _Emitter(n, max_anc, reserve)
    cache = _Cache(em, flips)
    look = _Lookahead(gates)

    for i, g in enumerate(gates):
        # 1. nodes this gate would corrupt are uncomputed first
        cache.invalidate(set(g.support) if g.kind == "raw" else set(g.writes))
        if g.kind == "raw":
            em.op(g.op, list(g.targets), list(g.clbits))
            continue

        cube = set(g.cube)
        rank = {q: r for r, q in enumerate(g.order)}
        # 2. deepest cached node inside the cube (singletons are free)
        cands = [k for k in cache.nodes if k <= cube]
        key = max(cands, key=lambda k: (len(k), -min(q for q, _ in k))) if cands else frozenset()
        if key:
            cache.stats["node_reuses"] += 1
        elif cube:
            lit = max(cube, key=lambda l: (len(look.uses(i, frozenset({l}))), -rank.get(l[0], 10 ** 6)))
            key = frozenset({lit})

        # 3. extend while the saving on this and later gates pays for the extension
        while True:
            residual = cube - key
            if not residual or not key:
                break
            best = None
            for lit in residual:
                nk = key | {lit}
                gain = 0
                for j in look.uses(i, nk):
                    h = gates[j]
                    gain += gate_cost(h, _residual_m(h, key)) - gate_cost(h, _residual_m(h, nk))
                if nk in cache.nodes:
                    cost = 0
                elif cache.flippable(key, lit):
                    cost = FLIP_COST
                else:
                    cost = NODE_COST
                cand = (gain - cost, -rank.get(lit[0], 10 ** 6), lit)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            net, _, lit = best
            if net < 0:
                break
            nk = key | {lit}
            if nk in cache.nodes:
                key = nk
                continue
            if len(key) >= 2 and key not in cache.nodes:
                break
            sib = cache.flippable(key, lit)
            if sib is not None:
                key = cache.flip(sib, lit)
                continue
            built = cache.build(key, lit)
            if built is None:
                leaves = [k for k in cache.nodes if not cache.children(k) and not k <= key]
                if not leaves:
                    break
                cache.drop(max(leaves, key=lambda k: (look.uses(i, k) or [10 ** 9])[0]))
                built = cache.build(key, lit)
                if built is None:
                    break
            key = built

        ctrl = cache.control_of(key)
        ctrls = ([ctrl] if ctrl else []) + sorted(cube - key)

        # 4. emit the core op under (node, residual literals)
        if g.kind == "x":
            em.x(ctrls, g.targets[0])
        elif g.kind == "z":
            em.z(ctrls, g.targets[0])
        elif g.kind == "phase":
            em.phase(ctrls, g.param)
        elif g.kind == "swap":
            em.swap(ctrls, *g.targets)
        elif g.kind == "ry":
            em.single(RYGate(g.param), ctrls, g.targets[0])
        elif g.kind == "h":
            em.single(HGate(), ctrls, g.targets[0])
        elif g.kind == "gate1":
            em.single(g.op, ctrls, g.targets[0])
        elif g.kind == "ucry":
            em.ucry(ctrls, g.param, g.selects, g.targets[0])
        else:
            raise ValueError(g.kind)

        # 5. uncompute nodes nothing later can use (keep a leaf whose sibling is wanted)
        changed = True
        while changed:
            changed = False
            for k in list(cache.nodes):
                if k in cache.nodes and not cache.children(k) and not look.uses(i + 1, k):
                    sib = cache.sibling(k)
                    if flips and len(k) >= 2 and sib not in cache.nodes and look.uses(i + 1, sib):
                        continue
                    cache.drop(k)
                    changed = True

    for k in sorted(cache.nodes, key=len, reverse=True):
        if k in cache.nodes:
            cache.drop(k)
    if em.busy:
        raise RuntimeError("internal error: ancillas left busy")
    return em.ops, em.used, em.global_phase, dict(cache.stats)
