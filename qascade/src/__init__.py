"""ctc_optimizer -- Control-Tree Compilation for Qiskit circuits.

    from ctc_optimizer import optimize
    optimized = optimize(my_circuit)

Reduces two-qubit gate count by treating multi-controlled gates as guarded operations:

Stage A (exact rewrites on the circuit, before synthesis)
    A1 cancel     cancel identical guarded gates across commuting neighbours
    A2 pauli      collapse X/Z products under the same controls (ZXZX -> a phase)
    A3 merge      merge the same op under controls differing in one polarity
    A4 multiplex  turn same-target controlled Ry gates into a uniformly controlled Ry

Stage B (synthesis with shared control on clean ancillas)
    B1 order      regroup commuting gates that share a selector
    B2 and_cache  compute shared control ANDs once on ancillas (relative-phase Toffolis)
    B3 flip       turn a cached AND into its sibling with one CX instead of rebuilding
"""

from .api import (
    ALL_STEPS,
    DEFAULT_BASIS,
    STAGE_A_STEPS,
    STAGE_B_STEPS,
    STEP_ALIASES,
    OptimizationReport,
    ancilla_tradeoff,
    optimize,
    transpile_exact,
)
from .verify import NotVerifiable, verify

__version__ = "0.1.0"

__all__ = [
    "optimize",
    "verify",
    "ancilla_tradeoff",
    "transpile_exact",
    "OptimizationReport",
    "NotVerifiable",
    "ALL_STEPS",
    "STAGE_A_STEPS",
    "STAGE_B_STEPS",
    "STEP_ALIASES",
    "DEFAULT_BASIS",
    "__version__",
]
