"""Command line:  python -m ctc_optimizer INPUT -o OUTPUT [options]

INPUT / OUTPUT: .qpy or .qasm (OpenQASM 2).
"""

import argparse
import sys

from qiskit import qasm2, qpy

from . import ALL_STEPS, optimize


def _load(path):
    if path.endswith(".qasm"):
        return qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    with open(path, "rb") as fh:
        circuits = qpy.load(fh)
    if len(circuits) != 1:
        sys.exit(f"{path}: expected one circuit, found {len(circuits)}")
    return circuits[0]


def _save(qc, path):
    if path.endswith(".qasm"):
        qasm2.dump(qc, path)
    else:
        with open(path, "wb") as fh:
            qpy.dump(qc, fh)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ctc_optimizer", description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--max-ancillas", type=int, default=None)
    ap.add_argument("--steps", nargs="+", default=list(ALL_STEPS),
                    help="subset of: " + " ".join(ALL_STEPS) + " (or A1..A4 B1..B3)")
    ap.add_argument("--basis", nargs="+", default=["cx", "rz", "sx", "x"],
                    help="basis gates, or 'none' to keep high-level gates")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--tket", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    basis = None if [b.lower() for b in args.basis] == ["none"] else args.basis
    qc = _load(args.input)
    out, report = optimize(qc, max_ancillas=args.max_ancillas, steps=args.steps,
                           basis_gates=basis, search=not args.no_search, tket=args.tket,
                           verify=args.verify, return_report=True)
    _save(out, args.output)
    print(report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
