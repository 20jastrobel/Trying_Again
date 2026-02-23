from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian
from pydephasing.quantum.pauli_polynomial_class import PauliPolynomial


def line_lattice_periodic_edges(length: int) -> List[Tuple[int, int]]:
    """
    Undirected nearest-neighbor edges for a 1D periodic line lattice.
    This matches LineLattice(..., boundary_condition=PERIODIC) bond connectivity
    while deduplicating reverse directions.
    """
    if length <= 1:
        return []

    edges = set()
    for i in range(length):
        j = (i + 1) % length
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        edges.add((a, b))
    return sorted(edges)


def _to_upper_pauli_string(raw: str) -> str:
    trans = {"e": "I", "x": "X", "y": "Y", "z": "Z"}
    return "".join(trans[ch] for ch in raw)


def pauli_polynomial_to_canonical_dict(
    polynomial: PauliPolynomial, tol: float = 1e-12
) -> Dict[str, float]:
    """
    Convert PauliPolynomial to canonical, sorted dictionary:
      {pauli_string: real_coefficient}
    Coefficients with |c| < tol are removed, and tiny imaginary residuals are dropped.
    Includes identity term explicitly.
    """
    accum: Dict[str, complex] = {}
    terms = polynomial.return_polynomial()
    nq = terms[0].nqubit() if terms else 0

    for term in terms:
        key = _to_upper_pauli_string(term.pw2strng())
        accum[key] = accum.get(key, 0.0 + 0.0j) + complex(term.p_coeff)

    identity_key = "I" * nq
    accum.setdefault(identity_key, 0.0 + 0.0j)

    out: Dict[str, float] = {}
    for key in sorted(accum.keys()):
        coeff = accum[key]
        if abs(coeff) < tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(
                f"Non-negligible imaginary coefficient for term {key}: {coeff}"
            )
        out[key] = float(coeff.real)

    if identity_key not in out:
        out[identity_key] = 0.0
    return dict(sorted(out.items()))


def half_filling_sector(L: int) -> Dict[str, object]:
    return {
        "filling": "half",
        "constraint": "N_alpha + N_beta = L",
        "total_particles": L,
        "balanced_spin_sectors_for_comparison": [
            [n_alpha, L - n_alpha]
            for n_alpha in range(L + 1)
            if abs((L - n_alpha) - n_alpha) <= 1
        ],
    }


def _spin_orbital_order(L: int, indexing: str) -> List[str]:
    if indexing == "blocked":
        return [f"alpha{i}" for i in range(L)] + [f"beta{i}" for i in range(L)]
    if indexing == "interleaved":
        out: List[str] = []
        for i in range(L):
            out.extend([f"alpha{i}", f"beta{i}"])
        return out
    raise ValueError(f"Unsupported indexing: {indexing}")


def build_reference_entry(
    L: int, t: float, U: float, tol: float, indexing: str
) -> Dict[str, object]:
    edges = line_lattice_periodic_edges(L)
    H = build_hubbard_hamiltonian(
        dims=L,
        t=t,
        U=U,
        indexing=indexing,
        edges=edges,
        pbc=True,
    )
    pauli_terms = pauli_polynomial_to_canonical_dict(H, tol=tol)
    identity_key = "I" * (2 * L)

    spin_orbital_order = _spin_orbital_order(L, indexing=indexing)
    metadata = {
        "L": L,
        "t": t,
        "U": U,
        "boundary_condition": "periodic",
        "edge_set": [list(edge) for edge in edges],
        "num_qubits": 2 * L,
        "num_pauli_terms": len(pauli_terms),
        "particle_sector": half_filling_sector(L),
        "constant_shift_identity": pauli_terms.get(identity_key, 0.0),
        "coefficient_tolerance": tol,
        "indexing": {
            "name": indexing,
            "spin_orbital_order_before_jw": spin_orbital_order,
            "pauli_string_qubit_order": "left_to_right = q_(n-1) ... q_0",
        },
    }

    return {
        "metadata": metadata,
        "pauli_terms": pauli_terms,
    }


def build_reference_data(
    t: float = 1.0,
    U: float = 4.0,
    tol: float = 1e-12,
    indexing: str = "blocked",
    lattice_sizes: Tuple[int, ...] = (2, 3),
) -> Dict[str, object]:
    if not lattice_sizes:
        raise ValueError("lattice_sizes must be non-empty")
    for L in lattice_sizes:
        if int(L) <= 0:
            raise ValueError("all lattice sizes must be positive")

    return {
        "description": "1D Fermi-Hubbard JW Hamiltonians",
        "parameters": {
            "t": t,
            "U": U,
            "boundary_condition": "periodic",
            "indexing": indexing,
            "lattice_sizes": [int(L) for L in lattice_sizes],
            "coefficient_tolerance": tol,
        },
        "cases": {
            f"L={L}": build_reference_entry(
                L=L, t=t, U=U, tol=tol, indexing=indexing
            )
            for L in lattice_sizes
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export periodic 1D Hubbard JW Pauli dictionaries."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "exports"
        / "hubbard_jw_L2_L3_periodic_blocked.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--indexing",
        choices=["interleaved", "blocked"],
        default="blocked",
        help="Spin-orbital ordering before JW mapping.",
    )
    parser.add_argument("--t", type=float, default=1.0, help="Hopping parameter.")
    parser.add_argument("--U", type=float, default=4.0, help="On-site interaction.")
    parser.add_argument(
        "--tol", type=float, default=1e-12, help="Coefficient cutoff tolerance."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Lattice sizes L to export (e.g. --sizes 4 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_reference_data(
        t=args.t,
        U=args.U,
        tol=args.tol,
        indexing=args.indexing,
        lattice_sizes=tuple(args.sizes),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    for key, case in data["cases"].items():
        metadata = case["metadata"]
        print(
            f"{key}: edges={metadata['edge_set']}, "
            f"num_qubits={metadata['num_qubits']}, "
            f"num_pauli_terms={metadata['num_pauli_terms']}, "
            f"constant_shift={metadata['constant_shift_identity']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
