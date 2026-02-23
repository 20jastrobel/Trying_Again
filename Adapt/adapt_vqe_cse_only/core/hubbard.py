"""Hubbard builders using canonical repo indexing and Qiskit-Nature operators."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import bravais_nearest_neighbor_edges, mode_index


def default_1d_chain_edges(n_sites: int, *, periodic: bool = False) -> list[tuple[int, int]]:
    edges = bravais_nearest_neighbor_edges(int(n_sites), pbc=bool(periodic))
    return [(int(i), int(j)) for i, j in edges]


def _site_potentials(v, *, n_sites: int) -> list[float]:
    if v is None:
        return [0.0] * int(n_sites)
    if isinstance(v, (int, float)):
        return [float(v)] * int(n_sites)

    vals = list(v)
    if len(vals) != int(n_sites):
        raise ValueError("Site-potential sequence length must match n_sites.")
    return [float(x) for x in vals]


def _accumulate_term(terms: dict[str, complex], label: str, coeff: complex, *, tol: float = 1e-15) -> None:
    if abs(coeff) <= tol:
        return
    terms[label] = terms.get(label, 0.0 + 0.0j) + complex(coeff)
    if abs(terms[label]) <= tol:
        del terms[label]


def build_fermionic_hubbard(
    *,
    n_sites: int,
    t: float,
    u: float,
    edges: Sequence[tuple[int, int]] | None = None,
    v=None,
    indexing: str = "blocked",
) -> FermionicOp:
    n_sites_i = int(n_sites)
    if n_sites_i <= 0:
        raise ValueError("n_sites must be positive.")

    edge_list = list(edges) if edges is not None else default_1d_chain_edges(n_sites_i, periodic=False)
    site_v = _site_potentials(v, n_sites=n_sites_i)

    terms: dict[str, complex] = {}

    # Kinetic hopping term: -t * (c_i^dag c_j + h.c.) for each spin.
    for i, j in edge_list:
        i_i = int(i)
        j_i = int(j)
        for spin in (0, 1):
            p = int(mode_index(i_i, spin, indexing=indexing, n_sites=n_sites_i))
            q = int(mode_index(j_i, spin, indexing=indexing, n_sites=n_sites_i))
            _accumulate_term(terms, f"+_{p} -_{q}", -float(t))
            _accumulate_term(terms, f"+_{q} -_{p}", -float(t))

    # On-site interaction term: U * n_{i,up} n_{i,down}.
    for i in range(n_sites_i):
        p_up = int(mode_index(i, 0, indexing=indexing, n_sites=n_sites_i))
        p_dn = int(mode_index(i, 1, indexing=indexing, n_sites=n_sites_i))
        _accumulate_term(terms, f"+_{p_up} -_{p_up} +_{p_dn} -_{p_dn}", float(u))

    # Local potential term: -v_i * n_{i,spin}.
    for i, vi in enumerate(site_v):
        if abs(vi) <= 1e-15:
            continue
        for spin in (0, 1):
            p = int(mode_index(i, spin, indexing=indexing, n_sites=n_sites_i))
            _accumulate_term(terms, f"+_{p} -_{p}", -float(vi))

    return FermionicOp(terms, num_spin_orbitals=2 * n_sites_i)


def build_qubit_hamiltonian_from_fermionic(
    ferm_op: FermionicOp,
    mapper: JordanWignerMapper | None = None,
):
    mapper = mapper if mapper is not None else JordanWignerMapper()
    qubit_op = mapper.map(ferm_op).simplify(atol=1e-12)
    return qubit_op, mapper
