"""Symmetry operators and sector utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.properties import Magnetization, ParticleNumber

try:
    from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
except Exception:  # pragma: no cover
    from qiskit_algorithms import NumPyMinimumEigensolver

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.quantum.hartree_fock_reference_state import mode_index as canonical_mode_index


def _extract_second_q_op(ops, name: str) -> FermionicOp:
    if isinstance(ops, dict):
        if name in ops:
            return ops[name]
        if len(ops) == 1:
            return next(iter(ops.values()))
    raise ValueError(f"Could not extract second-quantized operator '{name}'.")


def fermionic_number_op(n_sites: int) -> FermionicOp:
    ops = ParticleNumber(int(n_sites)).second_q_ops()
    return _extract_second_q_op(ops, "ParticleNumber")


def fermionic_sz_op(n_sites: int) -> FermionicOp:
    ops = Magnetization(int(n_sites)).second_q_ops()
    return _extract_second_q_op(ops, "Magnetization")


def map_symmetry_ops_to_qubits(
    mapper: JordanWignerMapper,
    n_sites: int,
) -> tuple[SparsePauliOp, SparsePauliOp]:
    n_op = fermionic_number_op(n_sites)
    sz_op = fermionic_sz_op(n_sites)
    n_q = mapper.map(n_op).simplify(atol=1e-12)
    sz_q = mapper.map(sz_op).simplify(atol=1e-12)
    return n_q, sz_q


def computational_basis_eigs(n_sites: int, bitstring_int: int) -> tuple[int, float]:
    n_orb = 2 * int(n_sites)
    bits = [(bitstring_int >> idx) & 1 for idx in range(n_orb)]
    n_total = int(sum(bits))
    up_indices = [
        int(canonical_mode_index(i, 0, n_sites=int(n_sites), indexing="blocked"))
        for i in range(int(n_sites))
    ]
    dn_indices = [
        int(canonical_mode_index(i, 1, n_sites=int(n_sites), indexing="blocked"))
        for i in range(int(n_sites))
    ]
    n_up = int(sum(bits[q] for q in up_indices))
    n_dn = int(sum(bits[q] for q in dn_indices))
    sz = 0.5 * (n_up - n_dn)
    return n_total, sz


def sector_basis_indices(
    n_sites: int,
    n_target: int,
    sz_target: float,
) -> list[int]:
    """Return computational-basis indices that lie in the requested (N, Sz) sector.

    Indexing convention matches Qiskit's little-endian basis ordering, i.e. qubit
    index 0 is the least-significant bit.
    """
    dim = 2 ** (2 * int(n_sites))
    indices: list[int] = []
    for idx in range(dim):
        n_val, sz_val = computational_basis_eigs(int(n_sites), idx)
        if n_val == int(n_target) and abs(float(sz_val) - float(sz_target)) < 1e-12:
            indices.append(int(idx))
    return indices


def exact_ground_state_sector(
    qubit_op: SparsePauliOp,
    n_sites: int,
    n_target: int,
    sz_target: float,
) -> tuple[float, np.ndarray]:
    """Return (E0, |psi0>) for the exact ground state in the target (N, Sz) sector."""
    n_q, sz_q = map_symmetry_ops_to_qubits(JordanWignerMapper(), int(n_sites))

    n_t = int(n_target)
    sz_t = float(sz_target)

    def _filter(_state, _energy, aux_values) -> bool:
        n_val = float(np.real(aux_values["N"][0]))
        sz_val = float(np.real(aux_values["Sz"][0]))
        return bool(np.isclose(n_val, n_t, atol=1e-9) and np.isclose(sz_val, sz_t, atol=1e-9))

    solver = NumPyMinimumEigensolver(filter_criterion=_filter)
    result = solver.compute_minimum_eigenvalue(
        qubit_op,
        aux_operators={"N": n_q, "Sz": sz_q},
    )
    if result.eigenvalue is None or result.eigenstate is None:
        raise ValueError("No eigenstate found in target sector.")

    e0 = float(np.real(result.eigenvalue))
    state = result.eigenstate
    if hasattr(state, "data"):
        psi = np.asarray(state.data, dtype=complex)
    else:
        psi = np.asarray(state, dtype=complex).reshape(-1)

    psi_norm = float(np.linalg.norm(psi))
    if psi_norm <= 0.0:
        raise RuntimeError("Exact eigenvector has zero norm (unexpected).")
    psi = psi / psi_norm
    return e0, psi


def exact_ground_energy_sector(
    qubit_op: SparsePauliOp,
    n_sites: int,
    n_target: int,
    sz_target: float,
) -> float:
    e0, _psi = exact_ground_state_sector(qubit_op, n_sites, n_target, sz_target)
    return e0


def commutes(op: SparsePauliOp, sym: SparsePauliOp, tol: float = 1e-12) -> bool:
    comm = (op @ sym - sym @ op).simplify(atol=tol)
    if len(comm.coeffs) == 0:
        return True
    return float(np.max(np.abs(comm.coeffs))) < tol
