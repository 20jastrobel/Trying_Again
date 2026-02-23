"""Fermi-Hubbard ground-state search with Qiskit built-ins.

This script:
1. Builds 1D Fermi-Hubbard models for L=2 and L=3.
2. Uses the built-in Jordan-Wigner mapper.
3. Uses built-in VQE with built-in UCCSD ansatz.
4. Reports the estimated ground-state energies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Silence matplotlib cache warnings triggered by qiskit_nature imports on some systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

DEFAULT_SPIN_ORBITAL_ORDERING = "blocked"


@dataclass
class HubbardVQEResult:
    """Container for one lattice-size ground-state calculation."""

    lattice_sites: int
    num_qubits: int
    num_particles: tuple[int, int]
    vqe_energy: float
    exact_filtered_energy: float
    best_restart: int
    spin_orbital_ordering: str
    uccsd_reps: int
    uccsd_num_parameters: int


def _interleaved_to_blocked_permutation(num_sites: int) -> list[int]:
    """Map site-interleaved spin ordering (a1,b1,a2,b2,...) to blocked ordering."""

    return [index for site in range(num_sites) for index in (site, num_sites + site)]


def _apply_spin_orbital_ordering(fermionic_op, num_sites: int, ordering: str):
    """Return a fermionic operator in the requested spin-orbital ordering."""

    normalized_ordering = ordering.strip().lower()
    if normalized_ordering == "blocked":
        return fermionic_op.permute_indices(_interleaved_to_blocked_permutation(num_sites))
    if normalized_ordering == "interleaved":
        return fermionic_op

    raise ValueError(
        f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'."
    )


def _half_filled_num_particles(num_sites: int) -> tuple[int, int]:
    """Half-filling particle tuple (n_alpha, n_beta)."""

    return ((num_sites + 1) // 2, num_sites // 2)


def _build_qubit_hamiltonian(
    num_sites: int,
    hopping_t: float,
    onsite_u: float,
    mapper: JordanWignerMapper,
    spin_orbital_ordering: str = DEFAULT_SPIN_ORBITAL_ORDERING,
):
    """Build and map the Fermi-Hubbard Hamiltonian to qubits."""

    lattice = LineLattice(
        num_nodes=num_sites,
        edge_parameter=-hopping_t,
        onsite_parameter=0.0,
        boundary_condition=BoundaryCondition.PERIODIC,
    )
    fermionic_op = FermiHubbardModel(lattice=lattice, onsite_interaction=onsite_u).second_q_op()
    fermionic_op = _apply_spin_orbital_ordering(fermionic_op, num_sites, spin_orbital_ordering)
    return mapper.map(fermionic_op)


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    """Return alpha/beta index sets for the selected spin-orbital ordering."""

    normalized_ordering = ordering.strip().lower()
    if normalized_ordering == "blocked":
        return list(range(num_sites)), list(range(num_sites, 2 * num_sites))
    if normalized_ordering == "interleaved":
        return list(range(0, 2 * num_sites, 2)), list(range(1, 2 * num_sites, 2))

    raise ValueError(
        f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'."
    )


def _build_number_aux_ops(
    num_sites: int,
    mapper: JordanWignerMapper,
    spin_orbital_ordering: str = DEFAULT_SPIN_ORBITAL_ORDERING,
) -> dict[str, object]:
    """Build alpha/beta particle-number operators for filtered exact diagonalization."""

    n_spin_orbitals = 2 * num_sites
    alpha_indices, beta_indices = _spin_orbital_index_sets(num_sites, spin_orbital_ordering)
    alpha_number = FermionicOp(
        {f"+_{i} -_{i}": 1.0 for i in alpha_indices},
        num_spin_orbitals=n_spin_orbitals,
    )
    beta_number = FermionicOp(
        {f"+_{i} -_{i}": 1.0 for i in beta_indices},
        num_spin_orbitals=n_spin_orbitals,
    )
    return {"N_alpha": mapper.map(alpha_number), "N_beta": mapper.map(beta_number)}


def _filtered_exact_energy(
    qubit_hamiltonian,
    num_particles: tuple[int, int],
    aux_operators: dict[str, object],
) -> float:
    """Compute exact minimum energy in the requested particle-number sector."""

    def filter_criterion(_state, _energy, aux_values):
        n_alpha = float(np.real(aux_values["N_alpha"][0]))
        n_beta = float(np.real(aux_values["N_beta"][0]))
        return np.isclose(n_alpha, num_particles[0]) and np.isclose(n_beta, num_particles[1])

    exact_solver = NumPyMinimumEigensolver(filter_criterion=filter_criterion)
    exact_result = exact_solver.compute_minimum_eigenvalue(
        qubit_hamiltonian,
        aux_operators=aux_operators,
    )
    return float(np.real(exact_result.eigenvalue))


def run_hubbard_vqe(
    num_sites: int,
    hopping_t: float = 1.0,
    onsite_u: float = 4.0,
    spin_orbital_ordering: str = DEFAULT_SPIN_ORBITAL_ORDERING,
    uccsd_reps: int = 2,
    vqe_restarts: int = 3,
    maxiter: int = 1800,
    seed: int = 7,
) -> HubbardVQEResult:
    """Solve one lattice size with JW + VQE(UCCSD)."""

    normalized_ordering = spin_orbital_ordering.strip().lower()
    if normalized_ordering not in {"blocked", "interleaved"}:
        raise ValueError(
            f"Unsupported spin_orbital_ordering='{spin_orbital_ordering}'. Use 'blocked' or 'interleaved'."
        )

    mapper = JordanWignerMapper()
    qubit_hamiltonian = _build_qubit_hamiltonian(
        num_sites,
        hopping_t,
        onsite_u,
        mapper,
        spin_orbital_ordering=normalized_ordering,
    )
    num_particles = _half_filled_num_particles(num_sites)

    hartree_fock = HartreeFock(
        num_spatial_orbitals=num_sites,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )
    ansatz = UCCSD(
        num_spatial_orbitals=num_sites,
        num_particles=num_particles,
        qubit_mapper=mapper,
        reps=uccsd_reps,
        initial_state=hartree_fock,
    )

    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    best_restart = -1
    estimator = StatevectorEstimator()

    for restart in range(vqe_restarts):
        initial_point = 0.3 * rng.normal(size=ansatz.num_parameters)
        optimizer = SLSQP(maxiter=maxiter)
        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
        vqe_result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
        energy = float(np.real(vqe_result.eigenvalue))
        if energy < best_energy:
            best_energy = energy
            best_restart = restart

    aux_ops = _build_number_aux_ops(
        num_sites,
        mapper,
        spin_orbital_ordering=normalized_ordering,
    )
    exact_filtered_energy = _filtered_exact_energy(qubit_hamiltonian, num_particles, aux_ops)

    return HubbardVQEResult(
        lattice_sites=num_sites,
        num_qubits=qubit_hamiltonian.num_qubits,
        num_particles=num_particles,
        vqe_energy=best_energy,
        exact_filtered_energy=exact_filtered_energy,
        best_restart=best_restart,
        spin_orbital_ordering=normalized_ordering,
        uccsd_reps=uccsd_reps,
        uccsd_num_parameters=ansatz.num_parameters,
    )


def main() -> None:
    """Run L=2 and L=3 calculations and print a concise summary."""

    print("Fermi-Hubbard Ground State (Jordan-Wigner + VQE/UCCSD)", flush=True)
    print(
        "Parameters: t=1.0, U=4.0, periodic boundary, half-filling, "
        f"spin-orbital ordering={DEFAULT_SPIN_ORBITAL_ORDERING}",
        flush=True,
    )
    print("-" * 74, flush=True)

    for lattice_size in (2, 3):
        result = run_hubbard_vqe(
            num_sites=lattice_size,
            spin_orbital_ordering=DEFAULT_SPIN_ORBITAL_ORDERING,
        )
        delta = result.vqe_energy - result.exact_filtered_energy
        print(
            f"L={result.lattice_sites} | qubits={result.num_qubits} | "
            f"particles={result.num_particles} | "
            f"VQE={result.vqe_energy:.12f} | "
            f"filtered_exact={result.exact_filtered_energy:.12f} | "
            f"delta={delta:+.3e} | "
            f"best_restart={result.best_restart} | "
            f"ordering={result.spin_orbital_ordering} | "
            f"uccsd(reps={result.uccsd_reps}, params={result.uccsd_num_parameters})",
            flush=True,
        )


if __name__ == "__main__":
    main()
