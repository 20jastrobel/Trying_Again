"""Standalone Hubbard dimer real-time dynamics with Qiskit built-ins.

This script is intentionally self-contained:
- No imports from local project modules.
- No edits to existing files are required.

Model:
- Two-site Fermi-Hubbard (dimer), half-filling.
- Jordan-Wigner mapping to qubits.

Time evolution:
- Approximate: built-in Suzuki-Trotter (or Lie-Trotter for order=1).
- Reference: built-in MatrixExponential synthesis.

Initial state:
- HF reference Slater determinant, or
- Qiskit built-in VQE with built-in UCCSD ansatz.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

# Avoid matplotlib cache warnings triggered by qiskit_nature imports on some systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, MatrixExponential, SuzukiTrotter
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except Exception:
    pass


DEFAULT_NUM_SITES = 2  # Hubbard dimer
DEFAULT_ORDERING = "blocked"


@dataclass
class DynamicsPoint:
    """One time-point comparison between exact and Suzuki-Trotter evolution."""

    time: float
    fidelity: float
    state_l2_error: float
    energy_exact: float
    energy_trotter: float
    n_up_site0_exact: float
    n_up_site0_trotter: float
    n_dn_site0_exact: float
    n_dn_site0_trotter: float
    doublon_exact: float
    doublon_trotter: float


@dataclass
class UCCSDVQEState:
    """VQE(UCCSD) initial-state payload for dynamics."""

    circuit: QuantumCircuit
    vqe_energy: float
    best_restart: int
    num_particles: tuple[int, int]
    uccsd_reps: int
    uccsd_num_parameters: int
    optimal_point: list[float]
    fidelity_vs_hf: float


@dataclass
class DynamicsSummary:
    """Aggregate error and trajectory metrics for interpretation/reporting."""

    min_fidelity: float
    mean_fidelity: float
    final_fidelity: float
    max_l2_error: float
    final_l2_error: float
    max_abs_energy_error: float
    mean_abs_energy_error: float
    final_abs_energy_error: float
    max_abs_n_up_error: float
    max_abs_n_dn_error: float
    max_abs_doublon_error: float
    exact_n_up_span: float
    exact_n_dn_span: float
    exact_doublon_span: float


def _interleaved_to_blocked_permutation(num_sites: int) -> list[int]:
    """Map interleaved spin ordering (a1,b1,a2,b2,...) to blocked ordering (a1,a2,...,b1,b2,...)."""

    return [index for site in range(num_sites) for index in (site, num_sites + site)]


def _apply_spin_orbital_ordering(op, num_sites: int, ordering: str):
    """Return fermionic operator in the requested spin-orbital ordering."""

    normalized = ordering.strip().lower()
    if normalized == "interleaved":
        return op
    if normalized == "blocked":
        return op.permute_indices(_interleaved_to_blocked_permutation(num_sites))
    raise ValueError(f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'.")


def _build_qubit_hamiltonian(
    num_sites: int,
    hopping_t: float,
    onsite_u: float,
    boundary: str,
    ordering: str,
    mapper: JordanWignerMapper,
) -> SparsePauliOp:
    """Construct and JW-map the Fermi-Hubbard Hamiltonian."""

    boundary_condition = (
        BoundaryCondition.PERIODIC if boundary.strip().lower() == "periodic" else BoundaryCondition.OPEN
    )
    lattice = LineLattice(
        num_nodes=num_sites,
        edge_parameter=-hopping_t,
        onsite_parameter=0.0,
        boundary_condition=boundary_condition,
    )
    fermionic_op = FermiHubbardModel(lattice=lattice, onsite_interaction=onsite_u).second_q_op()
    fermionic_op = _apply_spin_orbital_ordering(fermionic_op, num_sites=num_sites, ordering=ordering)
    return mapper.map(fermionic_op).simplify(atol=1e-12)


def _half_filled_particle_numbers(num_sites: int) -> tuple[int, int]:
    """Half-filling particle tuple (n_alpha, n_beta)."""

    return ((num_sites + 1) // 2, num_sites // 2)


def _map_interleaved_index(index_interleaved: int, num_sites: int, ordering: str) -> int:
    """Convert interleaved spin-orbital index to target ordering index."""

    normalized = ordering.strip().lower()
    if normalized == "interleaved":
        return index_interleaved
    if normalized == "blocked":
        return _interleaved_to_blocked_permutation(num_sites)[index_interleaved]
    raise ValueError(f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'.")


def _build_initial_state_circuit(num_sites: int, ordering: str) -> QuantumCircuit:
    """Build a half-filled Slater determinant reference state circuit."""

    n_alpha, n_beta = _half_filled_particle_numbers(num_sites)
    occupied_interleaved: list[int] = []
    occupied_interleaved.extend(2 * site for site in range(n_alpha))  # alpha spin orbitals
    occupied_interleaved.extend(2 * site + 1 for site in range(n_beta))  # beta spin orbitals
    occupied = [_map_interleaved_index(i, num_sites=num_sites, ordering=ordering) for i in occupied_interleaved]

    num_qubits = 2 * num_sites
    qc = QuantumCircuit(num_qubits)
    for qubit in occupied:
        qc.x(qubit)
    return qc


def _build_vqe_uccsd_initial_state(
    num_sites: int,
    ordering: str,
    hamiltonian: SparsePauliOp,
    mapper: JordanWignerMapper,
    uccsd_reps: int,
    vqe_restarts: int,
    vqe_maxiter: int,
    vqe_seed: int,
) -> UCCSDVQEState:
    """Run Qiskit VQE(UCCSD) and return the optimized initial state circuit."""

    _ = ordering  # Ordering has already been applied to the Hamiltonian construction path.
    num_particles = _half_filled_particle_numbers(num_sites)
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

    estimator = StatevectorEstimator()
    rng = np.random.default_rng(vqe_seed)
    best_energy = float("inf")
    best_restart = -1
    best_point: np.ndarray | None = None

    for restart in range(vqe_restarts):
        initial_point = 0.3 * rng.normal(size=ansatz.num_parameters)
        optimizer = SLSQP(maxiter=vqe_maxiter)
        solver = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
        result = solver.compute_minimum_eigenvalue(hamiltonian)
        energy = float(np.real(result.eigenvalue))
        if energy >= best_energy:
            continue

        best_energy = energy
        best_restart = restart
        if getattr(result, "optimal_point", None) is not None:
            best_point = np.asarray(result.optimal_point, dtype=float)
        elif getattr(result, "optimal_parameters", None):
            best_point = np.asarray([float(result.optimal_parameters[param]) for param in ansatz.parameters], dtype=float)
        else:
            raise RuntimeError("VQE result did not include optimal parameters for UCCSD state construction.")

    if best_point is None:
        raise RuntimeError("VQE(UCCSD) failed to produce an optimal parameter vector.")

    vqe_circuit = ansatz.assign_parameters(best_point)
    hf_state = Statevector.from_instruction(hartree_fock)
    vqe_state = Statevector.from_instruction(vqe_circuit)
    fidelity_vs_hf = float(abs(np.vdot(vqe_state.data, hf_state.data)) ** 2)
    return UCCSDVQEState(
        circuit=vqe_circuit,
        vqe_energy=best_energy,
        best_restart=best_restart,
        num_particles=num_particles,
        uccsd_reps=uccsd_reps,
        uccsd_num_parameters=ansatz.num_parameters,
        optimal_point=[float(x) for x in best_point.tolist()],
        fidelity_vs_hf=fidelity_vs_hf,
    )


def _number_operator_qubit(
    num_sites: int,
    site: int,
    spin: str,
    ordering: str,
    mapper: JordanWignerMapper,
) -> SparsePauliOp:
    """Build mapped number operator n_{site,spin}."""

    if spin not in {"up", "dn"}:
        raise ValueError("spin must be 'up' or 'dn'")
    orbital_interleaved = 2 * site + (0 if spin == "up" else 1)
    op = FermionicOp(
        {f"+_{orbital_interleaved} -_{orbital_interleaved}": 1.0},
        num_spin_orbitals=2 * num_sites,
    )
    op = _apply_spin_orbital_ordering(op, num_sites=num_sites, ordering=ordering)
    return mapper.map(op).simplify(atol=1e-12)


def _doublon_operator_qubit(
    num_sites: int,
    ordering: str,
    mapper: JordanWignerMapper,
) -> SparsePauliOp:
    """Build total doublon operator sum_i n_{i,up} n_{i,dn}."""

    total = SparsePauliOp.from_list([("I" * (2 * num_sites), 0.0)])
    for site in range(num_sites):
        up_interleaved = 2 * site
        dn_interleaved = 2 * site + 1
        op = FermionicOp(
            {f"+_{up_interleaved} -_{up_interleaved} +_{dn_interleaved} -_{dn_interleaved}": 1.0},
            num_spin_orbitals=2 * num_sites,
        )
        op = _apply_spin_orbital_ordering(op, num_sites=num_sites, ordering=ordering)
        total = (total + mapper.map(op)).simplify(atol=1e-12)
    return total


def _evolve_state(
    hamiltonian: SparsePauliOp | list[Pauli | SparsePauliOp],
    initial_state: QuantumCircuit,
    time: float,
    synthesis,
    decompose_reps: int = 0,
) -> Statevector:
    """Evolve a state by exp(-i * time * H) synthesized by the requested method.

    `decompose_reps` is used for product-formula paths so simulation applies the
    synthesized gate sequence rather than the exact matrix of PauliEvolutionGate.
    """

    qc = QuantumCircuit(initial_state.num_qubits)
    qc.compose(initial_state, inplace=True)
    qc.append(PauliEvolutionGate(operator=hamiltonian, time=time, synthesis=synthesis), range(initial_state.num_qubits))
    if decompose_reps > 0:
        qc = qc.decompose(reps=decompose_reps)
    return Statevector.from_instruction(qc)


def _split_hamiltonian_terms(hamiltonian: SparsePauliOp) -> list[SparsePauliOp]:
    """Split H into a list of single-Pauli terms for product-formula synthesis."""

    return [SparsePauliOp.from_list([(label, coeff)]) for label, coeff in hamiltonian.to_list()]


def _expectation_value(state: Statevector, operator: SparsePauliOp) -> float:
    """Real expectation value <state|operator|state>."""

    return float(np.real(state.expectation_value(operator)))


def _state_amplitudes_dict(state: Statevector, cutoff: float = 1e-10) -> dict[str, dict[str, float]]:
    """Serialize statevector amplitudes in computational basis q_(n-1)...q_0."""

    out: dict[str, dict[str, float]] = {}
    for basis_index, amp in enumerate(state.data):
        if abs(amp) < cutoff:
            continue
        bitstring = format(basis_index, f"0{state.num_qubits}b")
        out[bitstring] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _sparse_pauli_to_dict(op: SparsePauliOp, tol: float = 1e-12) -> dict[str, float | dict[str, float]]:
    """Serialize a SparsePauliOp into a stable dict label -> coefficient."""

    terms: dict[str, float | dict[str, float]] = {}
    for label, coeff in sorted(op.to_list(), key=lambda item: item[0]):
        c = complex(coeff)
        if abs(c) <= tol:
            continue
        if abs(c.imag) <= tol:
            terms[label] = float(c.real)
        else:
            terms[label] = {"re": float(c.real), "im": float(c.imag)}
    return terms


def _single_basis_label(state: Statevector, tol: float = 1e-12) -> str | None:
    """Return the unique basis label if state is a single computational-basis ket."""

    nonzero_indices = [idx for idx, amp in enumerate(state.data) if abs(amp) > tol]
    if len(nonzero_indices) != 1:
        return None
    return format(nonzero_indices[0], f"0{state.num_qubits}b")


def _summarize_dynamics(dynamics: list[DynamicsPoint]) -> DynamicsSummary:
    """Compute compact summary metrics used in JSON and PDF reporting."""

    fidelity = np.array([p.fidelity for p in dynamics], dtype=float)
    l2err = np.array([p.state_l2_error for p in dynamics], dtype=float)
    energy_abs_err = np.array([abs(p.energy_exact - p.energy_trotter) for p in dynamics], dtype=float)
    n_up_abs_err = np.array([abs(p.n_up_site0_exact - p.n_up_site0_trotter) for p in dynamics], dtype=float)
    n_dn_abs_err = np.array([abs(p.n_dn_site0_exact - p.n_dn_site0_trotter) for p in dynamics], dtype=float)
    doublon_abs_err = np.array([abs(p.doublon_exact - p.doublon_trotter) for p in dynamics], dtype=float)
    exact_n_up = np.array([p.n_up_site0_exact for p in dynamics], dtype=float)
    exact_n_dn = np.array([p.n_dn_site0_exact for p in dynamics], dtype=float)
    exact_doublon = np.array([p.doublon_exact for p in dynamics], dtype=float)
    return DynamicsSummary(
        min_fidelity=float(np.min(fidelity)),
        mean_fidelity=float(np.mean(fidelity)),
        final_fidelity=float(fidelity[-1]),
        max_l2_error=float(np.max(l2err)),
        final_l2_error=float(l2err[-1]),
        max_abs_energy_error=float(np.max(energy_abs_err)),
        mean_abs_energy_error=float(np.mean(energy_abs_err)),
        final_abs_energy_error=float(energy_abs_err[-1]),
        max_abs_n_up_error=float(np.max(n_up_abs_err)),
        max_abs_n_dn_error=float(np.max(n_dn_abs_err)),
        max_abs_doublon_error=float(np.max(doublon_abs_err)),
        exact_n_up_span=float(np.max(exact_n_up) - np.min(exact_n_up)),
        exact_n_dn_span=float(np.max(exact_n_dn) - np.min(exact_n_dn)),
        exact_doublon_span=float(np.max(exact_doublon) - np.min(exact_doublon)),
    )


def _write_time_dynamics_pdf(
    output_pdf: str,
    dynamics: list[DynamicsPoint],
    summary: DynamicsSummary,
    args: argparse.Namespace,
    initial_statevector: Statevector,
    vqe_state: UCCSDVQEState | None,
) -> None:
    """Render time-dynamics summary plots into a PDF."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    t = np.array([p.time for p in dynamics], dtype=float)
    fidelity = np.array([p.fidelity for p in dynamics], dtype=float)
    l2err = np.array([p.state_l2_error for p in dynamics], dtype=float)
    e_exact = np.array([p.energy_exact for p in dynamics], dtype=float)
    e_trot = np.array([p.energy_trotter for p in dynamics], dtype=float)
    nup_exact = np.array([p.n_up_site0_exact for p in dynamics], dtype=float)
    nup_trot = np.array([p.n_up_site0_trotter for p in dynamics], dtype=float)
    ndn_exact = np.array([p.n_dn_site0_exact for p in dynamics], dtype=float)
    ndn_trot = np.array([p.n_dn_site0_trotter for p in dynamics], dtype=float)
    d_exact = np.array([p.doublon_exact for p in dynamics], dtype=float)
    d_trot = np.array([p.doublon_trotter for p in dynamics], dtype=float)

    initial_label = _single_basis_label(initial_statevector)
    if initial_label is None:
        initial_label = "superposition"
    method_label = "Lie-Trotter" if args.suzuki_order == 1 else f"Suzuki order {args.suzuki_order}"

    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        ax00 = axes[0, 0]
        ax01 = axes[0, 1]
        ax10 = axes[1, 0]
        ax11 = axes[1, 1]

        ax00.plot(t, fidelity, label="Fidelity |<psi_exact|psi_trot>|^2", color="#1f77b4", linewidth=1.8)
        ax00_twin = ax00.twinx()
        ax00_twin.plot(t, l2err, label="L2 error ||dpsi||_2", color="#d62728", linestyle="--", linewidth=1.4)
        ax00.set_ylabel("Fidelity")
        ax00_twin.set_ylabel("L2 error")
        ax00.set_title("State Error Metrics")
        ax00.grid(alpha=0.25)
        lines_a, labels_a = ax00.get_legend_handles_labels()
        lines_b, labels_b = ax00_twin.get_legend_handles_labels()
        ax00.legend(lines_a + lines_b, labels_a + labels_b, loc="lower left", fontsize=8)

        ax01.plot(t, e_exact, label="Exact <H>", color="#2ca02c", linewidth=1.8)
        ax01.plot(t, e_trot, label="Trotter <H>", color="#ff7f0e", linestyle="--", linewidth=1.6)
        ax01.set_ylabel("Energy")
        ax01.set_title("Energy Dynamics")
        ax01.grid(alpha=0.25)
        ax01.legend(loc="best", fontsize=8)

        ax10.plot(t, nup_exact, label="<n_up,site0> exact", color="#17becf", linewidth=1.8)
        ax10.plot(t, nup_trot, label="<n_up,site0> trotter", color="#17becf", linestyle="--", linewidth=1.4)
        ax10.plot(t, ndn_exact, label="<n_dn,site0> exact", color="#9467bd", linewidth=1.8)
        ax10.plot(t, ndn_trot, label="<n_dn,site0> trotter", color="#9467bd", linestyle="--", linewidth=1.4)
        ax10.set_xlabel("Time")
        ax10.set_ylabel("Occupation")
        ax10.set_title("Site-0 Occupations")
        ax10.grid(alpha=0.25)
        ax10.legend(loc="best", fontsize=8)

        ax11.plot(t, d_exact, label="Doublon exact", color="#8c564b", linewidth=1.8)
        ax11.plot(t, d_trot, label="Doublon trotter", color="#e377c2", linestyle="--", linewidth=1.4)
        ax11.set_xlabel("Time")
        ax11.set_ylabel("Doublon")
        ax11.set_title("Doublon Dynamics")
        ax11.grid(alpha=0.25)
        ax11.legend(loc="best", fontsize=8)

        metadata_lines = [
            f"initial_state={args.initial_state} ({initial_label})",
            f"L=2, t={args.hopping_t}, U={args.onsite_u}, boundary={args.boundary}, ordering={args.ordering}",
            f"{method_label}, trotter_steps={args.trotter_steps}, t_final={args.t_final}, num_times={args.num_times}",
            "Summary: "
            f"min_fidelity={summary.min_fidelity:.10f}, "
            f"final_fidelity={summary.final_fidelity:.10f}, "
            f"max_|dE|={summary.max_abs_energy_error:.3e}",
        ]
        if vqe_state is not None:
            metadata_lines.append(
                "VQE(UCCSD): "
                f"energy={vqe_state.vqe_energy:.12f}, reps={vqe_state.uccsd_reps}, "
                f"params={vqe_state.uccsd_num_parameters}, best_restart={vqe_state.best_restart}, "
                f"F(vqe,hf)={vqe_state.fidelity_vs_hf:.10f}"
            )
        fig.text(0.02, 0.01, "\n".join(metadata_lines), fontsize=8.2, family="monospace")

        fig.suptitle("Hubbard Dimer Time Dynamics (Qiskit Built-ins)", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
        pdf.savefig(fig)
        plt.close(fig)

        fig2 = plt.figure(figsize=(11.0, 8.5))
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        key_metrics = [
            "Key quantitative outcomes",
            f"- min fidelity: {summary.min_fidelity:.10f}",
            f"- mean fidelity: {summary.mean_fidelity:.10f}",
            f"- final fidelity at t={args.t_final:.3f}: {summary.final_fidelity:.10f}",
            f"- max state L2 error: {summary.max_l2_error:.6e}",
            f"- final state L2 error: {summary.final_l2_error:.6e}",
            f"- max |energy_exact - energy_trotter|: {summary.max_abs_energy_error:.6e}",
            f"- mean |energy_exact - energy_trotter|: {summary.mean_abs_energy_error:.6e}",
            f"- max |n_up exact - trotter|: {summary.max_abs_n_up_error:.6e}",
            f"- max |n_dn exact - trotter|: {summary.max_abs_n_dn_error:.6e}",
            f"- max |doublon exact - trotter|: {summary.max_abs_doublon_error:.6e}",
            f"- exact-trajectory span in n_up: {summary.exact_n_up_span:.6e}",
            f"- exact-trajectory span in n_dn: {summary.exact_n_dn_span:.6e}",
            f"- exact-trajectory span in doublon: {summary.exact_doublon_span:.6e}",
        ]
        interpretation_lines = [
            "Interpretation and meaning",
            "- The exact curve uses MatrixExponential synthesis and is the reference trajectory.",
            f"- The approximate curve uses {method_label} with reps={args.trotter_steps}; all deviations from exact come from product-formula error.",
            "- Fidelity close to 1 across the full interval means the Trotterized state remains close to the exact statevector.",
            "- The energy drift between exact and Trotter trajectories quantifies accumulated approximation error.",
            "- For a time-independent Hamiltonian, exact <H> is conserved; any drift in <H>_trotter is numerical/Trotterization error.",
            "- Occupation and doublon agreement indicates that physically relevant observables are preserved well by this Trotter setting.",
        ]
        if args.initial_state == "vqe_uccsd":
            interpretation_lines.append(
                "- Because the initial state is the VQE(UCCSD) ground-state candidate, small exact observable span indicates it is close to an eigenstate of H."
            )
            if vqe_state is not None:
                interpretation_lines.append(
                    f"- VQE preparation quality marker: F(|psi_vqe>, |psi_hf>)={vqe_state.fidelity_vs_hf:.10f} (lower means VQE moved away from HF)."
                )
        use_for_baseline_lines = [
            "How to use this baseline",
            "- This run is a Qiskit built-in reference for later hardcoded-algorithm validation.",
            "- Compare hardcoded trajectories against the exact and Trotter curves with identical Hamiltonian, ordering, and initial state.",
            "- Matching fidelity/observable error scales indicates the hardcoded implementation reproduces the same dynamics regime.",
        ]
        full_text = "\n\n".join(
            [
                "\n".join(key_metrics),
                "\n".join(interpretation_lines),
                "\n".join(use_for_baseline_lines),
            ]
        )
        wrapped_text = "\n".join(textwrap.fill(line, width=118) if line and not line.startswith("-") else line for line in full_text.splitlines())
        ax2.text(
            0.03,
            0.97,
            "Detailed Explanation of Dynamics Results",
            fontsize=16,
            weight="bold",
            va="top",
            ha="left",
        )
        ax2.text(0.03, 0.92, wrapped_text, fontsize=10.5, family="monospace", va="top", ha="left")
        pdf.savefig(fig2)
        plt.close(fig2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Hubbard dimer real-time dynamics using Qiskit built-ins: "
            "Suzuki-Trotter (approximate) vs MatrixExponential (reference)."
        )
    )
    parser.add_argument("--hopping-t", type=float, default=1.0, help="Hopping parameter t.")
    parser.add_argument("--onsite-u", type=float, default=4.0, help="On-site interaction U.")
    parser.add_argument(
        "--boundary",
        choices=("open", "periodic"),
        default="periodic",
        help="Lattice boundary condition.",
    )
    parser.add_argument(
        "--ordering",
        choices=("blocked", "interleaved"),
        default=DEFAULT_ORDERING,
        help="Spin-orbital ordering before JW map.",
    )
    parser.add_argument("--t-final", type=float, default=2.0, help="Final evolution time.")
    parser.add_argument("--num-times", type=int, default=11, help="Number of time samples (includes t=0).")
    parser.add_argument(
        "--initial-state",
        choices=("hf", "vqe_uccsd"),
        default="hf",
        help="Initial state for dynamics: HF determinant or VQE-optimized UCCSD state.",
    )
    parser.add_argument("--uccsd-reps", type=int, default=2, help="UCCSD ansatz reps used when --initial-state=vqe_uccsd.")
    parser.add_argument("--vqe-restarts", type=int, default=3, help="Number of VQE random restarts.")
    parser.add_argument("--vqe-maxiter", type=int, default=1800, help="SLSQP max iterations per VQE restart.")
    parser.add_argument("--vqe-seed", type=int, default=7, help="Random seed for VQE restart initialization.")
    parser.add_argument(
        "--suzuki-order",
        type=int,
        default=2,
        help="Product-formula order. Use 1 for Lie-Trotter, or even number >=2 for Suzuki.",
    )
    parser.add_argument("--trotter-steps", type=int, default=8, help="Number of Trotter steps per sample time.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON path for machine-readable trajectory/results.",
    )
    parser.add_argument(
        "--output-pdf",
        type=str,
        default=None,
        help="Optional output PDF path for summary plots of time dynamics.",
    )
    parser.add_argument(
        "--dump-states",
        action="store_true",
        help="Include exact/trotter statevector amplitudes for every time point in JSON output.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if DEFAULT_NUM_SITES != 2:
        raise RuntimeError("This script is intended for the Hubbard dimer (num_sites=2).")
    if args.num_times < 2:
        raise ValueError("--num-times must be >= 2")
    if args.trotter_steps < 1:
        raise ValueError("--trotter-steps must be >= 1")
    if args.vqe_restarts < 1:
        raise ValueError("--vqe-restarts must be >= 1")
    if args.vqe_maxiter < 1:
        raise ValueError("--vqe-maxiter must be >= 1")
    if args.suzuki_order < 1:
        raise ValueError("--suzuki-order must be >= 1")
    if args.suzuki_order > 1 and args.suzuki_order % 2 != 0:
        raise ValueError("--suzuki-order must be 1 (Lie-Trotter) or an even integer (Suzuki).")

    mapper = JordanWignerMapper()
    hamiltonian = _build_qubit_hamiltonian(
        num_sites=DEFAULT_NUM_SITES,
        hopping_t=args.hopping_t,
        onsite_u=args.onsite_u,
        boundary=args.boundary,
        ordering=args.ordering,
        mapper=mapper,
    )
    hamiltonian_terms = _split_hamiltonian_terms(hamiltonian)
    vqe_state: UCCSDVQEState | None = None
    if args.initial_state == "hf":
        initial_state = _build_initial_state_circuit(num_sites=DEFAULT_NUM_SITES, ordering=args.ordering)
    else:
        vqe_state = _build_vqe_uccsd_initial_state(
            num_sites=DEFAULT_NUM_SITES,
            ordering=args.ordering,
            hamiltonian=hamiltonian,
            mapper=mapper,
            uccsd_reps=args.uccsd_reps,
            vqe_restarts=args.vqe_restarts,
            vqe_maxiter=args.vqe_maxiter,
            vqe_seed=args.vqe_seed,
        )
        initial_state = vqe_state.circuit
    initial_statevector = Statevector.from_instruction(initial_state)

    n_up_site0 = _number_operator_qubit(
        num_sites=DEFAULT_NUM_SITES,
        site=0,
        spin="up",
        ordering=args.ordering,
        mapper=mapper,
    )
    n_dn_site0 = _number_operator_qubit(
        num_sites=DEFAULT_NUM_SITES,
        site=0,
        spin="dn",
        ordering=args.ordering,
        mapper=mapper,
    )
    doublon = _doublon_operator_qubit(
        num_sites=DEFAULT_NUM_SITES,
        ordering=args.ordering,
        mapper=mapper,
    )

    exact_synthesis = MatrixExponential()
    if args.suzuki_order == 1:
        trotter_synthesis = LieTrotter(reps=args.trotter_steps, preserve_order=True)
    else:
        trotter_synthesis = SuzukiTrotter(
            order=args.suzuki_order,
            reps=args.trotter_steps,
            preserve_order=True,
        )

    times = np.linspace(0.0, args.t_final, args.num_times)
    dynamics: list[DynamicsPoint] = []
    dynamics_json: list[dict[str, object]] = []

    for t in times:
        exact_state = _evolve_state(
            hamiltonian=hamiltonian,
            initial_state=initial_state,
            time=float(t),
            synthesis=exact_synthesis,
            decompose_reps=0,
        )
        trotter_state = _evolve_state(
            hamiltonian=hamiltonian_terms,
            initial_state=initial_state,
            time=float(t),
            synthesis=trotter_synthesis,
            decompose_reps=2,
        )

        overlap = np.vdot(exact_state.data, trotter_state.data)
        point = DynamicsPoint(
            time=float(t),
            fidelity=float(abs(overlap) ** 2),
            state_l2_error=float(np.linalg.norm(exact_state.data - trotter_state.data)),
            energy_exact=_expectation_value(exact_state, hamiltonian),
            energy_trotter=_expectation_value(trotter_state, hamiltonian),
            n_up_site0_exact=_expectation_value(exact_state, n_up_site0),
            n_up_site0_trotter=_expectation_value(trotter_state, n_up_site0),
            n_dn_site0_exact=_expectation_value(exact_state, n_dn_site0),
            n_dn_site0_trotter=_expectation_value(trotter_state, n_dn_site0),
            doublon_exact=_expectation_value(exact_state, doublon),
            doublon_trotter=_expectation_value(trotter_state, doublon),
        )
        dynamics.append(point)

        point_json = asdict(point)
        if args.dump_states:
            point_json["exact_state"] = _state_amplitudes_dict(exact_state)
            point_json["trotter_state"] = _state_amplitudes_dict(trotter_state)
        dynamics_json.append(point_json)

    print("Hubbard Dimer Real-Time Dynamics (Qiskit built-ins)")
    print(
        "Settings: "
        f"L=2, t={args.hopping_t}, U={args.onsite_u}, boundary={args.boundary}, "
        f"ordering={args.ordering}, initial_state={args.initial_state}, "
        f"t_final={args.t_final}, num_times={args.num_times}, "
        f"suzuki_order={args.suzuki_order}, trotter_steps={args.trotter_steps}"
    )
    single_basis_label = _single_basis_label(initial_statevector)
    if single_basis_label is None:
        print("Initial state (q3 q2 q1 q0): superposition")
    else:
        print("Initial state (q3 q2 q1 q0):", single_basis_label)
    if vqe_state is not None:
        print(
            "VQE(UCCSD) summary: "
            f"energy={vqe_state.vqe_energy:.12f}, reps={vqe_state.uccsd_reps}, "
            f"params={vqe_state.uccsd_num_parameters}, best_restart={vqe_state.best_restart}, "
            f"num_particles={vqe_state.num_particles}, F(vqe,hf)={vqe_state.fidelity_vs_hf:.10f}"
        )
    print("-" * 120)
    print(
        "time    fidelity        ||dpsi||_2      <H>_exact       <H>_trotter     "
        "<n_up,0>_exact   <n_up,0>_trotter  doublon_exact   doublon_trotter"
    )
    for point in dynamics:
        print(
            f"{point.time:5.2f}  "
            f"{point.fidelity: .10f}  "
            f"{point.state_l2_error: .3e}  "
            f"{point.energy_exact: .10f}  "
            f"{point.energy_trotter: .10f}  "
            f"{point.n_up_site0_exact: .10f}  "
            f"{point.n_up_site0_trotter: .10f}  "
            f"{point.doublon_exact: .10f}  "
            f"{point.doublon_trotter: .10f}"
        )

    summary = _summarize_dynamics(dynamics)
    print(
        "Summary metrics: "
        f"min_fidelity={summary.min_fidelity:.10f}, "
        f"final_fidelity={summary.final_fidelity:.10f}, "
        f"max_|dE|={summary.max_abs_energy_error:.3e}, "
        f"max_L2={summary.max_l2_error:.3e}"
    )

    if args.output_json:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "description": (
                "Standalone Hubbard dimer time dynamics using Qiskit built-ins "
                "(Suzuki/Lie-Trotter vs MatrixExponential reference)."
            ),
            "settings": {
                "num_sites": DEFAULT_NUM_SITES,
                "hopping_t": args.hopping_t,
                "onsite_u": args.onsite_u,
                "boundary": args.boundary,
                "spin_orbital_ordering": args.ordering,
                "initial_state": args.initial_state,
                "t_final": args.t_final,
                "num_times": args.num_times,
                "suzuki_order": args.suzuki_order,
                "trotter_steps": args.trotter_steps,
                "initial_state_label_qn_to_q0": _single_basis_label(initial_statevector),
                "initial_state_amplitudes_qn_to_q0": _state_amplitudes_dict(initial_statevector),
            },
            "summary_metrics": asdict(summary),
            "hamiltonian_jw_terms": _sparse_pauli_to_dict(hamiltonian),
            "trajectory": dynamics_json,
        }
        if vqe_state is not None:
            payload["vqe_uccsd"] = {
                "energy": vqe_state.vqe_energy,
                "best_restart": vqe_state.best_restart,
                "num_particles": {
                    "n_up": int(vqe_state.num_particles[0]),
                    "n_dn": int(vqe_state.num_particles[1]),
                },
                "uccsd_reps": vqe_state.uccsd_reps,
                "uccsd_num_parameters": vqe_state.uccsd_num_parameters,
                "fidelity_vs_hf": vqe_state.fidelity_vs_hf,
                "optimal_point": vqe_state.optimal_point,
            }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
        print(f"Wrote JSON results to: {args.output_json}")
    if args.output_pdf:
        _write_time_dynamics_pdf(
            output_pdf=args.output_pdf,
            dynamics=dynamics,
            summary=summary,
            args=args,
            initial_statevector=initial_statevector,
            vqe_state=vqe_state,
        )
        print(f"Wrote PDF plot to: {args.output_pdf}")


if __name__ == "__main__":
    main()
