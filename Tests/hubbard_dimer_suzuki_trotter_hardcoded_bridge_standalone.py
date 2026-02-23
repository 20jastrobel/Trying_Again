"""Standalone bridge: hardcoded Hubbard primitives + Qiskit time evolution.

Purpose:
- Keep the same Qiskit built-in time-evolution approach used in
  `hubbard_dimer_suzuki_trotter_standalone.py`.
- Build the Hubbard dimer Hamiltonian via the local primitive builder
  (`pydephasing.quantum.hubbard_latex_python_pairs.build_hubbard_hamiltonian`).
- Recover the hardcoded VQE wavevector by executing the local
  `vqe_latex_python_pairs.ipynb` workflow namespace and running its hardcoded
  VQE routine.
- Compare this primitive/hardcoded pipeline to the Qiskit-Nature closed-form
  reference pipeline.

This file is self-contained and does not modify other repository files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Avoid matplotlib cache warnings triggered by notebook-loaded code on some systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, MatrixExponential, SuzukiTrotter
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
DEFAULT_BOUNDARY = "periodic"


@dataclass
class HardcodedVQEPayload:
    """Hardcoded VQE objects/materialized values loaded from local primitives."""

    quantum_dir: str
    notebook_path: str
    theta: list[float]
    energy: float
    hf_bitstring: str
    num_particles: tuple[int, int]
    psi_vqe: np.ndarray
    hamiltonian: SparsePauliOp


@dataclass
class BridgePoint:
    """One time point in bridge comparison."""

    time: float
    fid_exact_hardcoded_vs_qiskit: float
    l2_exact_hardcoded_vs_qiskit: float
    fid_trotter_hardcoded_vs_exact_qiskit: float
    fid_trotter_qiskit_vs_exact_qiskit: float
    energy_exact_qiskit_vqeinit: float
    energy_exact_hardcoded_vqeinit: float
    energy_trotter_hardcoded_vqeinit: float
    doublon_exact_qiskit_vqeinit: float
    doublon_exact_hardcoded_vqeinit: float
    doublon_trotter_hardcoded_vqeinit: float
    doublon_exact_qiskit_hfinit: float
    fid_exact_qiskit_vqeinit_vs_hfinit: float


def _interleaved_to_blocked_permutation(num_sites: int) -> list[int]:
    """Map interleaved spin ordering (a1,b1,a2,b2,...) to blocked ordering."""

    return [index for site in range(num_sites) for index in (site, num_sites + site)]


def _apply_spin_orbital_ordering(op, num_sites: int, ordering: str):
    """Return fermionic operator in requested spin-orbital ordering."""

    normalized = ordering.strip().lower()
    if normalized == "interleaved":
        return op
    if normalized == "blocked":
        return op.permute_indices(_interleaved_to_blocked_permutation(num_sites))
    raise ValueError(f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'.")


def _build_qubit_hamiltonian_qiskit_nature(
    num_sites: int,
    hopping_t: float,
    onsite_u: float,
    boundary: str,
    ordering: str,
    mapper: JordanWignerMapper,
) -> SparsePauliOp:
    """Construct and JW-map Fermi-Hubbard Hamiltonian via Qiskit Nature."""

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
    return ((num_sites + 1) // 2, num_sites // 2)


def _map_interleaved_index(index_interleaved: int, num_sites: int, ordering: str) -> int:
    normalized = ordering.strip().lower()
    if normalized == "interleaved":
        return index_interleaved
    if normalized == "blocked":
        return _interleaved_to_blocked_permutation(num_sites)[index_interleaved]
    raise ValueError(f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'.")


def _build_initial_hf_state_circuit(num_sites: int, ordering: str) -> QuantumCircuit:
    """Build half-filled HF computational basis state."""

    n_alpha, n_beta = _half_filled_particle_numbers(num_sites)
    occupied_interleaved: list[int] = []
    occupied_interleaved.extend(2 * site for site in range(n_alpha))
    occupied_interleaved.extend(2 * site + 1 for site in range(n_beta))
    occupied = [_map_interleaved_index(i, num_sites=num_sites, ordering=ordering) for i in occupied_interleaved]

    num_qubits = 2 * num_sites
    qc = QuantumCircuit(num_qubits)
    for qubit in occupied:
        qc.x(qubit)
    return qc


def _statevector_to_circuit(psi: np.ndarray) -> QuantumCircuit:
    """Initialize an exact statevector as a QuantumCircuit state-preparation circuit."""

    psi_arr = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(psi_arr))
    if nrm <= 0.0:
        raise ValueError("Statevector has zero norm.")
    psi_arr = psi_arr / nrm

    nq = int(round(np.log2(psi_arr.size)))
    if psi_arr.size != 2**nq:
        raise ValueError("Statevector length is not a power of 2.")

    qc = QuantumCircuit(nq)
    qc.initialize(psi_arr, qc.qubits)
    return qc


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
    """Evolve state by exp(-i time H) using Qiskit PauliEvolutionGate synthesis."""

    qc = QuantumCircuit(initial_state.num_qubits)
    qc.compose(initial_state, inplace=True)
    qc.append(PauliEvolutionGate(operator=hamiltonian, time=time, synthesis=synthesis), range(initial_state.num_qubits))
    if decompose_reps > 0:
        qc = qc.decompose(reps=decompose_reps)
    return Statevector.from_instruction(qc)


def _split_hamiltonian_terms(hamiltonian: SparsePauliOp) -> list[SparsePauliOp]:
    """Split H into single-Pauli terms for product-formula synthesis."""

    return [SparsePauliOp.from_list([(label, coeff)]) for label, coeff in hamiltonian.to_list()]


def _expectation_value(state: Statevector, operator: SparsePauliOp) -> float:
    return float(np.real(state.expectation_value(operator)))


def _state_fidelity(a: Statevector | np.ndarray, b: Statevector | np.ndarray) -> float:
    """State fidelity |<a|b>|^2."""

    va = np.asarray(a.data if isinstance(a, Statevector) else a, dtype=complex).reshape(-1)
    vb = np.asarray(b.data if isinstance(b, Statevector) else b, dtype=complex).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError("Statevectors have different shapes.")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0.0 or nb <= 0.0:
        raise ValueError("Statevector norm must be > 0.")
    va /= na
    vb /= nb
    return float(abs(np.vdot(va, vb)) ** 2)


def _state_l2_error(a: Statevector | np.ndarray, b: Statevector | np.ndarray) -> float:
    va = np.asarray(a.data if isinstance(a, Statevector) else a, dtype=complex).reshape(-1)
    vb = np.asarray(b.data if isinstance(b, Statevector) else b, dtype=complex).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError("Statevectors have different shapes.")
    return float(np.linalg.norm(va - vb))


def _sparse_pauli_to_coeff_map(op: SparsePauliOp, tol: float = 1e-12) -> dict[str, complex]:
    """Map Pauli label -> coefficient (combined)."""

    coeffs: dict[str, complex] = {}
    for label, coeff in op.to_list():
        c = complex(coeff)
        if abs(c) <= tol:
            continue
        coeffs[label] = coeffs.get(label, 0.0 + 0.0j) + c
    return coeffs


def _hamiltonian_delta_metrics(a: SparsePauliOp, b: SparsePauliOp, tol: float = 1e-12) -> dict[str, float]:
    """Coefficient-space difference metrics for two SparsePauliOps."""

    ma = _sparse_pauli_to_coeff_map(a, tol=tol)
    mb = _sparse_pauli_to_coeff_map(b, tol=tol)
    keys = sorted(set(ma.keys()) | set(mb.keys()))

    diffs = [ma.get(k, 0.0 + 0.0j) - mb.get(k, 0.0 + 0.0j) for k in keys]
    if diffs:
        max_abs = float(max(abs(d) for d in diffs))
        l2 = float(np.linalg.norm(np.asarray(diffs, dtype=complex)))
    else:
        max_abs = 0.0
        l2 = 0.0

    return {
        "num_terms_a": float(len(ma)),
        "num_terms_b": float(len(mb)),
        "num_terms_union": float(len(keys)),
        "max_abs_coeff_delta": max_abs,
        "l2_coeff_delta": l2,
    }


def _sparse_pauli_to_jsonable(op: SparsePauliOp, tol: float = 1e-12) -> dict[str, float | dict[str, float]]:
    """Stable JSON representation label -> coefficient."""

    out: dict[str, float | dict[str, float]] = {}
    for label, coeff in sorted(op.to_list(), key=lambda x: x[0]):
        c = complex(coeff)
        if abs(c) <= tol:
            continue
        if abs(c.imag) <= tol:
            out[label] = float(c.real)
        else:
            out[label] = {"re": float(c.real), "im": float(c.imag)}
    return out


def _pauli_polynomial_to_sparse_pauli_op(hardcoded_poly, tol: float = 1e-12) -> SparsePauliOp:
    """Convert local hardcoded PauliPolynomial (e/x/y/z alphabet) to SparsePauliOp."""

    terms: list[tuple[str, complex]] = []
    poly_terms = list(hardcoded_poly.return_polynomial())
    for term in poly_terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        label = term.pw2strng().replace("e", "I").upper()
        terms.append((label, coeff))

    if not terms:
        nq = int(poly_terms[0].nqubit()) if poly_terms else 1
        terms = [("I" * nq, 0.0)]

    return SparsePauliOp.from_list(terms).simplify(atol=tol)


def _candidate_quantum_dirs(explicit_repo_root: str | None) -> list[Path]:
    candidates: list[Path] = []

    if explicit_repo_root:
        root = Path(explicit_repo_root).expanduser().resolve()
        candidates.extend([root / "pydephasing" / "quantum", root / "quantum"])

    roots = [Path.cwd(), *Path.cwd().parents[:4]]
    for root in roots:
        candidates.extend([root / "pydephasing" / "quantum", root / "quantum"])

    # Legacy fallback used by prior notebooks in this workspace.
    candidates.append(Path("/Users/jakestrobel/Downloads/dephasing-code-base-main 2/pydephasing/quantum"))

    seen: set[str] = set()
    uniq: list[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _bootstrap_pydephasing_quantum_alias(explicit_repo_root: str | None) -> Path:
    """Load local pydephasing.quantum modules from filesystem paths, without package install."""

    quantum_dir: Path | None = None
    for cand in _candidate_quantum_dirs(explicit_repo_root):
        if (cand / "pauli_polynomial_class.py").exists() and (cand / "hubbard_latex_python_pairs.py").exists():
            quantum_dir = cand.resolve()
            break

    if quantum_dir is None:
        raise FileNotFoundError(
            "Could not locate local pydephasing quantum directory containing "
            "pauli_polynomial_class.py and hubbard_latex_python_pairs.py."
        )

    root_pkg = sys.modules.get("pydephasing")
    if root_pkg is None:
        root_pkg = types.ModuleType("pydephasing")
        root_pkg.__path__ = [str(quantum_dir.parent)]
        sys.modules["pydephasing"] = root_pkg

    quantum_pkg = types.ModuleType("pydephasing.quantum")
    quantum_pkg.__path__ = [str(quantum_dir)]
    sys.modules["pydephasing.quantum"] = quantum_pkg

    load_order = [
        "pauli_letters_module",
        "pauli_words",
        "pauli_polynomial_class",
        "hartree_fock_reference_state",
        "hubbard_latex_python_pairs",
    ]

    for name in load_order:
        full_name = f"pydephasing.quantum.{name}"
        file_path = quantum_dir / f"{name}.py"
        if full_name in sys.modules:
            # Keep already-loaded module if it resolves to the same path.
            mod_file = getattr(sys.modules[full_name], "__file__", None)
            if mod_file and Path(mod_file).resolve() == file_path.resolve():
                continue

        spec = importlib.util.spec_from_file_location(full_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec for {full_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    return quantum_dir


def _resolve_vqe_notebook_path(quantum_dir: Path, explicit_repo_root: str | None) -> Path:
    """Resolve vqe_latex_python_pairs.ipynb used by hardcoded VQE workflow."""

    candidates: list[Path] = []

    if explicit_repo_root:
        root = Path(explicit_repo_root).expanduser().resolve()
        candidates.extend([
            root / "pydephasing" / "quantum" / "vqe_latex_python_pairs.ipynb",
            root / "quantum" / "vqe_latex_python_pairs.ipynb",
            root / "vqe_latex_python_pairs.ipynb",
        ])

    roots = [Path.cwd(), *Path.cwd().parents[:4]]
    for root in roots:
        candidates.extend([
            root / "pydephasing" / "quantum" / "vqe_latex_python_pairs.ipynb",
            root / "quantum" / "vqe_latex_python_pairs.ipynb",
            root / "vqe_latex_python_pairs.ipynb",
        ])

    candidates.append(quantum_dir / "vqe_latex_python_pairs.ipynb")

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError("Could not locate vqe_latex_python_pairs.ipynb for hardcoded VQE setup.")


def _load_hardcoded_vqe_namespace(vqe_nb_path: Path) -> dict[str, Any]:
    """Exec code cells from VQE notebook into a Python namespace."""

    payload = json.loads(vqe_nb_path.read_text())
    ns: dict[str, Any] = {}

    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Skip long final benchmark driver; keep definitions.
        if "Final benchmark: hardcoded VQE vs Qiskit VQE" in src:
            continue
        exec(src, ns)

    required = [
        "half_filled_num_particles",
        "HubbardTermwiseAnsatz",
        "hartree_fock_bitstring",
        "basis_state",
        "vqe_minimize",
    ]
    missing = [k for k in required if k not in ns]
    if missing:
        raise RuntimeError(f"Missing required symbols from VQE notebook namespace: {missing}")

    return ns


def _build_hardcoded_vqe_payload(
    *,
    num_sites: int,
    hopping_t: float,
    onsite_u: float,
    ordering: str,
    boundary: str,
    vqe_reps: int,
    vqe_restarts: int,
    vqe_seed: int,
    vqe_maxiter: int,
    explicit_repo_root: str | None,
) -> HardcodedVQEPayload:
    """Build hardcoded Hamiltonian + hardcoded VQE wavevector from local primitives."""

    if num_sites != 2:
        raise ValueError("Hardcoded bridge currently targets Hubbard dimer only (num_sites=2).")

    quantum_dir = _bootstrap_pydephasing_quantum_alias(explicit_repo_root)
    vqe_nb_path = _resolve_vqe_notebook_path(quantum_dir, explicit_repo_root)
    vqe_ns = _load_hardcoded_vqe_namespace(vqe_nb_path)

    from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian

    pbc = boundary.strip().lower() == "periodic"

    num_particles = tuple(vqe_ns["half_filled_num_particles"](num_sites))

    h_hardcoded = build_hubbard_hamiltonian(
        dims=num_sites,
        t=float(hopping_t),
        U=float(onsite_u),
        v=None,
        repr_mode="JW",
        indexing=ordering,
        pbc=pbc,
    )

    hardcoded_ansatz = vqe_ns["HubbardTermwiseAnsatz"](
        dims=num_sites,
        t=float(hopping_t),
        U=float(onsite_u),
        v=None,
        reps=int(vqe_reps),
        repr_mode="JW",
        indexing=ordering,
        pbc=pbc,
        include_potential_terms=True,
    )

    hf_bits = str(
        vqe_ns["hartree_fock_bitstring"](
            n_sites=num_sites,
            num_particles=num_particles,
            indexing=ordering,
        )
    )
    psi_ref = np.asarray(vqe_ns["basis_state"](2 * num_sites, hf_bits), dtype=complex)

    vqe_result = vqe_ns["vqe_minimize"](
        h_hardcoded,
        hardcoded_ansatz,
        psi_ref,
        restarts=int(vqe_restarts),
        seed=int(vqe_seed),
        maxiter=int(vqe_maxiter),
        method="SLSQP",
    )

    theta = np.asarray(vqe_result.theta, dtype=float).reshape(-1)
    energy = float(vqe_result.energy)

    psi_vqe = np.asarray(hardcoded_ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(psi_vqe))
    if nrm <= 0.0:
        raise RuntimeError("Hardcoded ansatz produced zero-norm state.")
    psi_vqe = psi_vqe / nrm

    h_sparse = _pauli_polynomial_to_sparse_pauli_op(h_hardcoded)

    return HardcodedVQEPayload(
        quantum_dir=str(quantum_dir),
        notebook_path=str(vqe_nb_path),
        theta=[float(x) for x in theta],
        energy=energy,
        hf_bitstring=hf_bits,
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        psi_vqe=psi_vqe,
        hamiltonian=h_sparse,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge script: hardcoded Hubbard builder + hardcoded VQE wavevector "
            "with Qiskit built-in Suzuki/Lie-Trotter dynamics, compared to "
            "Qiskit-Nature closed-form dynamics."
        )
    )
    parser.add_argument("--hopping-t", type=float, default=1.0, help="Hopping parameter t.")
    parser.add_argument("--onsite-u", type=float, default=4.0, help="On-site interaction U.")
    parser.add_argument(
        "--boundary",
        choices=("open", "periodic"),
        default=DEFAULT_BOUNDARY,
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
        "--suzuki-order",
        type=int,
        default=2,
        help="Product-formula order. Use 1 for Lie-Trotter, or even number >=2 for Suzuki.",
    )
    parser.add_argument("--trotter-steps", type=int, default=8, help="Number of Trotter steps per sample time.")

    parser.add_argument(
        "--hardcoded-repo-root",
        type=str,
        default=None,
        help="Optional repo root containing pydephasing/quantum and vqe_latex_python_pairs.ipynb.",
    )
    parser.add_argument("--hardcoded-reps", type=int, default=2, help="Hardcoded HVA reps for VQE state preparation.")
    parser.add_argument("--hardcoded-restarts", type=int, default=3, help="Hardcoded VQE restart count.")
    parser.add_argument("--hardcoded-seed", type=int, default=7, help="Hardcoded VQE random seed.")
    parser.add_argument("--hardcoded-maxiter", type=int, default=1800, help="Hardcoded VQE SLSQP maxiter.")

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON path for machine-readable comparison trajectory.",
    )
    parser.add_argument(
        "--dump-states",
        action="store_true",
        help="Include serialized state amplitudes for selected trajectories in JSON output.",
    )
    return parser


def _state_amplitudes_dict(state: Statevector, cutoff: float = 1e-10) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(state.data):
        if abs(amp) < cutoff:
            continue
        bitstring = format(idx, f"0{state.num_qubits}b")
        out[bitstring] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def main() -> None:
    args = _build_parser().parse_args()

    if DEFAULT_NUM_SITES != 2:
        raise RuntimeError("This script targets Hubbard dimer only (num_sites=2).")
    if args.num_times < 2:
        raise ValueError("--num-times must be >= 2")
    if args.trotter_steps < 1:
        raise ValueError("--trotter-steps must be >= 1")
    if args.suzuki_order < 1:
        raise ValueError("--suzuki-order must be >= 1")
    if args.suzuki_order > 1 and args.suzuki_order % 2 != 0:
        raise ValueError("--suzuki-order must be 1 (Lie-Trotter) or an even integer (Suzuki).")

    mapper = JordanWignerMapper()

    h_qiskit = _build_qubit_hamiltonian_qiskit_nature(
        num_sites=DEFAULT_NUM_SITES,
        hopping_t=args.hopping_t,
        onsite_u=args.onsite_u,
        boundary=args.boundary,
        ordering=args.ordering,
        mapper=mapper,
    )

    hardcoded = _build_hardcoded_vqe_payload(
        num_sites=DEFAULT_NUM_SITES,
        hopping_t=args.hopping_t,
        onsite_u=args.onsite_u,
        ordering=args.ordering,
        boundary=args.boundary,
        vqe_reps=args.hardcoded_reps,
        vqe_restarts=args.hardcoded_restarts,
        vqe_seed=args.hardcoded_seed,
        vqe_maxiter=args.hardcoded_maxiter,
        explicit_repo_root=args.hardcoded_repo_root,
    )

    h_hardcoded = hardcoded.hamiltonian

    # Same initial state for hardcoded-vs-qiskit Hamiltonian comparison.
    qc_vqe_init = _statevector_to_circuit(hardcoded.psi_vqe)

    # Previous closed-form reference context from earlier standalone script (HF initial state).
    qc_hf_init = _build_initial_hf_state_circuit(num_sites=DEFAULT_NUM_SITES, ordering=args.ordering)

    hf_sv = Statevector.from_instruction(qc_hf_init)
    vqe_sv = Statevector(hardcoded.psi_vqe)
    fid_vqe_vs_hf_initial = _state_fidelity(vqe_sv, hf_sv)

    n_up_site0 = _number_operator_qubit(
        num_sites=DEFAULT_NUM_SITES,
        site=0,
        spin="up",
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
        trotter_synthesis = SuzukiTrotter(order=args.suzuki_order, reps=args.trotter_steps, preserve_order=True)

    h_qiskit_terms = _split_hamiltonian_terms(h_qiskit)
    h_hardcoded_terms = _split_hamiltonian_terms(h_hardcoded)

    h_delta = _hamiltonian_delta_metrics(h_hardcoded, h_qiskit)

    times = np.linspace(0.0, args.t_final, args.num_times)
    trajectory: list[BridgePoint] = []
    trajectory_json: list[dict[str, Any]] = []

    for t in times:
        sv_exact_qiskit_vqe = _evolve_state(
            hamiltonian=h_qiskit,
            initial_state=qc_vqe_init,
            time=float(t),
            synthesis=exact_synthesis,
            decompose_reps=0,
        )
        sv_exact_hardcoded_vqe = _evolve_state(
            hamiltonian=h_hardcoded,
            initial_state=qc_vqe_init,
            time=float(t),
            synthesis=exact_synthesis,
            decompose_reps=0,
        )
        sv_trotter_hardcoded_vqe = _evolve_state(
            hamiltonian=h_hardcoded_terms,
            initial_state=qc_vqe_init,
            time=float(t),
            synthesis=trotter_synthesis,
            decompose_reps=2,
        )
        sv_trotter_qiskit_vqe = _evolve_state(
            hamiltonian=h_qiskit_terms,
            initial_state=qc_vqe_init,
            time=float(t),
            synthesis=trotter_synthesis,
            decompose_reps=2,
        )

        sv_exact_qiskit_hf = _evolve_state(
            hamiltonian=h_qiskit,
            initial_state=qc_hf_init,
            time=float(t),
            synthesis=exact_synthesis,
            decompose_reps=0,
        )

        row = BridgePoint(
            time=float(t),
            fid_exact_hardcoded_vs_qiskit=_state_fidelity(sv_exact_hardcoded_vqe, sv_exact_qiskit_vqe),
            l2_exact_hardcoded_vs_qiskit=_state_l2_error(sv_exact_hardcoded_vqe, sv_exact_qiskit_vqe),
            fid_trotter_hardcoded_vs_exact_qiskit=_state_fidelity(sv_trotter_hardcoded_vqe, sv_exact_qiskit_vqe),
            fid_trotter_qiskit_vs_exact_qiskit=_state_fidelity(sv_trotter_qiskit_vqe, sv_exact_qiskit_vqe),
            energy_exact_qiskit_vqeinit=_expectation_value(sv_exact_qiskit_vqe, h_qiskit),
            energy_exact_hardcoded_vqeinit=_expectation_value(sv_exact_hardcoded_vqe, h_hardcoded),
            energy_trotter_hardcoded_vqeinit=_expectation_value(sv_trotter_hardcoded_vqe, h_qiskit),
            doublon_exact_qiskit_vqeinit=_expectation_value(sv_exact_qiskit_vqe, doublon),
            doublon_exact_hardcoded_vqeinit=_expectation_value(sv_exact_hardcoded_vqe, doublon),
            doublon_trotter_hardcoded_vqeinit=_expectation_value(sv_trotter_hardcoded_vqe, doublon),
            doublon_exact_qiskit_hfinit=_expectation_value(sv_exact_qiskit_hf, doublon),
            fid_exact_qiskit_vqeinit_vs_hfinit=_state_fidelity(sv_exact_qiskit_vqe, sv_exact_qiskit_hf),
        )
        trajectory.append(row)

        row_json = asdict(row)
        if args.dump_states:
            row_json["state_exact_qiskit_vqeinit"] = _state_amplitudes_dict(sv_exact_qiskit_vqe)
            row_json["state_exact_hardcoded_vqeinit"] = _state_amplitudes_dict(sv_exact_hardcoded_vqe)
            row_json["state_trotter_hardcoded_vqeinit"] = _state_amplitudes_dict(sv_trotter_hardcoded_vqe)
        trajectory_json.append(row_json)

    print("Hubbard Dimer Bridge: hardcoded primitives + Qiskit built-in dynamics")
    print(
        "Settings: "
        f"L=2, t={args.hopping_t}, U={args.onsite_u}, boundary={args.boundary}, ordering={args.ordering}, "
        f"t_final={args.t_final}, num_times={args.num_times}, suzuki_order={args.suzuki_order}, trotter_steps={args.trotter_steps}"
    )
    print(
        "Hardcoded VQE source: "
        f"quantum_dir={hardcoded.quantum_dir}, notebook={hardcoded.notebook_path}"
    )
    print(
        "Hardcoded VQE summary: "
        f"energy={hardcoded.energy:.12f}, theta_len={len(hardcoded.theta)}, "
        f"num_particles={hardcoded.num_particles}, hf_bitstring={hardcoded.hf_bitstring}"
    )
    print(
        "Hamiltonian delta (hardcoded vs qiskit-nature): "
        f"max|dc|={h_delta['max_abs_coeff_delta']:.3e}, "
        f"l2|dc|={h_delta['l2_coeff_delta']:.3e}, "
        f"terms_union={int(h_delta['num_terms_union'])}"
    )
    print(
        "Initial-state fidelity: "
        f"F(|psi_vqe_hardcoded>, |psi_hf>)={fid_vqe_vs_hf_initial:.10f}"
    )
    print("-" * 160)
    print(
        "time    F(exact_hc,exact_qk)   ||dpsi||_2(exact)   F(trot_hc,exact_qk)   F(trot_qk,exact_qk)   "
        "D_exact_qk(vqe)   D_exact_hc(vqe)   D_trot_hc(vqe)   D_exact_qk(hf)   F(exact_qk(vqe), exact_qk(hf))"
    )
    for row in trajectory:
        print(
            f"{row.time:5.2f}  "
            f"{row.fid_exact_hardcoded_vs_qiskit: .10f}  "
            f"{row.l2_exact_hardcoded_vs_qiskit: .3e}  "
            f"{row.fid_trotter_hardcoded_vs_exact_qiskit: .10f}  "
            f"{row.fid_trotter_qiskit_vs_exact_qiskit: .10f}  "
            f"{row.doublon_exact_qiskit_vqeinit: .10f}  "
            f"{row.doublon_exact_hardcoded_vqeinit: .10f}  "
            f"{row.doublon_trotter_hardcoded_vqeinit: .10f}  "
            f"{row.doublon_exact_qiskit_hfinit: .10f}  "
            f"{row.fid_exact_qiskit_vqeinit_vs_hfinit: .10f}"
        )

    if args.output_json:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "description": (
                "Bridge run comparing hardcoded primitive Hubbard builder + hardcoded VQE wavevector "
                "to Qiskit-Nature Hamiltonian under Qiskit built-in exact and Suzuki/Lie-Trotter evolution."
            ),
            "settings": {
                "num_sites": DEFAULT_NUM_SITES,
                "hopping_t": args.hopping_t,
                "onsite_u": args.onsite_u,
                "boundary": args.boundary,
                "spin_orbital_ordering": args.ordering,
                "t_final": args.t_final,
                "num_times": args.num_times,
                "suzuki_order": args.suzuki_order,
                "trotter_steps": args.trotter_steps,
                "hardcoded_repo_root": args.hardcoded_repo_root,
                "hardcoded_reps": args.hardcoded_reps,
                "hardcoded_restarts": args.hardcoded_restarts,
                "hardcoded_seed": args.hardcoded_seed,
                "hardcoded_maxiter": args.hardcoded_maxiter,
            },
            "hardcoded_vqe": {
                "quantum_dir": hardcoded.quantum_dir,
                "notebook": hardcoded.notebook_path,
                "energy": hardcoded.energy,
                "theta": hardcoded.theta,
                "theta_len": len(hardcoded.theta),
                "hf_bitstring": hardcoded.hf_bitstring,
                "num_particles": {
                    "n_up": int(hardcoded.num_particles[0]),
                    "n_dn": int(hardcoded.num_particles[1]),
                },
                "fidelity_vs_hf_initial": fid_vqe_vs_hf_initial,
            },
            "hamiltonian_comparison": h_delta,
            "hamiltonian_jw_terms_hardcoded": _sparse_pauli_to_jsonable(h_hardcoded),
            "hamiltonian_jw_terms_qiskit": _sparse_pauli_to_jsonable(h_qiskit),
            "trajectory": trajectory_json,
            "observables": {
                "n_up_site0_terms": _sparse_pauli_to_jsonable(n_up_site0),
                "doublon_terms": _sparse_pauli_to_jsonable(doublon),
            },
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
        print(f"Wrote JSON results to: {args.output_json}")


if __name__ == "__main__":
    main()
