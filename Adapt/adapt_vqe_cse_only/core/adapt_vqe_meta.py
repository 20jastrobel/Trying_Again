"""Meta-ADAPT-VQE implementation using a coordinate-wise LSTM optimizer."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

from .meta_lstm_optimizer import CoordinateWiseLSTMOptimizer, LSTMState
from .meta_lstm import preprocess_gradients_torch
from .symmetry import commutes, map_symmetry_ops_to_qubits

try:  # Qiskit >=1.0
    from qiskit.primitives import BaseEstimatorV2
except ImportError:  # Qiskit 0.x compatibility (type-only fallback)
    BaseEstimatorV2 = object  # type: ignore[assignment]

HARDCODED_ONZOT_POOL_MODES = {"hardcoded_uccsd_onzots", "uccsd_onzots"}
GROUPED_POOL_MODES = {"cse_density_ops", "uccsd_excitations"} | HARDCODED_ONZOT_POOL_MODES


@dataclass
class MetaAdaptVQEResult:
    energy: float
    params: list[float]
    operators: list
    diagnostics: dict


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str, last_energy: float | None = None):
        self.reason = reason
        self.last_energy = last_energy
        super().__init__(f"Budget exceeded: {reason}")


def _canonicalize_to_hermitian(op: SparsePauliOp, *, atol: float) -> SparsePauliOp:
    """Return a numerically Hermitian SparsePauliOp from a possibly non-Hermitian input.

    If the operator is closer to anti-Hermitian than Hermitian, rotate by -i so
    the returned operator is Hermitian. This is used to avoid silently dropping
    Pauli terms with nontrivial imaginary components when building grouped pools.
    """
    op = op.simplify(atol=atol)
    herm = 0.5 * (op + op.adjoint())
    anti = 0.5 * (op - op.adjoint())
    herm_norm = float(np.sum(np.abs(herm.coeffs))) if len(herm.coeffs) else 0.0
    anti_norm = float(np.sum(np.abs(anti.coeffs))) if len(anti.coeffs) else 0.0
    if anti_norm > herm_norm:
        op = (-1j) * anti
    else:
        op = herm
    return op.simplify(atol=atol)


def _grouped_spec_from_sparse_pauli_op(
    op: SparsePauliOp,
    *,
    name: str,
    dedup_tol: float,
    seen: set[tuple] | None = None,
) -> dict | None:
    """Convert an operator to the grouped {name, paulis} spec used by grouped ADAPT pools.

    The conversion:
    - removes identity components,
    - coerces numerically-real coefficients to real weights,
    - rejects truly complex coefficients after Hermitian canonicalization,
    - drops near-zero weights and deduplicates based on a rounded signature.
    """
    if op is None:
        return None

    op = _canonicalize_to_hermitian(op, atol=dedup_tol)
    identity = "I" * op.num_qubits

    paulis: list[tuple[str, float]] = []
    for lbl, coeff in zip(op.paulis.to_labels(), op.coeffs):
        if lbl == identity:
            continue
        if abs(coeff.imag) > dedup_tol:
            raise ValueError(
                f"Non-real Pauli coefficient after Hermitian canonicalization in {name}: {coeff}."
            )
        weight = float(coeff.real)
        if abs(weight) <= dedup_tol:
            continue
        paulis.append((lbl, weight))

    if not paulis:
        return None

    key = tuple((lbl, round(w, 12)) for lbl, w in sorted(paulis, key=lambda item: item[0]))
    if seen is not None:
        if key in seen:
            return None
        seen.add(key)
    return {"name": name, "paulis": paulis}


def build_operator_pool_from_hamiltonian(
    qubit_op: SparsePauliOp,
    *,
    mode: str = "ham_terms_plus_imag_partners",
) -> list[str]:
    """Return a Pauli-string operator pool derived from the Hamiltonian."""
    identity = "I" * qubit_op.num_qubits
    seen: set[str] = set()
    base_pool: list[str] = []
    for label in qubit_op.paulis.to_labels():
        if label == identity:
            continue
        if label not in seen:
            seen.add(label)
            base_pool.append(label)

    if mode == "ham_terms":
        return base_pool
    if mode != "ham_terms_plus_imag_partners":
        raise ValueError(f"Unknown pool mode: {mode}")

    pool = list(base_pool)
    pool_set = set(pool)
    for label in base_pool:
        for partner in _imaginary_partners(label):
            if partner != identity and partner not in pool_set:
                pool_set.add(partner)
                pool.append(partner)
    # Add single-qubit X/Y terms to enrich the pool and avoid trivial stagnation.
    for q in range(qubit_op.num_qubits):
        for pauli in ("X", "Y"):
            label = ["I"] * qubit_op.num_qubits
            label[q] = pauli
            term = "".join(label)
            if term not in pool_set:
                pool_set.add(term)
                pool.append(term)
    return pool


def build_cse_density_pool_from_fermionic(
    ferm_op: FermionicOp,
    mapper: JordanWignerMapper,
    *,
    include_diagonal: bool = False,
    include_antihermitian_part: bool = True,
    include_hermitian_part: bool = True,
    dedup_tol: float = 1e-12,
    enforce_symmetry: bool = False,
    n_sites: int | None = None,
) -> list[dict]:
    """Return CSE-inspired density operator pool from a fermionic Hamiltonian."""
    if ferm_op is None:
        raise ValueError("ferm_op must be provided for cse_density_ops pool.")

    pool: list[dict] = []
    seen: set[tuple] = set()

    for term_ops in _iter_significant_fermionic_terms(ferm_op, tol=dedup_tol):
        label = _fermionic_ops_to_label(term_ops)

        if len(term_ops) == 2 and term_ops[0][0] == "+" and term_ops[1][0] == "-":
            p = int(term_ops[0][1])
            q = int(term_ops[1][1])
            if p == q and not include_diagonal:
                continue
            if enforce_symmetry and n_sites is not None:
                if not _fermionic_ops_in_sector(term_ops, int(n_sites)):
                    continue
            gamma = FermionicOp({label: 1.0}, num_spin_orbitals=ferm_op.num_spin_orbitals)
        elif include_diagonal and len(term_ops) == 4:
            if enforce_symmetry and n_sites is not None:
                if not _fermionic_ops_in_sector(term_ops, int(n_sites)):
                    continue
            gamma = FermionicOp({label: 1.0}, num_spin_orbitals=ferm_op.num_spin_orbitals)
        else:
            continue

        if include_antihermitian_part:
            # Imag/current-like quadrature: A = i (gamma - gamma^\dagger) (Hermitian).
            k_op = gamma - gamma.adjoint()
            if not k_op.is_zero():
                qubit_k = mapper.map(k_op)
                if qubit_k is not None:
                    op_im = (-1j) * qubit_k
                    spec = _grouped_spec_from_sparse_pauli_op(
                        op_im,
                        name=f"gamma_im({label})",
                        dedup_tol=dedup_tol,
                        seen=seen,
                    )
                    if spec is not None:
                        pool.append(spec)

        if include_hermitian_part:
            # Real/hopping-like quadrature: B = gamma + gamma^\dagger (Hermitian).
            h_op = gamma + gamma.adjoint()
            if not h_op.is_zero():
                qubit_h = mapper.map(h_op)
                if qubit_h is not None:
                    spec = _grouped_spec_from_sparse_pauli_op(
                        qubit_h,
                        name=f"gamma_re({label})",
                        dedup_tol=dedup_tol,
                        seen=seen,
                    )
                    if spec is not None:
                        pool.append(spec)

    return pool


def build_uccsd_excitation_pool(
    *,
    n_sites: int,
    num_particles: tuple[int, int],
    mapper: JordanWignerMapper,
    reps: int = 1,
    include_imaginary: bool = False,
    preserve_spin: bool = True,
    generalized: bool = False,
    dedup_tol: float = 1e-12,
) -> list[dict]:
    """Return grouped Pauli operators from UCCSD excitation generators."""
    if mapper is None:
        raise ValueError("mapper must be provided for UCCSD excitation pool.")

    try:
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    except ImportError:  # pragma: no cover
        from qiskit_nature.circuit.library import HartreeFock, UCCSD

    try:
        initial_state = HartreeFock(
            num_spatial_orbitals=int(n_sites),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            qubit_mapper=mapper,
        )
    except TypeError:
        initial_state = HartreeFock(int(n_sites), (int(num_particles[0]), int(num_particles[1])), mapper)

    try:
        ansatz = UCCSD(
            num_spatial_orbitals=int(n_sites),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            qubit_mapper=mapper,
            reps=int(reps),
            initial_state=initial_state,
            generalized=generalized,
            preserve_spin=preserve_spin,
            include_imaginary=include_imaginary,
        )
    except TypeError:
        ansatz = UCCSD(
            int(n_sites),
            (int(num_particles[0]), int(num_particles[1])),
            mapper,
            reps=int(reps),
            initial_state=initial_state,
        )

    operators = list(getattr(ansatz, "operators", []))
    excitation_list = list(getattr(ansatz, "excitation_list", []))

    pool: list[dict] = []
    seen: set[tuple] = set()
    for idx, op in enumerate(operators):
        exc = excitation_list[idx] if idx < len(excitation_list) else None
        name = f"uccsd_exc({exc})" if exc is not None else f"uccsd_exc_{idx}"
        spec = _grouped_spec_from_sparse_pauli_op(
            op,
            name=name,
            dedup_tol=dedup_tol,
            seen=seen,
        )
        if spec is not None:
            pool.append(spec)

    return pool


def _exyz_to_ixyz(label: str) -> str:
    return str(label).translate(str.maketrans({"e": "I", "x": "X", "y": "Y", "z": "Z"}))


def _sparse_pauli_op_from_hardcoded_polynomial(polynomial, *, tol: float) -> SparsePauliOp | None:
    terms: list[tuple[str, complex]] = []
    for term in polynomial.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        label_exyz = str(term.pw2strng())
        terms.append((_exyz_to_ixyz(label_exyz), coeff))
    if not terms:
        return None
    return SparsePauliOp.from_list(terms).simplify(atol=float(tol))


def build_hardcoded_uccsd_onzot_pool(
    *,
    n_sites: int,
    num_particles: tuple[int, int],
    indexing: str = "blocked",
    include_singles: bool = True,
    include_doubles: bool = True,
    dedup_tol: float = 1e-12,
) -> list[dict]:
    """Return grouped pool specs from canonical hardcoded UCCSD ONZOT generators."""
    try:
        from src.quantum.vqe_latex_python_pairs import HardcodedUCCSDAnsatz
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.quantum.vqe_latex_python_pairs import HardcodedUCCSDAnsatz

    ansatz = HardcodedUCCSDAnsatz(
        dims=int(n_sites),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        reps=1,
        repr_mode="JW",
        indexing=str(indexing),
        include_singles=bool(include_singles),
        include_doubles=bool(include_doubles),
    )

    pool: list[dict] = []
    seen: set[tuple] = set()
    for idx, term in enumerate(getattr(ansatz, "base_terms", [])):
        op = _sparse_pauli_op_from_hardcoded_polynomial(term.polynomial, tol=dedup_tol)
        if op is None:
            continue
        name = str(getattr(term, "label", f"hardcoded_uccsd_onzot_{idx}"))
        spec = _grouped_spec_from_sparse_pauli_op(
            op,
            name=f"onzot({name})",
            dedup_tol=dedup_tol,
            seen=seen,
        )
        if spec is not None:
            pool.append(spec)
    return pool


def _iter_significant_fermionic_terms(
    ferm_op: FermionicOp,
    *,
    tol: float,
) -> list[list[tuple[str, int]]]:
    out: list[list[tuple[str, int]]] = []
    for term_ops, coeff in ferm_op.terms():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= float(tol):
            continue
        parsed = [(str(action), int(idx)) for action, idx in term_ops]
        if not parsed:
            continue
        out.append(parsed)
    return out


def _fermionic_ops_to_label(term_ops: Sequence[tuple[str, int]]) -> str:
    return " ".join(f"{action}_{int(idx)}" for action, idx in term_ops)


def _parse_fermionic_label(label: str) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for token in str(label).split():
        action, idx = token.split("_", 1)
        parsed.append((action, int(idx)))
    return parsed


def _fermionic_ops_in_sector(term_ops: Sequence[tuple[str, int]], n_sites: int) -> bool:
    delta_n = 0
    delta_sz = 0.0
    for action, idx in term_ops:
        idx_i = int(idx)
        if action == "+":
            delta_n += 1
            delta_sz += 0.5 if idx_i < int(n_sites) else -0.5
        elif action == "-":
            delta_n -= 1
            delta_sz -= 0.5 if idx_i < int(n_sites) else -0.5
    return delta_n == 0 and abs(delta_sz) < 1e-12


def _fermionic_term_in_sector(label: str, n_sites: int) -> bool:
    return _fermionic_ops_in_sector(_parse_fermionic_label(label), int(n_sites))


def build_reference_state(num_qubits: int, occupations: Sequence[int]) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for idx in occupations:
        if idx < 0 or idx >= num_qubits:
            raise ValueError(f"Occupation index out of range: {idx}")
        qc.x(idx)
    return qc


def build_adapt_circuit(
    reference_state: QuantumCircuit,
    operators: Sequence[str],
) -> tuple[QuantumCircuit, ParameterVector]:
    """Build the ADAPT circuit with parameter placeholders."""
    num_qubits = reference_state.num_qubits
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(reference_state, inplace=True)

    params = ParameterVector("theta", len(operators))
    for idx, label in enumerate(operators):
        pauli = Pauli(label)
        gate = PauliEvolutionGate(pauli, time=params[idx] / 2)
        circuit.append(gate, list(range(num_qubits)))

    return circuit, params


def build_adapt_circuit_grouped(
    reference_state: QuantumCircuit,
    operator_specs: Sequence[dict],
    *,
    trotter_reps: int = 1,
    trotter_order: int = 1,
    generator_scale: float = 0.5,
    grouped_term_order: str = "as_given",
) -> tuple[QuantumCircuit, ParameterVector]:
    """Build ADAPT circuit from grouped generator specs.

    The grouped generator implemented for each parameter is:
        exp(-i * generator_scale * theta_k * sum_j w_j P_j)

    Defaults preserve legacy grouped behavior (generator_scale=0.5). For
    hardcoded UCCSD compatibility, run_meta_adapt_vqe selects generator_scale=1.0.
    """
    num_qubits = reference_state.num_qubits
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(reference_state, inplace=True)

    reps = max(1, int(trotter_reps))
    scale = float(generator_scale)
    if trotter_order not in (1, 2):
        raise ValueError(f"Unsupported grouped trotter_order={trotter_order}; expected 1 or 2.")

    all_qubits = list(range(num_qubits))
    params = ParameterVector("theta", len(operator_specs))
    for idx, spec in enumerate(operator_specs):
        paulis = list(spec["paulis"])
        if grouped_term_order == "canonical_sorted":
            paulis = sorted(paulis, key=lambda item: item[0])
        elif grouped_term_order != "as_given":
            raise ValueError(
                f"Unsupported grouped_term_order={grouped_term_order}; "
                "expected 'as_given' or 'canonical_sorted'."
            )

        if trotter_order == 1:
            dt = (scale * params[idx]) / reps
            for _ in range(reps):
                for label, weight in paulis:
                    pauli = Pauli(label)
                    gate = PauliEvolutionGate(pauli, time=dt * weight)
                    circuit.append(gate, all_qubits)
        else:
            dt = (scale * params[idx]) / (2.0 * reps)
            for _ in range(reps):
                for label, weight in paulis:
                    pauli = Pauli(label)
                    gate = PauliEvolutionGate(pauli, time=dt * weight)
                    circuit.append(gate, all_qubits)
                for label, weight in reversed(paulis):
                    pauli = Pauli(label)
                    gate = PauliEvolutionGate(pauli, time=dt * weight)
                    circuit.append(gate, all_qubits)

    return circuit, params


def estimate_energy(
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    qubit_op: SparsePauliOp,
    params: Sequence[float],
) -> float:
    return _estimate_expectation(estimator, circuit, qubit_op, params)


def estimate_expectation(
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    op: SparsePauliOp,
    params: Sequence[float],
) -> float:
    return _estimate_expectation(estimator, circuit, op, params)


def _extract_expectation_value(result: object) -> float | None:
    # Qiskit v2-style primitive result.
    try:
        pub0 = result[0]  # type: ignore[index]
    except Exception:
        pub0 = None
    if pub0 is not None:
        data = getattr(pub0, "data", None)
        if data is not None and hasattr(data, "evs"):
            arr = np.asarray(getattr(data, "evs"))
            if arr.size:
                return float(np.real(arr.reshape(-1)[0]))

    # Qiskit v1-style estimator result.
    values = getattr(result, "values", None)
    if values is not None:
        arr = np.asarray(values)
        if arr.size:
            return float(np.real(arr.reshape(-1)[0]))

    # Fallback for custom wrappers that expose evs directly.
    evs = getattr(result, "evs", None)
    if evs is not None:
        arr = np.asarray(evs)
        if arr.size:
            return float(np.real(arr.reshape(-1)[0]))
    return None


def _estimate_expectation(
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    op: SparsePauliOp,
    params: Sequence[float],
) -> float:
    params_list = list(params)

    # Prefer the modern pub-based Estimator V2 API.
    try:
        job = estimator.run([(circuit, op, params_list)])
        ev = _extract_expectation_value(job.result())
        if ev is not None:
            return ev
    except (TypeError, ValueError):
        pass

    # Fallback to Estimator V1 API.
    job = estimator.run([circuit], [op], [params_list])
    ev = _extract_expectation_value(job.result())
    if ev is None:
        raise RuntimeError("Could not extract expectation value from estimator result.")
    return ev


def compute_sector_diagnostics(
    *,
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    params: Sequence[float],
    mapper: JordanWignerMapper,
    n_sites: int,
    n_target: int,
    sz_target: float,
    simplify_atol: float = 1e-12,
) -> dict:
    """Compute N/Sz moments and simple sector leakage diagnostics for a circuit state."""
    n_q, sz_q = map_symmetry_ops_to_qubits(mapper, int(n_sites))

    n_mean = estimate_expectation(estimator, circuit, n_q, params)
    sz_mean = estimate_expectation(estimator, circuit, sz_q, params)

    n2_q = (n_q @ n_q).simplify(atol=simplify_atol)
    sz2_q = (sz_q @ sz_q).simplify(atol=simplify_atol)

    n2_mean = estimate_expectation(estimator, circuit, n2_q, params)
    sz2_mean = estimate_expectation(estimator, circuit, sz2_q, params)

    var_n = float(max(0.0, n2_mean - n_mean * n_mean))
    var_sz = float(max(0.0, sz2_mean - sz_mean * sz_mean))

    return {
        "N_mean": float(n_mean),
        "Sz_mean": float(sz_mean),
        "VarN": var_n,
        "VarSz": var_sz,
        "abs_N_err": float(abs(n_mean - float(n_target))),
        "abs_Sz_err": float(abs(sz_mean - float(sz_target))),
    }


def parameter_shift_grad(
    energy_fn,
    theta: np.ndarray,
    *,
    shift: float = math.pi / 2,
) -> np.ndarray:
    n = theta.size
    grad = np.zeros(n, dtype=float)
    for idx in range(n):
        plus = theta.copy()
        minus = theta.copy()
        plus[idx] += shift
        minus[idx] -= shift
        grad[idx] = 0.5 * (energy_fn(plus) - energy_fn(minus))
    return grad


def finite_difference_grad(energy_fn, theta: np.ndarray, *, eps: float) -> np.ndarray:
    n = theta.size
    grad = np.zeros(n, dtype=float)
    for idx in range(n):
        plus = theta.copy()
        minus = theta.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (energy_fn(plus) - energy_fn(minus)) / (2.0 * eps)
    return grad


def _append_probe_gate(
    circuit: QuantumCircuit,
    label: str,
    *,
    angle: float,
) -> QuantumCircuit:
    num_qubits = circuit.num_qubits
    probe = QuantumCircuit(num_qubits)
    probe.compose(circuit, inplace=True)
    gate = PauliEvolutionGate(Pauli(label), time=angle / 2)
    probe.append(gate, list(range(num_qubits)))
    return probe


def _imaginary_partners(label: str) -> list[str]:
    partners: list[str] = []
    chars = list(label)
    for idx, char in enumerate(chars):
        if char == "X":
            partner = chars.copy()
            partner[idx] = "Y"
            partners.append("".join(partner))
        elif char == "Y":
            partner = chars.copy()
            partner[idx] = "X"
            partners.append("".join(partner))
    return partners


def _wrap_angles(theta: np.ndarray) -> np.ndarray:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def _probe_shift_and_coeff(convention: str) -> tuple[float, float]:
    if convention == "half_angle":
        return math.pi / 2, 0.5
    if convention == "full_angle":
        return math.pi / 4, 1.0
    raise ValueError(f"Unknown probe convention: {convention}")


def _compute_grouped_pool_gradients(
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    qubit_op: SparsePauliOp,
    theta: np.ndarray,
    pool_specs: Sequence[dict],
    *,
    probe_convention: str,
    energy_eval=None,
) -> list[float]:
    """Legacy probe-based grouped gradient (kept for debug comparison only)."""
    shift, coeff = _probe_shift_and_coeff(probe_convention)
    label_set: set[str] = set()
    for spec in pool_specs:
        for label, _weight in spec["paulis"]:
            label_set.add(label)

    g_cache: dict[str, float] = {}
    for label in sorted(label_set):
        probe_plus = _append_probe_gate(circuit, label, angle=shift)
        probe_minus = _append_probe_gate(circuit, label, angle=-shift)
        if energy_eval is None:
            e_plus = estimate_energy(estimator, probe_plus, qubit_op, theta)
            e_minus = estimate_energy(estimator, probe_minus, qubit_op, theta)
        else:
            e_plus = energy_eval(probe_plus, theta)
            e_minus = energy_eval(probe_minus, theta)
        g_cache[label] = coeff * (e_plus - e_minus)

    gradients: list[float] = []
    for spec in pool_specs:
        g_val = 0.0
        for label, weight in spec["paulis"]:
            g_val += weight * g_cache[label]
        gradients.append(g_val)
    return gradients


def _compute_grouped_pool_gradients_fd(
    estimator: BaseEstimatorV2,
    reference_state: QuantumCircuit,
    qubit_op: SparsePauliOp,
    current_ops: Sequence[dict],
    theta: np.ndarray,
    pool_specs: Sequence[dict],
    *,
    delta: float = 1e-5,
    trotter_reps: int = 1,
    trotter_order: int = 1,
    generator_scale: float = 0.5,
    grouped_term_order: str = "as_given",
    energy_eval_fn=None,
) -> list[float]:
    """Exact directional finite-difference gradient for grouped ADAPT selection.

    For each candidate grouped operator, builds a trial circuit with
    current_ops + [candidate] and evaluates:
        grad ≈ (E(theta_aug + delta*e_last) - E(theta_aug - delta*e_last)) / (2*delta)
    where theta_aug = [theta..., 0] and e_last is the unit vector for the new parameter.
    """
    gradients: list[float] = []
    theta_arr = np.asarray(theta, dtype=float)

    for spec in pool_specs:
        trial_ops = list(current_ops) + [spec]
        trial_circuit, _trial_params = build_adapt_circuit_grouped(
            reference_state,
            trial_ops,
            trotter_reps=int(trotter_reps),
            trotter_order=int(trotter_order),
            generator_scale=float(generator_scale),
            grouped_term_order=str(grouped_term_order),
        )
        theta_plus = np.concatenate([theta_arr, [float(delta)]])
        theta_minus = np.concatenate([theta_arr, [-float(delta)]])

        if energy_eval_fn is not None:
            e_plus = energy_eval_fn(trial_circuit, theta_plus)
            e_minus = energy_eval_fn(trial_circuit, theta_minus)
        else:
            e_plus = estimate_energy(estimator, trial_circuit, qubit_op, theta_plus)
            e_minus = estimate_energy(estimator, trial_circuit, qubit_op, theta_minus)

        grad_fd = (e_plus - e_minus) / (2.0 * float(delta))
        gradients.append(float(grad_fd))

    return gradients


def _maybe_import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Torch is required for meta/hybrid inner optimization. "
            "Install torch or choose --inner-optimizer lbfgs/slsqp."
        ) from exc
    return torch


def _lbfgs_optimize(
    theta0: np.ndarray,
    energy_fn,
    grad_fn,
    *,
    theta_bound: str = "none",
    maxiter: int | None = None,
    restarts: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    bounds = None
    if theta_bound == "pi":
        bounds = [(-math.pi, math.pi) for _ in range(theta0.size)]
    elif theta_bound != "none":
        raise ValueError(f"Unknown theta bound: {theta_bound}")

    if restarts < 1:
        restarts = 1

    best_x = None
    best_fun = None
    total_nfev = 0
    total_njev = 0
    best_stats = {}

    for idx in range(restarts):
        if idx == 0:
            x0 = theta0
        else:
            if rng is None:
                rng = np.random.default_rng()
            x0 = rng.uniform(-math.pi, math.pi, size=theta0.size)

        result = minimize(
            fun=lambda x: energy_fn(x),
            x0=x0,
            jac=lambda x: grad_fn(x),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter} if maxiter is not None else None,
        )

        total_nfev += int(result.nfev)
        total_njev += int(result.njev) if result.njev is not None else 0

        if best_fun is None or result.fun < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x, dtype=float)
            best_stats = {
                "status": int(result.status),
                "success": bool(result.success),
                "message": str(result.message),
                "grad_norm": float(np.linalg.norm(result.jac)) if result.jac is not None else None,
            }

    stats = {
        "nfev": total_nfev,
        "njev": total_njev,
        "restarts": restarts,
        **best_stats,
    }
    return best_x if best_x is not None else theta0, stats


def _slsqp_optimize(
    theta0: np.ndarray,
    energy_fn,
    *,
    theta_bound: str = "none",
    maxiter: int | None = None,
    restarts: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    bounds = None
    if theta_bound == "pi":
        bounds = [(-math.pi, math.pi) for _ in range(theta0.size)]
    elif theta_bound != "none":
        raise ValueError(f"Unknown theta bound: {theta_bound}")

    if restarts < 1:
        restarts = 1

    best_x = None
    best_fun = None
    total_nfev = 0
    total_njev = 0
    best_stats = {}

    for idx in range(restarts):
        if idx == 0:
            x0 = theta0
        else:
            if rng is None:
                rng = np.random.default_rng()
            x0 = rng.uniform(-math.pi, math.pi, size=theta0.size)

        result = minimize(
            fun=lambda x: energy_fn(x),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": maxiter} if maxiter is not None else None,
        )

        total_nfev += int(getattr(result, "nfev", 0))
        njev_val = getattr(result, "njev", None)
        if njev_val is not None:
            total_njev += int(njev_val)

        if best_fun is None or result.fun < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x, dtype=float)
            best_stats = {
                "status": int(result.status),
                "success": bool(result.success),
                "message": str(result.message),
                "grad_norm": float(np.linalg.norm(result.jac))
                if getattr(result, "jac", None) is not None
                else None,
            }

    stats = {
        "nfev": total_nfev,
        "njev": total_njev,
        "restarts": restarts,
        **best_stats,
    }
    return best_x if best_x is not None else theta0, stats


def _meta_optimize(
    theta0: np.ndarray,
    energy_fn,
    grad_fn,
    *,
    steps: int,
    model,
    r: float,
    step_scale: float,
    dtheta_clip: float | None,
    alpha0: float = 1.0,
    alpha_k: float = 0.0,
    verbose: bool,
) -> tuple[np.ndarray, dict]:
    torch = _maybe_import_torch()
    device = next(model.parameters()).device
    model.eval()

    theta = theta0.copy()
    n_params = theta.size
    if n_params == 0:
        return theta, {"avg_delta_norm": 0.0}

    h = torch.zeros((n_params, model.hidden_size), device=device)
    c = torch.zeros((n_params, model.hidden_size), device=device)

    delta_norms: list[float] = []
    max_delta_vals: list[float] = []
    energies: list[float] = []

    for inner_idx in range(steps):
        energy = energy_fn(theta)
        grad = grad_fn(theta)

        x = preprocess_gradients_torch(grad, r=r, device=device)
        with torch.no_grad():
            delta, (h, c) = model(x, (h, c))
        delta = delta.cpu().numpy().astype(float)

        if dtheta_clip is not None:
            delta = np.clip(delta, -dtheta_clip, dtheta_clip)
        alpha = alpha0 * (1.0 + alpha_k * (inner_idx / max(1, steps - 1)))
        delta = alpha * step_scale * delta

        theta = theta + delta

        grad_norm = float(np.linalg.norm(grad))
        delta_norm = float(np.linalg.norm(delta))
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        energies.append(energy)
        delta_norms.append(delta_norm)
        max_delta_vals.append(max_delta)

        if verbose:
            print(
                f"  inner {inner_idx + 1}/{steps}: "
                f"E={energy:.10f}, ||g||={grad_norm:.3e}, "
                f"||Δθ||={delta_norm:.3e}, max|Δθ|={max_delta:.3e}"
            )

    stats = {
        "avg_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
        "avg_max_delta": float(np.mean(max_delta_vals)) if max_delta_vals else 0.0,
        "final_energy": energies[-1] if energies else energy_fn(theta),
    }
    return theta, stats


def _hybrid_optimize(
    theta0: np.ndarray,
    energy_fn,
    grad_fn,
    *,
    model,
    r: float,
    step_scale: float,
    dtheta_clip: float | None,
    warmup_steps: int,
    theta_bound: str,
    lbfgs_maxiter: int | None,
    lbfgs_restarts: int,
    verbose: bool,
) -> tuple[np.ndarray, dict]:
    theta_meta, meta_stats = _meta_optimize(
        theta0,
        energy_fn,
        grad_fn,
        steps=warmup_steps,
        model=model,
        r=r,
        step_scale=step_scale,
        dtheta_clip=dtheta_clip,
        verbose=verbose,
    )
    theta_lbfgs, lbfgs_stats = _lbfgs_optimize(
        theta_meta,
        energy_fn,
        grad_fn,
        theta_bound=theta_bound,
        maxiter=lbfgs_maxiter,
        restarts=lbfgs_restarts,
    )
    stats = {
        "meta": meta_stats,
        "lbfgs": lbfgs_stats,
    }
    return theta_lbfgs, stats


def debug_validate_hardcoded_uccsd_semantics(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    mapper,
    reference_state: QuantumCircuit,
    build_grouped_circuit_fn,
    pool_specs: Sequence[dict],
    grouped_term_order: str,
    grouped_trotter_order: int,
    grouped_trotter_reps: int,
    grouped_generator_scale: float = 1.0,
    hardcoded_indexing: str = "blocked",
    hardcoded_include_singles: bool = True,
    hardcoded_include_doubles: bool = True,
    n_trials: int = 10,
    seed: int = 0,
    theta_std: float = 0.3,
    atol: float = 1e-10,
    verbose: bool = True,
) -> dict:
    """Validate grouped hardcoded-UCCSD circuit semantics against canonical state prep."""
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError("Semantics validator currently requires JordanWignerMapper.")

    try:
        from src.quantum import vqe_latex_python_pairs as vqe_mod
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.quantum import vqe_latex_python_pairs as vqe_mod

    n_sites_i = int(n_sites)
    n_up_i = int(n_up)
    n_down_i = int(n_down)
    indexing = str(hardcoded_indexing)

    hf_bitstring = vqe_mod.hartree_fock_bitstring(
        n_sites_i,
        (n_up_i, n_down_i),
        indexing=indexing,
    )
    psi_ref = np.asarray(vqe_mod.basis_state(2 * n_sites_i, hf_bitstring), dtype=complex)
    psi_ref_qiskit = Statevector.from_instruction(reference_state).data
    ref_overlap = np.vdot(psi_ref, psi_ref_qiskit)
    ref_fidelity = float(abs(ref_overlap) ** 2)
    if ref_fidelity < 1.0 - float(atol):
        raise RuntimeError(
            "Reference-state mismatch in semantics validator: "
            f"fidelity={ref_fidelity:.12f} < {1.0 - float(atol):.12f}."
        )

    ansatz = vqe_mod.HardcodedUCCSDAnsatz(
        dims=n_sites_i,
        num_particles=(n_up_i, n_down_i),
        reps=1,
        repr_mode="JW",
        indexing=indexing,
        include_singles=bool(hardcoded_include_singles),
        include_doubles=bool(hardcoded_include_doubles),
    )
    n_params = len(pool_specs)
    if int(ansatz.num_parameters) != n_params:
        raise RuntimeError(
            "Semantics validator parameter-count mismatch between canonical hardcoded ansatz "
            f"({int(ansatz.num_parameters)}) and grouped pool specs ({n_params})."
        )

    circuit, params = build_grouped_circuit_fn(
        reference_state,
        pool_specs,
        trotter_reps=int(grouped_trotter_reps),
        trotter_order=int(grouped_trotter_order),
        generator_scale=float(grouped_generator_scale),
        grouped_term_order=str(grouped_term_order),
    )
    if len(params) != n_params:
        raise RuntimeError(
            "Grouped circuit parameter count does not match pool spec count in semantics validator."
        )

    rng = np.random.default_rng(int(seed))
    fidelities: list[float] = []
    min_fidelity = 1.0
    failing_trial = -1

    for trial_idx in range(max(1, int(n_trials))):
        theta = rng.normal(scale=float(theta_std), size=n_params).astype(float)
        psi_canon = np.asarray(ansatz.prepare_state(theta, psi_ref, sort_terms=True), dtype=complex)
        bound = circuit.assign_parameters(
            {params[i]: float(theta[i]) for i in range(n_params)},
            inplace=False,
        )
        psi_grouped = Statevector.from_instruction(bound).data
        overlap = np.vdot(psi_canon, psi_grouped)
        fidelity = float(abs(overlap) ** 2)
        fidelities.append(fidelity)
        if fidelity < min_fidelity:
            min_fidelity = fidelity
            failing_trial = trial_idx

    result = {
        "n_trials": int(max(1, int(n_trials))),
        "seed": int(seed),
        "theta_std": float(theta_std),
        "grouped_term_order": str(grouped_term_order),
        "grouped_trotter_order": int(grouped_trotter_order),
        "grouped_trotter_reps": int(grouped_trotter_reps),
        "grouped_generator_scale": float(grouped_generator_scale),
        "reference_fidelity": ref_fidelity,
        "min_fidelity": float(min_fidelity),
        "mean_fidelity": float(np.mean(fidelities)) if fidelities else float("nan"),
        "failing_trial": int(failing_trial),
    }
    if verbose:
        print(
            "Hardcoded-UCCSD semantics check: "
            f"min fidelity={result['min_fidelity']:.12f}, "
            f"mean fidelity={result['mean_fidelity']:.12f}"
        )
    if min_fidelity < 1.0 - float(atol):
        raise RuntimeError(
            "Hardcoded-UCCSD semantics validation failed: "
            f"min fidelity={min_fidelity:.12f} < {1.0 - float(atol):.12f} "
            f"(trial={failing_trial}, term_order={grouped_term_order}, "
            f"trotter_order={grouped_trotter_order}, reps={grouped_trotter_reps})."
        )
    return result


def debug_validate_grouped_outer_gradients(
    *,
    estimator: BaseEstimatorV2,
    qubit_op: SparsePauliOp,
    reference_state: QuantumCircuit,
    current_ops: Sequence[dict],
    theta: np.ndarray,
    pool_specs: Sequence[dict],
    probe_convention: str,
    grouped_trotter_order: int,
    grouped_trotter_reps: int,
    grouped_generator_scale: float,
    grouped_term_order: str,
    n_checks: int = 5,
    delta: float = 1e-5,
    rtol: float = 1e-3,
    atol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """Compare grouped probe gradients against directional finite differences."""
    circuit, _params = build_adapt_circuit_grouped(
        reference_state,
        current_ops,
        trotter_reps=int(grouped_trotter_reps),
        trotter_order=int(grouped_trotter_order),
        generator_scale=float(grouped_generator_scale),
        grouped_term_order=str(grouped_term_order),
    )
    theta_arr = np.asarray(theta, dtype=float)
    probe_grads = _compute_grouped_pool_gradients(
        estimator,
        circuit,
        qubit_op,
        theta_arr,
        pool_specs,
        probe_convention=probe_convention,
    )

    ranked_indices = np.argsort(np.abs(np.asarray(probe_grads, dtype=float)))[::-1]
    checks = max(1, min(int(n_checks), len(pool_specs)))
    chosen_indices = ranked_indices[:checks]
    delta_f = float(delta)
    if delta_f <= 0.0:
        raise ValueError("delta must be positive for grouped outer-gradient validation.")

    details = []
    max_rel_err = 0.0

    for idx in chosen_indices:
        spec = pool_specs[int(idx)]
        trial_ops = list(current_ops) + [spec]
        trial_circuit, _trial_params = build_adapt_circuit_grouped(
            reference_state,
            trial_ops,
            trotter_reps=int(grouped_trotter_reps),
            trotter_order=int(grouped_trotter_order),
            generator_scale=float(grouped_generator_scale),
            grouped_term_order=str(grouped_term_order),
        )

        theta_plus = np.concatenate([theta_arr, [delta_f]])
        theta_minus = np.concatenate([theta_arr, [-delta_f]])
        e_plus = estimate_energy(estimator, trial_circuit, qubit_op, theta_plus)
        e_minus = estimate_energy(estimator, trial_circuit, qubit_op, theta_minus)
        grad_fd = (e_plus - e_minus) / (2.0 * delta_f)
        grad_probe = float(probe_grads[int(idx)])
        abs_err = abs(grad_probe - grad_fd)
        denom = max(float(atol), abs(grad_probe), abs(grad_fd))
        rel_err = abs_err / denom
        max_rel_err = max(max_rel_err, rel_err)
        details.append(
            {
                "pool_index": int(idx),
                "op_name": str(spec.get("name", f"op_{idx}")),
                "grad_probe": float(grad_probe),
                "grad_fd": float(grad_fd),
                "abs_err": float(abs_err),
                "rel_err": float(rel_err),
            }
        )

    summary = {
        "checked": int(len(details)),
        "delta": float(delta_f),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_rel_err": float(max_rel_err),
        "details": details,
    }
    if verbose:
        print(
            "Grouped outer-gradient check: "
            f"checked={summary['checked']}, max rel err={summary['max_rel_err']:.3e}"
        )
    if max_rel_err > float(rtol):
        raise RuntimeError(
            "Grouped outer-gradient validation failed: "
            f"max relative error {max_rel_err:.3e} > rtol {float(rtol):.3e}."
        )
    return summary


def run_meta_adapt_vqe(
    qubit_op: SparsePauliOp,
    reference_state: QuantumCircuit,
    estimator: BaseEstimatorV2,
    *,
    pool: Sequence[str] | None = None,
    pool_mode: str = "ham_terms_plus_imag_partners",
    ferm_op: FermionicOp | None = None,
    mapper: JordanWignerMapper | None = None,
    cse_include_diagonal: bool = True,
    cse_include_antihermitian_part: bool = True,
    cse_include_hermitian_part: bool = True,
    uccsd_reps: int = 1,
    uccsd_include_imaginary: bool = False,
    uccsd_generalized: bool = False,
    uccsd_preserve_spin: bool = True,
    hardcoded_indexing: str = "blocked",
    hardcoded_include_singles: bool = True,
    hardcoded_include_doubles: bool = True,
    n_sites: int | None = None,
    n_up: int | None = None,
    n_down: int | None = None,
    enforce_sector: bool = False,
    max_depth: int = 20,
    inner_steps: int = 30,
    eps_grad: float = 1e-4,
    eps_energy: float = 1e-3,
    inner_fd_eps: float = 1e-6,
    lstm_optimizer: CoordinateWiseLSTMOptimizer | None = None,
    meta_model=None,
    seed: int = 11,
    r: float = 10.0,
    reuse_lstm_state: bool = True,
    wrap_angles: bool = True,
    probe_convention: str = "half_angle",
    grouped_selection_gradient: str = "directional_fd",
    grouped_selection_fd_delta: float = 1e-5,
    inner_optimizer: str = "lbfgs",
    theta_bound: str = "none",
    meta_step_scale: float = 1.0,
    meta_dtheta_clip: float | None = 0.25,
    meta_warmup_steps: int = 15,
    lbfgs_maxiter: int | None = None,
    lbfgs_restarts: int = 1,
    theta_init_noise: float = 0.0,
    allow_repeats: bool = False,
    grouped_trotter_reps: int = 1,
    grouped_trotter_order: int = 1,
    grouped_generator_scale: float | None = None,
    grouped_term_order: str | None = None,
    selection_mode: str = "adapt",
    selection_strategy: str = "max_gradient",
    lookahead_top_k: int = 3,
    lookahead_inner_steps: int = 30,
    warmup_steps: int = 5,
    polish_steps: int = 3,
    meta_alpha0: float = 1.0,
    meta_alpha_k: float = 0.0,
    max_time_s: float | None = None,
    inner_steps_schedule: Callable[[int], int] | None = None,
    debug_validate_semantics: bool = False,
    debug_semantics_trials: int = 10,
    debug_semantics_theta_std: float = 0.3,
    debug_semantics_atol: float = 1e-10,
    debug_validate_outer_grads: bool = False,
    debug_outer_grad_checks: int = 5,
    debug_outer_grad_delta: float = 1e-5,
    debug_outer_grad_rtol: float = 1e-3,
    debug_outer_grad_atol: float = 1e-8,
    compute_var_h: bool = False,
    var_h_simplify_atol: float = 1e-12,
    logger=None,
    log_every: int = 1,
    verbose: bool = True,
    **legacy_kwargs,
) -> MetaAdaptVQEResult:
    t0 = time.perf_counter()
    stop_reason: str | None = None
    _ = legacy_kwargs  # Backward-compatibility sink for removed legacy budget kwargs.

    def _budget_reason() -> str | None:
        # time budget
        if max_time_s is not None and (time.perf_counter() - t0) >= float(max_time_s):
            return "max_time_s"
        return None

    def _raise_if_over_budget(last_energy: float | None = None) -> None:
        reason = _budget_reason()
        if reason is not None:
            raise BudgetExceeded(reason, last_energy=last_energy)

    pool_specs: list[dict] | None = None
    pool_labels: list[str] | None = None
    grouped_generator_scale_eff: float | None = None
    grouped_term_order_eff: str = "as_given"
    n_q = None
    sz_q = None
    if enforce_sector:
        if mapper is None or n_sites is None:
            raise ValueError("mapper and n_sites are required when enforce_sector is True.")
        n_q, sz_q = map_symmetry_ops_to_qubits(mapper, n_sites)

    if pool_mode in GROUPED_POOL_MODES:
        if pool is not None:
            pool_specs = list(pool)  # assume list of dict specs
        else:
            if pool_mode == "cse_density_ops":
                if ferm_op is None or mapper is None:
                    raise ValueError("ferm_op and mapper are required for cse_density_ops pool.")
                pool_specs = build_cse_density_pool_from_fermionic(
                    ferm_op,
                    mapper,
                    include_diagonal=cse_include_diagonal,
                    include_antihermitian_part=cse_include_antihermitian_part,
                    include_hermitian_part=cse_include_hermitian_part,
                    enforce_symmetry=enforce_sector,
                    n_sites=n_sites,
                )
            elif pool_mode == "uccsd_excitations":
                if mapper is None or n_sites is None or n_up is None or n_down is None:
                    raise ValueError("mapper, n_sites, n_up, n_down are required for uccsd_excitations pool.")
                pool_specs = build_uccsd_excitation_pool(
                    n_sites=n_sites,
                    num_particles=(int(n_up), int(n_down)),
                    mapper=mapper,
                    reps=uccsd_reps,
                    include_imaginary=uccsd_include_imaginary,
                    preserve_spin=uccsd_preserve_spin,
                    generalized=uccsd_generalized,
                )
            elif pool_mode in HARDCODED_ONZOT_POOL_MODES:
                if n_sites is None or n_up is None or n_down is None:
                    raise ValueError(
                        "n_sites, n_up, n_down are required for hardcoded_uccsd_onzots pool."
                    )
                pool_specs = build_hardcoded_uccsd_onzot_pool(
                    n_sites=int(n_sites),
                    num_particles=(int(n_up), int(n_down)),
                    indexing=str(hardcoded_indexing),
                    include_singles=bool(hardcoded_include_singles),
                    include_doubles=bool(hardcoded_include_doubles),
                )
            else:
                raise ValueError(f"Unknown pool mode: {pool_mode}")
        if not pool_specs:
            raise ValueError("Operator pool is empty.")
        if grouped_generator_scale is None:
            if pool_mode in HARDCODED_ONZOT_POOL_MODES:
                grouped_generator_scale_eff = 1.0
            else:
                grouped_generator_scale_eff = 0.5
        else:
            grouped_generator_scale_eff = float(grouped_generator_scale)
        if grouped_term_order is None:
            if pool_mode in HARDCODED_ONZOT_POOL_MODES:
                grouped_term_order_eff = "canonical_sorted"
            else:
                grouped_term_order_eff = "as_given"
        else:
            grouped_term_order_eff = str(grouped_term_order)
        if grouped_term_order_eff not in {"as_given", "canonical_sorted"}:
            raise ValueError(
                f"Unknown grouped_term_order={grouped_term_order_eff}; "
                "expected 'as_given' or 'canonical_sorted'."
            )
        if enforce_sector and n_q is not None and sz_q is not None:
            filtered = []
            for spec in pool_specs:
                op = SparsePauliOp.from_list(spec["paulis"])
                if commutes(op, n_q) and commutes(op, sz_q):
                    filtered.append(spec)
            pool_specs = filtered
            if not pool_specs:
                raise ValueError("Operator pool empty after symmetry filtering.")
    else:
        if pool is None:
            pool = build_operator_pool_from_hamiltonian(qubit_op, mode=pool_mode)
        pool_labels = list(pool)
        if not pool_labels:
            raise ValueError("Operator pool is empty.")
        if enforce_sector and n_q is not None and sz_q is not None:
            filtered = []
            for label in pool_labels:
                op = SparsePauliOp.from_list([(label, 1.0)])
                if commutes(op, n_q) and commutes(op, sz_q):
                    filtered.append(label)
            pool_labels = filtered
            if not pool_labels:
                raise ValueError("Operator pool empty after symmetry filtering.")

    if selection_mode not in {"adapt", "fixed_sequence"}:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    if selection_strategy not in {"max_gradient", "lookahead_topk"}:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")
    if grouped_selection_gradient not in {"directional_fd", "probe_legacy"}:
        raise ValueError(
            f"Unknown grouped_selection_gradient={grouped_selection_gradient}; "
            "expected 'directional_fd' or 'probe_legacy'."
        )
    if float(grouped_selection_fd_delta) <= 0.0:
        raise ValueError("grouped_selection_fd_delta must be positive.")
    if selection_mode == "fixed_sequence" and pool_mode not in GROUPED_POOL_MODES:
        raise ValueError("selection_mode='fixed_sequence' requires a grouped pool mode.")

    if lstm_optimizer is None:
        lstm_optimizer = CoordinateWiseLSTMOptimizer(seed=seed)

    h2_op = None
    if compute_var_h:
        # For small-L statevector benchmarks this is cheap enough and gives a
        # strong correctness signal (Var(H)=0 for exact eigenstates).
        h2_op = (qubit_op @ qubit_op).simplify(atol=var_h_simplify_atol)

    rng = np.random.default_rng(seed)
    ops: list = []
    theta = np.zeros((0,), dtype=float)

    def _energy_obj(circuit: QuantumCircuit, values: Sequence[float]) -> float:
        e = estimate_energy(estimator, circuit, qubit_op, list(values))
        return e

    def _energy_obj_safe(circuit: QuantumCircuit, values: Sequence[float]) -> float:
        """Energy objective that also checks budget — use only outside SciPy minimize."""
        e = estimate_energy(estimator, circuit, qubit_op, list(values))
        _raise_if_over_budget(last_energy=e)
        return e

    def _energy_grad(circuit: QuantumCircuit, values: Sequence[float]) -> float:
        e = estimate_energy(estimator, circuit, qubit_op, list(values))
        return e

    def _build_circuit_for_ops(selected_ops: Sequence) -> tuple[QuantumCircuit, ParameterVector]:
        if pool_mode in GROUPED_POOL_MODES:
            return build_adapt_circuit_grouped(
                reference_state,
                selected_ops,
                trotter_reps=grouped_trotter_reps,
                trotter_order=grouped_trotter_order,
                generator_scale=float(grouped_generator_scale_eff),
                grouped_term_order=str(grouped_term_order_eff),
            )
        return build_adapt_circuit(reference_state, selected_ops)

    def _grad_fn_for_circuit(circuit: QuantumCircuit):
        def energy_fn_grad(values: Iterable[float]) -> float:
            return _energy_grad(circuit, list(values))

        def grad_fn(values: Iterable[float]) -> np.ndarray:
            x = np.asarray(values, dtype=float)
            if pool_mode in GROUPED_POOL_MODES:
                return finite_difference_grad(energy_fn_grad, x, eps=float(inner_fd_eps))
            return parameter_shift_grad(energy_fn_grad, x)

        return grad_fn

    def _run_inner_optimizer(
        theta_in: np.ndarray,
        energy_fn,
        grad_fn,
        *,
        steps: int,
    ) -> tuple[np.ndarray, dict]:
        if inner_optimizer == "meta":
            if meta_model is None:
                raise RuntimeError("Meta optimizer requested without a loaded model.")
            return _meta_optimize(
                theta_in,
                energy_fn,
                grad_fn,
                steps=steps,
                model=meta_model,
                r=r,
                step_scale=meta_step_scale,
                dtheta_clip=meta_dtheta_clip,
                alpha0=meta_alpha0,
                alpha_k=meta_alpha_k,
                verbose=verbose,
            )
        if inner_optimizer == "lbfgs":
            return _lbfgs_optimize(
                theta_in,
                energy_fn,
                grad_fn,
                theta_bound=theta_bound,
                maxiter=steps if steps > 0 else None,
                restarts=lbfgs_restarts,
                rng=rng,
            )
        if inner_optimizer == "slsqp":
            return _slsqp_optimize(
                theta_in,
                energy_fn,
                theta_bound=theta_bound,
                maxiter=steps if steps > 0 else None,
                restarts=lbfgs_restarts,
                rng=rng,
            )
        if inner_optimizer == "hybrid":
            warm_steps = max(0, min(steps, warmup_steps))
            meta_steps = max(0, steps - warm_steps)

            theta_warm, warm_stats = _lbfgs_optimize(
                theta_in,
                energy_fn,
                grad_fn,
                theta_bound=theta_bound,
                maxiter=warm_steps if warm_steps > 0 else None,
                restarts=1,
                rng=rng,
            )
            if verbose and warm_steps > 0:
                e_warm = energy_fn(theta_warm)
                g_warm = grad_fn(theta_warm)
                print(
                    f"  warm-start: E={e_warm:.10f}, ||g||={np.linalg.norm(g_warm):.3e}, "
                    f"max|g|={np.max(np.abs(g_warm)):.3e}"
                )

            theta_meta = theta_warm.copy()
            delta_norms = []
            if meta_steps > 0:
                if meta_model is not None:
                    torch = _maybe_import_torch()
                    device = next(meta_model.parameters()).device
                    meta_model.eval()
                    n_params = theta_meta.size
                    h = torch.zeros((n_params, meta_model.hidden_size), device=device)
                    c = torch.zeros((n_params, meta_model.hidden_size), device=device)

                    for step_idx in range(meta_steps):
                        grad = grad_fn(theta_meta)
                        x = preprocess_gradients_torch(grad, r=r, device=device)
                        with torch.no_grad():
                            delta, (h, c) = meta_model(x, (h, c))
                        delta = delta.cpu().numpy().astype(float)
                        if meta_dtheta_clip is not None:
                            delta = np.clip(delta, -meta_dtheta_clip, meta_dtheta_clip)
                        alpha = meta_alpha0 * (1.0 + meta_alpha_k * (step_idx / max(1, meta_steps - 1)))
                        delta = alpha * delta
                        theta_meta = theta_meta + delta
                        delta_norms.append(float(np.linalg.norm(delta)))
                else:
                    for step_idx in range(meta_steps):
                        grad = grad_fn(theta_meta)
                        delta = -grad
                        if meta_dtheta_clip is not None:
                            delta = np.clip(delta, -meta_dtheta_clip, meta_dtheta_clip)
                        alpha = meta_alpha0 * (1.0 + meta_alpha_k * (step_idx / max(1, meta_steps - 1)))
                        delta = alpha * delta
                        theta_meta = theta_meta + delta
                        delta_norms.append(float(np.linalg.norm(delta)))

            meta_stats = {"avg_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0}

            theta_polish, polish_stats = _lbfgs_optimize(
                theta_meta,
                energy_fn,
                grad_fn,
                theta_bound=theta_bound,
                maxiter=polish_steps if polish_steps > 0 else None,
                restarts=1,
                rng=rng,
            )
            if verbose and polish_steps > 0:
                e_polish = energy_fn(theta_polish)
                g_polish = grad_fn(theta_polish)
                print(
                    f"  polish: E={e_polish:.10f}, ||g||={np.linalg.norm(g_polish):.3e}, "
                    f"max|g|={np.max(np.abs(g_polish)):.3e}"
                )
            return theta_polish, {
                "warm": warm_stats,
                "meta": meta_stats,
                "polish": polish_stats,
            }
        raise ValueError(f"Unknown inner optimizer: {inner_optimizer}")

    try:
        energy_current = _energy_obj(reference_state, [])
    except BudgetExceeded as exc:
        energy_current = float(exc.last_energy) if exc.last_energy is not None else float("nan")
        stop_reason = f"budget:{exc.reason}"
    prev_energy: float | None = None

    lstm_state = lstm_optimizer.init_state(0)

    diagnostics: dict[str, object] = {"outer": []}
    if pool_mode in GROUPED_POOL_MODES:
        diagnostics["pool_size"] = [len(pool_specs)]
    else:
        diagnostics["pool_size"] = [len(pool_labels)]
    diagnostics["selection_mode"] = selection_mode
    diagnostics["selection_strategy"] = selection_strategy
    diagnostics["inner_optimizer"] = inner_optimizer
    if pool_mode in GROUPED_POOL_MODES:
        diagnostics["grouped_term_order"] = grouped_term_order_eff
        diagnostics["grouped_trotter_order"] = int(grouped_trotter_order)
        diagnostics["grouped_trotter_reps"] = int(grouped_trotter_reps)
        diagnostics["grouped_selection_gradient"] = str(grouped_selection_gradient)
        diagnostics["grouped_selection_fd_delta"] = float(grouped_selection_fd_delta)

    def _selected_sequence_snapshot(selected_ops: Sequence) -> list:
        if pool_mode in GROUPED_POOL_MODES:
            out: list[dict] = []
            for idx, spec in enumerate(selected_ops):
                if isinstance(spec, dict):
                    paulis = []
                    for term in spec.get("paulis", []):
                        if isinstance(term, (tuple, list)) and len(term) == 2:
                            paulis.append([str(term[0]), float(term[1])])
                    out.append(
                        {
                            "index": int(idx),
                            "name": str(spec.get("name", f"op_{idx}")),
                            "paulis": paulis,
                        }
                    )
                else:
                    out.append({"index": int(idx), "name": str(spec), "paulis": []})
            return out
        return [str(op) for op in selected_ops]

    if debug_validate_semantics and pool_mode in HARDCODED_ONZOT_POOL_MODES:
        if mapper is None or n_sites is None or n_up is None or n_down is None:
            raise ValueError(
                "mapper, n_sites, n_up, n_down are required for hardcoded semantics validation."
            )
        diagnostics["debug_semantics"] = debug_validate_hardcoded_uccsd_semantics(
            n_sites=int(n_sites),
            n_up=int(n_up),
            n_down=int(n_down),
            mapper=mapper,
            reference_state=reference_state,
            build_grouped_circuit_fn=build_adapt_circuit_grouped,
            pool_specs=pool_specs,
            grouped_term_order=grouped_term_order_eff,
            grouped_trotter_order=int(grouped_trotter_order),
            grouped_trotter_reps=int(grouped_trotter_reps),
            grouped_generator_scale=float(grouped_generator_scale_eff),
            hardcoded_indexing=str(hardcoded_indexing),
            hardcoded_include_singles=bool(hardcoded_include_singles),
            hardcoded_include_doubles=bool(hardcoded_include_doubles),
            n_trials=int(debug_semantics_trials),
            seed=int(seed),
            theta_std=float(debug_semantics_theta_std),
            atol=float(debug_semantics_atol),
            verbose=verbose,
        )

    shift, coeff = _probe_shift_and_coeff(probe_convention)
    wrap_skip_note_printed = False

    if logger is not None:
        sector_extra = None
        if stop_reason is None and mapper is not None and n_sites is not None and n_up is not None and n_down is not None:
            sector_extra = compute_sector_diagnostics(
                estimator=estimator,
                circuit=reference_state,
                params=[],
                mapper=mapper,
                n_sites=int(n_sites),
                n_target=int(n_up) + int(n_down),
                sz_target=0.5 * (int(n_up) - int(n_down)),
            )
        var_h_extra = None
        if stop_reason is None and h2_op is not None:
            e2 = estimate_expectation(estimator, reference_state, h2_op, [])
            var_h_extra = {"VarH": float(max(0.0, e2 - energy_current * energy_current))}
        logger.log_point(
            it=0,
            energy=energy_current,
            max_grad=None,
            chosen_op=None,
            t_iter_s=0.0,
            t_cum_s=0.0,
            extra={
                "ansatz_len": 0,
                "n_params": 0,
                "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                "stop_reason": stop_reason,
                **(var_h_extra or {}),
                **(sector_extra or {}),
            },
        )

    if stop_reason is not None:
        diagnostics["selected_sequence"] = _selected_sequence_snapshot(ops)
        diagnostics["n_selected_ops"] = int(len(ops))
        diagnostics["stop_reason"] = stop_reason
        diagnostics["t_total_s"] = float(time.perf_counter() - t0)
        return MetaAdaptVQEResult(
            energy=energy_current,
            params=[],
            operators=[],
            diagnostics=diagnostics,
        )

    if selection_mode == "fixed_sequence":
        if pool_mode not in GROUPED_POOL_MODES:
            raise ValueError("selection_mode='fixed_sequence' requires grouped pool operators.")
        if not pool_specs:
            raise ValueError("selection_mode='fixed_sequence' received an empty grouped pool.")

        ops = list(pool_specs)
        theta = np.zeros((len(ops),), dtype=float)
        if theta_init_noise > 0.0 and theta.size > 0:
            theta = theta + rng.normal(scale=theta_init_noise, size=theta.size)

        if logger is not None:
            logger.start_iter()

        # --- Check budget BEFORE inner optimization ---
        budget_reason = _budget_reason()
        if budget_reason is not None:
            stop_reason = f"budget:{budget_reason}"
            if logger is not None:
                dt, tc = logger.end_iter()
                logger.log_point(
                    it=1,
                    energy=energy_current,
                    max_grad=None,
                    chosen_op=None,
                    t_iter_s=dt,
                    t_cum_s=tc,
                    extra={
                        "ansatz_len": len(ops),
                        "n_params": len(theta),
                        "pool_size": len(pool_specs),
                        "stop_reason": stop_reason,
                        "selection_mode": "fixed_sequence",
                    },
                )
            diagnostics["stop_reason"] = stop_reason
            diagnostics["t_total_s"] = float(time.perf_counter() - t0)
            return MetaAdaptVQEResult(
                energy=energy_current,
                params=list(map(float, theta)),
                operators=ops,
                diagnostics=diagnostics,
            )

        circuit, _params = _build_circuit_for_ops(ops)

        def energy_fn_fixseq(values: Iterable[float]) -> float:
            return _energy_obj(circuit, list(values))

        # Deterministic restarts for fixed-sequence canonical benchmark
        fixed_seq_restarts = max(1, int(lbfgs_restarts))
        best_theta_fs = theta.copy()
        best_energy_fs = energy_fn_fixseq(theta)
        inner_stats_best = {}

        for restart_idx in range(fixed_seq_restarts):
            if restart_idx == 0:
                theta_restart = theta.copy()
            else:
                theta_restart = rng.uniform(-math.pi, math.pi, size=theta.size)

            # --- Check budget before each restart ---
            budget_reason = _budget_reason()
            if budget_reason is not None:
                stop_reason = f"budget:{budget_reason}"
                break

            if inner_optimizer == "slsqp":
                theta_opt, inner_stats = _slsqp_optimize(
                    theta_restart,
                    energy_fn_fixseq,
                    theta_bound=theta_bound if theta_bound != "none" else "pi",
                    maxiter=inner_steps if inner_steps > 0 else None,
                    restarts=1,
                    rng=rng,
                )
            else:
                grad_fn_fs = _grad_fn_for_circuit(circuit)
                theta_opt, inner_stats = _lbfgs_optimize(
                    theta_restart,
                    energy_fn_fixseq,
                    grad_fn_fs,
                    theta_bound=theta_bound if theta_bound != "none" else "pi",
                    maxiter=inner_steps if inner_steps > 0 else None,
                    restarts=1,
                    rng=rng,
                )

            e_opt = energy_fn_fixseq(theta_opt)
            if e_opt < best_energy_fs:
                best_energy_fs = e_opt
                best_theta_fs = theta_opt.copy()
                inner_stats_best = inner_stats

        theta = best_theta_fs
        energy_current = best_energy_fs

        # --- Check budget AFTER inner optimization ---
        budget_reason = _budget_reason()
        if budget_reason is not None and stop_reason is None:
            stop_reason = f"budget:{budget_reason}"

        if wrap_angles and verbose:
            print(
                "Angle wrapping skipped for grouped pools: shared weighted generators "
                "are not 2pi-periodic in theta."
            )

        sector_diag = None
        if mapper is not None and n_sites is not None and n_up is not None and n_down is not None:
            sector_diag = compute_sector_diagnostics(
                estimator=estimator,
                circuit=circuit,
                params=theta,
                mapper=mapper,
                n_sites=int(n_sites),
                n_target=int(n_up) + int(n_down),
                sz_target=0.5 * (int(n_up) - int(n_down)),
            )

        var_h_diag = None
        if h2_op is not None:
            e2 = estimate_expectation(estimator, circuit, h2_op, theta)
            var_h_diag = float(max(0.0, e2 - energy_current * energy_current))

        if stop_reason is None:
            stop_reason = "fixed_sequence_complete"

        diagnostics["outer"].append(
            {
                "outer_iter": 1,
                "chosen_op": "fixed_sequence_all",
                "chosen_components": None,
                "max_grad": None,
                "energy": energy_current,
                "VarH": var_h_diag,
                "pool_gradients": None,
                "inner_stats": inner_stats_best,
                "inner_optimizer": inner_optimizer,
                "sector": sector_diag,
                "selection_mode": "fixed_sequence",
                "n_ops": len(ops),
                "restarts": fixed_seq_restarts,
            }
        )

        if logger is not None:
            dt, tc = logger.end_iter()
            logger.log_point(
                it=1,
                energy=energy_current,
                max_grad=None,
                chosen_op="fixed_sequence_all",
                t_iter_s=dt,
                t_cum_s=tc,
                extra={
                    "ansatz_len": len(ops),
                    "n_params": len(theta),
                    "pool_size": len(pool_specs),
                    "stop_reason": stop_reason,
                    "selection_mode": "fixed_sequence",
                    **({"VarH": float(var_h_diag)} if var_h_diag is not None else {}),
                    **(sector_diag or {}),
                },
            )

        diagnostics["stop_reason"] = stop_reason
        diagnostics["selected_sequence"] = _selected_sequence_snapshot(ops)
        diagnostics["n_selected_ops"] = int(len(ops))
        diagnostics["t_total_s"] = float(time.perf_counter() - t0)
        return MetaAdaptVQEResult(
            energy=energy_current,
            params=list(map(float, theta)),
            operators=ops,
            diagnostics=diagnostics,
        )

    for outer_idx in range(max_depth):
        iter_t0 = time.perf_counter()
        if logger is not None:
            logger.start_iter()

        # --- Safe budget check BEFORE outer iteration ---
        budget_reason = _budget_reason()
        if budget_reason is not None:
            stop_reason = f"budget:{budget_reason}"
            if logger is not None:
                dt, tc = logger.end_iter()
                logger.log_point(
                    it=outer_idx + 1,
                    energy=energy_current,
                    max_grad=None,
                    chosen_op=None,
                    t_iter_s=dt,
                    t_cum_s=tc,
                    extra={
                        "ansatz_len": len(ops),
                        "n_params": len(theta),
                        "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                        "stop_reason": stop_reason,
                    },
                )
            break

        try:
            if pool_mode in GROUPED_POOL_MODES and not pool_specs:
                if verbose:
                    print("ADAPT stop: operator pool exhausted.")
                stop_reason = "pool_exhausted"
                break
            if pool_mode not in GROUPED_POOL_MODES and not pool_labels:
                if verbose:
                    print("ADAPT stop: operator pool exhausted.")
                stop_reason = "pool_exhausted"
                break
            circuit, _params = _build_circuit_for_ops(ops)

            if pool_mode in GROUPED_POOL_MODES:
                # Selection gradients for grouped pools:
                # - directional_fd: exact directional finite-difference on trial-augmented circuit
                # - probe_legacy: legacy probe-based approximation
                if grouped_selection_gradient == "directional_fd":
                    pool_gradients = _compute_grouped_pool_gradients_fd(
                        estimator,
                        reference_state,
                        qubit_op,
                        ops,
                        theta,
                        pool_specs,
                        delta=float(grouped_selection_fd_delta),
                        trotter_reps=int(grouped_trotter_reps),
                        trotter_order=int(grouped_trotter_order),
                        generator_scale=float(grouped_generator_scale_eff),
                        grouped_term_order=str(grouped_term_order_eff),
                    )
                else:
                    pool_gradients = _compute_grouped_pool_gradients(
                        estimator,
                        circuit,
                        qubit_op,
                        theta,
                        pool_specs,
                        probe_convention=probe_convention,
                    )
                max_abs_grad = (
                    float(np.max(np.abs(np.asarray(pool_gradients, dtype=float))))
                    if pool_gradients
                    else 0.0
                )
                if debug_validate_outer_grads and outer_idx < 2 and pool_specs:
                    debug_outer = debug_validate_grouped_outer_gradients(
                        estimator=estimator,
                        qubit_op=qubit_op,
                        reference_state=reference_state,
                        current_ops=ops,
                        theta=theta,
                        pool_specs=pool_specs,
                        probe_convention=probe_convention,
                        grouped_trotter_order=int(grouped_trotter_order),
                        grouped_trotter_reps=int(grouped_trotter_reps),
                        grouped_generator_scale=float(grouped_generator_scale_eff),
                        grouped_term_order=str(grouped_term_order_eff),
                        n_checks=int(debug_outer_grad_checks),
                        delta=float(debug_outer_grad_delta),
                        rtol=float(debug_outer_grad_rtol),
                        atol=float(debug_outer_grad_atol),
                        verbose=verbose,
                    )
                    diagnostics.setdefault("debug_outer_grads", []).append(
                        {"outer_iter": outer_idx + 1, **debug_outer}
                    )
            else:
                pool_gradients = []
                max_abs_grad = -1.0
                for idx, label in enumerate(pool_labels):
                    probe_plus = _append_probe_gate(circuit, label, angle=shift)
                    probe_minus = _append_probe_gate(circuit, label, angle=-shift)
                    e_plus = _energy_grad(probe_plus, theta)
                    e_minus = _energy_grad(probe_minus, theta)
                    grad_val = coeff * (e_plus - e_minus)
                    pool_gradients.append(grad_val)
                    if abs(grad_val) > max_abs_grad:
                        max_abs_grad = abs(grad_val)

            top_k_indices = np.argsort(np.abs(np.asarray(pool_gradients, dtype=float)))[::-1][:5]
            top_k_candidates: list[dict] = []
            for idx in top_k_indices:
                if pool_mode in GROUPED_POOL_MODES:
                    cand_name = str(pool_specs[int(idx)].get("name", f"op_{idx}"))
                else:
                    cand_name = str(pool_labels[int(idx)])
                top_k_candidates.append(
                    {
                        "pool_index": int(idx),
                        "op_name": cand_name,
                        "gradient": float(pool_gradients[int(idx)]),
                    }
                )

            if prev_energy is not None and abs(energy_current - prev_energy) < eps_energy:
                if verbose:
                    print(
                        f"ADAPT stop: |ΔE|={abs(energy_current - prev_energy):.3e} < {eps_energy}"
                    )
                stop_reason = "eps_energy"
                if logger is not None:
                    dt, tc = logger.end_iter()
                    if (outer_idx + 1) % max(1, log_every) == 0:
                        var_h_extra = None
                        if h2_op is not None:
                            e2 = estimate_expectation(estimator, circuit, h2_op, theta)
                            var_h_extra = {"VarH": float(max(0.0, e2 - energy_current * energy_current))}
                        logger.log_point(
                            it=outer_idx + 1,
                            energy=energy_current,
                            max_grad=max_abs_grad,
                            chosen_op=None,
                            t_iter_s=dt,
                            t_cum_s=tc,
                            extra={
                                "ansatz_len": len(ops),
                                "n_params": len(theta),
                                "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                                "stop_reason": stop_reason,
                                **(var_h_extra or {}),
                            },
                        )
                break
            if max_abs_grad < eps_grad:
                if verbose:
                    print(f"ADAPT stop: max|g|={max_abs_grad:.3e} < {eps_grad}")
                stop_reason = "eps_grad"
                if logger is not None:
                    dt, tc = logger.end_iter()
                    if (outer_idx + 1) % max(1, log_every) == 0:
                        var_h_extra = None
                        if h2_op is not None:
                            e2 = estimate_expectation(estimator, circuit, h2_op, theta)
                            var_h_extra = {"VarH": float(max(0.0, e2 - energy_current * energy_current))}
                        logger.log_point(
                            it=outer_idx + 1,
                            energy=energy_current,
                            max_grad=max_abs_grad,
                            chosen_op=None,
                            t_iter_s=dt,
                            t_cum_s=tc,
                            extra={
                                "ansatz_len": len(ops),
                                "n_params": len(theta),
                                "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                                "stop_reason": stop_reason,
                                **(var_h_extra or {}),
                            },
                        )
                break

            lookahead_diag = None
            if selection_strategy == "max_gradient":
                max_idx = int(np.argmax(np.abs(np.asarray(pool_gradients, dtype=float))))
            elif selection_strategy == "lookahead_topk":
                grad_abs = np.abs(np.asarray(pool_gradients, dtype=float))
                top_k = max(1, min(int(lookahead_top_k), grad_abs.size))
                candidate_indices = np.argsort(grad_abs)[::-1][:top_k]
                lookahead_steps = max(1, int(lookahead_inner_steps))
                best_idx = int(candidate_indices[0])
                best_energy = float("inf")
                lookahead_trials = []

                for cand_idx in candidate_indices:
                    if pool_mode in GROUPED_POOL_MODES:
                        trial_op = pool_specs[int(cand_idx)]
                        trial_name = str(trial_op.get("name", f"op_{cand_idx}"))
                    else:
                        trial_op = pool_labels[int(cand_idx)]
                        trial_name = str(trial_op)

                    trial_ops = list(ops) + [trial_op]
                    trial_theta = np.concatenate([theta, [0.0]])
                    if theta_init_noise > 0.0:
                        trial_theta[-1] = rng.normal(scale=theta_init_noise)
                    trial_circuit, _trial_params = _build_circuit_for_ops(trial_ops)

                    def trial_energy_fn(values: Iterable[float]) -> float:
                        return _energy_obj(trial_circuit, list(values))

                    trial_grad_fn = _grad_fn_for_circuit(trial_circuit)
                    if inner_optimizer == "slsqp":
                        trial_theta_opt, trial_stats = _slsqp_optimize(
                            trial_theta,
                            trial_energy_fn,
                            theta_bound=theta_bound,
                            maxiter=lookahead_steps,
                            restarts=1,
                            rng=rng,
                        )
                    else:
                        trial_theta_opt, trial_stats = _lbfgs_optimize(
                            trial_theta,
                            trial_energy_fn,
                            trial_grad_fn,
                            theta_bound=theta_bound,
                            maxiter=lookahead_steps,
                            restarts=1,
                            rng=rng,
                        )
                    trial_energy = float(trial_energy_fn(trial_theta_opt))
                    lookahead_trials.append(
                        {
                            "pool_index": int(cand_idx),
                            "op_name": trial_name,
                            "energy": trial_energy,
                            "grad_abs": float(grad_abs[int(cand_idx)]),
                            "inner_stats": trial_stats,
                        }
                    )
                    if trial_energy < best_energy:
                        best_energy = trial_energy
                        best_idx = int(cand_idx)

                max_idx = best_idx
                lookahead_diag = {
                    "top_k": int(top_k),
                    "steps": int(lookahead_steps),
                    "selected_index": int(max_idx),
                    "selected_energy": float(best_energy),
                    "trials": lookahead_trials,
                }
            else:
                raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

            if pool_mode in GROUPED_POOL_MODES:
                chosen_spec = pool_specs[max_idx]
                ops.append(chosen_spec)
                chosen_op = chosen_spec["name"]
                if not allow_repeats:
                    pool_specs.pop(max_idx)
            else:
                chosen_op = pool_labels[max_idx]
                ops.append(chosen_op)
                if not allow_repeats:
                    pool_labels.pop(max_idx)
            if verbose:
                print(
                    f"ADAPT iter {len(ops)}: op={chosen_op}, max|g|={max_abs_grad:.6e}"
                )
                if pool_mode in GROUPED_POOL_MODES:
                    comp_preview = ", ".join(
                        f"{lbl}:{wt:.3f}" for lbl, wt in chosen_spec["paulis"][:6]
                    )
                    if len(chosen_spec["paulis"]) > 6:
                        comp_preview += ", ..."
                    print(f"  components: {comp_preview}")

            theta = np.concatenate([theta, [0.0]])
            if theta_init_noise > 0.0:
                theta[-1] = rng.normal(scale=theta_init_noise)
            if reuse_lstm_state:
                h = np.vstack([lstm_state.h, np.zeros((1, lstm_optimizer.hidden_size))])
                c = np.vstack([lstm_state.c, np.zeros((1, lstm_optimizer.hidden_size))])
                lstm_state = LSTMState(h=h, c=c)
            else:
                lstm_state = lstm_optimizer.init_state(len(theta))

            circuit, _params = _build_circuit_for_ops(ops)

            def energy_fn(values: Iterable[float]) -> float:
                return _energy_obj(circuit, list(values))

            # --- Check budget BEFORE inner solve ---
            budget_reason = _budget_reason()
            if budget_reason is not None:
                stop_reason = f"budget:{budget_reason}"
                energy_current = energy_fn(theta)
                if logger is not None:
                    dt, tc = logger.end_iter()
                    logger.log_point(
                        it=outer_idx + 1,
                        energy=energy_current,
                        max_grad=max_abs_grad,
                        chosen_op=chosen_op,
                        t_iter_s=dt,
                        t_cum_s=tc,
                        extra={
                            "ansatz_len": len(ops),
                            "n_params": len(theta),
                            "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                            "stop_reason": stop_reason,
                        },
                    )
                break

            pre_opt_energy = energy_fn(theta)

            grad_fn = _grad_fn_for_circuit(circuit)

            steps_this_iter = int(inner_steps)
            if inner_steps_schedule is not None:
                steps_this_iter = int(inner_steps_schedule(len(theta)))
            steps_this_iter = max(0, steps_this_iter)

            theta, inner_stats = _run_inner_optimizer(
                theta,
                energy_fn,
                grad_fn,
                steps=steps_this_iter,
            )

            if wrap_angles and pool_mode not in GROUPED_POOL_MODES:
                theta = _wrap_angles(theta)
            elif wrap_angles and pool_mode in GROUPED_POOL_MODES and verbose and not wrap_skip_note_printed:
                print(
                    "Angle wrapping skipped for grouped pools: shared weighted generators are not 2pi-periodic in theta."
                )
                wrap_skip_note_printed = True

            prev_energy = energy_current
            energy_current = energy_fn(theta)
            energy_drop = pre_opt_energy - energy_current
            iter_wall_time = time.perf_counter() - iter_t0
            cum_wall_time = time.perf_counter() - t0

            # --- Check budget AFTER inner solve ---
            budget_reason = _budget_reason()
            if budget_reason is not None:
                stop_reason = f"budget:{budget_reason}"

            sector_diag = None
            if mapper is not None and n_sites is not None and n_up is not None and n_down is not None:
                sector_diag = compute_sector_diagnostics(
                    estimator=estimator,
                    circuit=circuit,
                    params=theta,
                    mapper=mapper,
                    n_sites=int(n_sites),
                    n_target=int(n_up) + int(n_down),
                    sz_target=0.5 * (int(n_up) - int(n_down)),
                )

            var_h_diag = None
            if h2_op is not None:
                e2 = estimate_expectation(estimator, circuit, h2_op, theta)
                var_h_diag = float(max(0.0, e2 - energy_current * energy_current))

            # Enhanced per-iteration diagnostics
            top_k_grads = [(item["pool_index"], item["gradient"]) for item in top_k_candidates]
            inner_stop_reason = inner_stats.get("message", inner_stats.get("status", "unknown"))

            if verbose:
                print(
                    f"  post-opt: E={energy_current:.10f}, "
                    f"ΔE_drop={energy_drop:.3e}, "
                    f"VarH={var_h_diag if var_h_diag is not None else 'N/A'}, "
                    f"wall={iter_wall_time:.1f}s, cum={cum_wall_time:.1f}s"
                )
                print(f"  inner stop reason: {inner_stop_reason}")
                if top_k_grads:
                    grad_str = ", ".join(
                        f"[{idx}]={g:.3e}" for idx, g in top_k_grads
                    )
                    print(f"  top-k gradients: {grad_str}")

            if logger is not None:
                dt, tc = logger.end_iter()
                if (outer_idx + 1) % max(1, log_every) == 0:
                    logger.log_point(
                        it=outer_idx + 1,
                        energy=energy_current,
                        max_grad=max_abs_grad,
                        chosen_op=chosen_op,
                        t_iter_s=dt,
                        t_cum_s=tc,
                        extra={
                            "ansatz_len": len(ops),
                            "n_params": len(theta),
                            "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                            "stop_reason": stop_reason,
                            **({"VarH": float(var_h_diag)} if var_h_diag is not None else {}),
                            **(sector_diag or {}),
                        },
                    )
            diagnostics["outer"].append(
                {
                    "outer_iter": len(ops),
                    "chosen_op": chosen_op,
                    "selected_operator": chosen_op,
                    "selected_pool_index": int(max_idx),
                    "chosen_components": chosen_spec["paulis"] if pool_mode in GROUPED_POOL_MODES else None,
                    "max_grad": max_abs_grad,
                    "pre_opt_energy": pre_opt_energy,
                    "energy_pre_inner": pre_opt_energy,
                    "energy": energy_current,
                    "energy_post_inner": energy_current,
                    "energy_drop": energy_drop,
                    "VarH": var_h_diag,
                    "top_k_gradients": top_k_grads,
                    "candidate_topk": top_k_candidates,
                    "inner_stats": inner_stats,
                    "inner_stop_reason": str(inner_stop_reason),
                    "inner_optimizer": inner_optimizer,
                    "selection_strategy": selection_strategy,
                    "lookahead": lookahead_diag,
                    "sector": sector_diag,
                    "iter_wall_time_s": iter_wall_time,
                    "cum_wall_time_s": cum_wall_time,
                }
            )

            # Break if budget was exceeded after inner solve
            if stop_reason is not None and stop_reason.startswith("budget:"):
                break
        except BudgetExceeded as exc:
            # Fallback: catch any remaining BudgetExceeded (e.g. from gradient eval)
            stop_reason = f"budget:{exc.reason}"
            if exc.last_energy is not None:
                energy_current = float(exc.last_energy)
            if logger is not None:
                dt, tc = logger.end_iter()
                if (outer_idx + 1) % max(1, log_every) == 0:
                    logger.log_point(
                        it=outer_idx + 1,
                        energy=energy_current,
                        max_grad=None,
                        chosen_op=None,
                        t_iter_s=dt,
                        t_cum_s=tc,
                        extra={
                            "ansatz_len": len(ops),
                            "n_params": len(theta),
                            "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                            "stop_reason": stop_reason,
                        },
                    )
            break

    diagnostics["stop_reason"] = stop_reason
    diagnostics["selected_sequence"] = _selected_sequence_snapshot(ops)
    diagnostics["n_selected_ops"] = int(len(ops))
    diagnostics["t_total_s"] = float(time.perf_counter() - t0)

    return MetaAdaptVQEResult(
        energy=energy_current,
        params=list(map(float, theta)),
        operators=ops,
        diagnostics=diagnostics,
    )
