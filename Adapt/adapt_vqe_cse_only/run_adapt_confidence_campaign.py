#!/usr/bin/env python3
"""Confidence campaign for ADAPT vs regular hardcoded/Qiskit VQE baselines."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from qiskit.quantum_info import Statevector

try:  # Qiskit >=1.0
    from qiskit.primitives import StatevectorEstimator

    def _make_estimator():
        return StatevectorEstimator()

except ImportError:  # Qiskit 0.x fallback
    from qiskit.primitives import Estimator

    def _make_estimator():
        return Estimator()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    build_reference_state,
    default_1d_chain_edges,
    exact_ground_energy_sector,
    half_filling_sector,
    jw_reference_occupations_from_particles,
    run_meta_adapt_vqe,
)
from src.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian
from src.quantum.vqe_latex_python_pairs import (
    HardcodedUCCSDAnsatz,
    basis_state,
    hartree_fock_bitstring,
    vqe_minimize,
)


@dataclass(frozen=True)
class GateTarget:
    vqe_l2: float = 1e-8
    vqe_l3: float = 1e-6
    adapt_uccsd_fixed_l2: float = 1e-8
    adapt_uccsd_fixed_l3: float = 1e-6
    adapt_uccsd_adapt_l2: float = 1e-7
    adapt_uccsd_adapt_l3: float = 1e-5
    cse_gap_closure_min: float = 0.90
    sector_err_tol: float = 1e-8
    monotonic_tol: float = 1e-9


METHOD_ORDER = [
    "exact",
    "hf",
    "hardcoded_vqe",
    "qiskit_vqe",
    "adapt_uccsd_fixed",
    "adapt_uccsd_adapt",
    "adapt_cse_adapt",
]

METHOD_LABELS = {
    "exact": "Exact (Sector)",
    "hf": "Hartree-Fock Ref",
    "hardcoded_vqe": "Hardcoded VQE (UCCSD)",
    "qiskit_vqe": "Qiskit VQE (UCCSD)",
    "adapt_uccsd_fixed": "ADAPT-UCCSD (Fixed Sequence)",
    "adapt_uccsd_adapt": "ADAPT-UCCSD (Adaptive)",
    "adapt_cse_adapt": "ADAPT-CSE (Adaptive)",
}

BENCHMARK_METHODS = [
    "hardcoded_vqe",
    "qiskit_vqe",
    "adapt_uccsd_fixed",
    "adapt_uccsd_adapt",
    "adapt_cse_adapt",
]

REPORT_TITLE = "ADAPT Confidence Campaign"
REPORT_FIGSIZE = (11.0, 8.5)


def _resolve_sector(n_sites: int, n_up: int | None, n_down: int | None, odd_policy: str) -> tuple[int, int]:
    if n_up is None and n_down is None:
        return half_filling_sector(int(n_sites), odd_policy=str(odd_policy))
    if n_up is None or n_down is None:
        raise ValueError("Set both --n-up and --n-down together, or set neither.")
    return int(n_up), int(n_down)


def _site_potential_default(n_sites: int, dv: float) -> list[float] | None:
    if int(n_sites) == 2 and abs(float(dv)) > 1e-15:
        half = 0.5 * float(dv)
        return [-half, half]
    return None


def _build_problem(
    *,
    n_sites: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
    n_up: int,
    n_down: int,
) -> tuple[Any, Any, Any, Any, Any, list[float] | None]:
    periodic = str(boundary).strip().lower() == "periodic"
    edges = default_1d_chain_edges(int(n_sites), periodic=periodic)
    v_spec = _site_potential_default(int(n_sites), float(dv))

    ferm_op = build_fermionic_hubbard(
        n_sites=int(n_sites),
        t=float(t),
        u=float(u),
        edges=edges,
        v=v_spec,
        indexing=str(ordering),
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    occupations = jw_reference_occupations_from_particles(int(n_sites), int(n_up), int(n_down))
    reference = build_reference_state(qubit_op.num_qubits, occupations)

    h_poly = build_hubbard_hamiltonian(
        dims=int(n_sites),
        t=float(t),
        U=float(u),
        v=v_spec,
        repr_mode="JW",
        indexing=str(ordering),
        edges=edges,
        pbc=periodic,
    )
    return ferm_op, qubit_op, mapper, reference, h_poly, v_spec


def _statevector_energy(qubit_op, circuit) -> float:
    psi = Statevector.from_instruction(circuit)
    return float(np.real(psi.expectation_value(qubit_op)))


def _threshold_for_l(l_value: int, l2: float, l3: float) -> float:
    if int(l_value) <= 2:
        return float(l2)
    return float(l3)


def _adapt_trace_summary(outer: list[dict[str, Any]], monotonic_tol: float) -> dict[str, Any]:
    energies: list[float] = []
    var_h: list[float] = []
    max_abs_n_err = 0.0
    max_abs_sz_err = 0.0
    for entry in outer:
        if isinstance(entry, dict) and entry.get("energy") is not None:
            energies.append(float(entry["energy"]))
        if isinstance(entry, dict) and entry.get("VarH") is not None:
            var_h.append(float(entry["VarH"]))
        sector = entry.get("sector") if isinstance(entry, dict) else None
        if isinstance(sector, dict):
            max_abs_n_err = max(max_abs_n_err, float(abs(sector.get("abs_N_err", 0.0))))
            max_abs_sz_err = max(max_abs_sz_err, float(abs(sector.get("abs_Sz_err", 0.0))))

    monotonic_violations = 0
    monotonic_max_upstep = 0.0
    for prev, curr in zip(energies, energies[1:]):
        up = float(curr - prev)
        if up > float(monotonic_tol):
            monotonic_violations += 1
            monotonic_max_upstep = max(monotonic_max_upstep, up)

    return {
        "outer_energies": energies,
        "outer_var_h": var_h,
        "outer_iters": int(len(outer)),
        "monotonic_violations": int(monotonic_violations),
        "monotonic_max_upstep": float(monotonic_max_upstep),
        "max_abs_N_err": float(max_abs_n_err),
        "max_abs_Sz_err": float(max_abs_sz_err),
    }


def _run_hardcoded_vqe_trial(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    h_poly,
    ordering: str,
    trial_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    num_particles = (int(n_up), int(n_down))
    hf_bits = str(
        hartree_fock_bitstring(
            n_sites=int(n_sites),
            num_particles=num_particles,
            indexing=str(ordering),
        )
    )
    psi_ref = np.asarray(basis_state(2 * int(n_sites), hf_bits), dtype=complex)
    ansatz = HardcodedUCCSDAnsatz(
        dims=int(n_sites),
        num_particles=num_particles,
        reps=int(trial_params["reps"]),
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    result = vqe_minimize(
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(trial_params["restarts"]),
        seed=int(seed),
        method="SLSQP",
        maxiter=int(trial_params["maxiter"]),
        bounds=(-math.pi, math.pi),
    )
    elapsed = time.perf_counter() - start
    return {
        "ok": True,
        "energy": float(result.energy),
        "elapsed_s": float(elapsed),
        "ansatz_len": int(ansatz.num_parameters),
        "trial_params": dict(trial_params),
        "details": {
            "success": bool(result.success),
            "message": str(result.message),
            "best_restart": int(result.best_restart),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "num_parameters": int(ansatz.num_parameters),
        },
    }


def _run_qiskit_vqe_trial(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    qubit_op,
    mapper,
    trial_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    from qiskit_algorithms.minimum_eigensolvers import VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

    start = time.perf_counter()
    estimator = _make_estimator()

    hf = HartreeFock(
        num_spatial_orbitals=int(n_sites),
        num_particles=(int(n_up), int(n_down)),
        qubit_mapper=mapper,
    )
    ansatz = UCCSD(
        num_spatial_orbitals=int(n_sites),
        num_particles=(int(n_up), int(n_down)),
        qubit_mapper=mapper,
        initial_state=hf,
        reps=int(trial_params["reps"]),
    )

    rng = np.random.default_rng(int(seed))
    best_energy = float("inf")
    best_restart = -1
    best_result = None
    for restart in range(max(1, int(trial_params["restarts"]))):
        x0 = 0.3 * rng.normal(size=ansatz.num_parameters)
        optimizer = SLSQP(maxiter=max(1, int(trial_params["maxiter"])))
        solver = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=x0,
        )
        result = solver.compute_minimum_eigenvalue(qubit_op)
        e = float(np.real(result.eigenvalue))
        if e < best_energy:
            best_energy = e
            best_restart = restart
            best_result = result

    elapsed = time.perf_counter() - start
    nfev = None
    nit = None
    if best_result is not None and getattr(best_result, "optimizer_result", None) is not None:
        opt = best_result.optimizer_result
        nfev = getattr(opt, "nfev", None)
        nit = getattr(opt, "nit", None)

    return {
        "ok": True,
        "energy": float(best_energy),
        "elapsed_s": float(elapsed),
        "ansatz_len": int(ansatz.num_parameters),
        "trial_params": dict(trial_params),
        "details": {
            "success": True,
            "method": "qiskit_vqe_uccsd",
            "num_parameters": int(ansatz.num_parameters),
            "best_restart": int(best_restart),
            "nfev": int(nfev) if nfev is not None else None,
            "nit": int(nit) if nit is not None else None,
            "effective_maxiter": int(trial_params["maxiter"]),
        },
    }


def _run_adapt_trial(
    *,
    qubit_op,
    reference,
    ferm_op,
    mapper,
    n_sites: int,
    n_up: int,
    n_down: int,
    trial_params: dict[str, Any],
    pool_mode: str,
    selection_mode: str,
    seed: int,
    ordering: str,
    monotonic_tol: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    estimator = _make_estimator()
    kwargs: dict[str, Any] = {
        "pool_mode": str(pool_mode),
        "ferm_op": ferm_op,
        "mapper": mapper,
        "n_sites": int(n_sites),
        "n_up": int(n_up),
        "n_down": int(n_down),
        "enforce_sector": True,
        "selection_mode": str(selection_mode),
        "selection_strategy": "max_gradient",
        "inner_optimizer": "lbfgs",
        "max_depth": int(trial_params.get("max_depth", 1)),
        "inner_steps": int(trial_params["inner_steps"]),
        "lbfgs_maxiter": int(trial_params["inner_steps"]),
        "lbfgs_restarts": int(trial_params["lbfgs_restarts"]),
        "eps_grad": float(trial_params.get("eps_grad", 1e-6)),
        "eps_energy": float(trial_params.get("eps_energy", 1e-8)),
        "max_time_s": float(trial_params["max_time_s"]) if trial_params.get("max_time_s") is not None else None,
        "theta_bound": "pi",
        "seed": int(seed),
        "allow_repeats": bool(trial_params.get("allow_repeats", str(selection_mode) == "adapt")),
        "compute_var_h": True,
        "verbose": False,
    }
    if str(pool_mode) in {"hardcoded_uccsd_onzots", "uccsd_onzots"}:
        kwargs.update(
            {
                "hardcoded_indexing": str(ordering),
                "grouped_generator_scale": 1.0,
                "grouped_term_order": "canonical_sorted",
            }
        )
    if str(selection_mode) == "fixed_sequence":
        kwargs.update(
            {
                "debug_validate_semantics": True,
                "debug_semantics_trials": int(trial_params.get("debug_semantics_trials", 10)),
                "debug_semantics_theta_std": float(trial_params.get("debug_semantics_theta_std", 0.3)),
            }
        )

    result = run_meta_adapt_vqe(qubit_op, reference, estimator, **kwargs)
    elapsed = time.perf_counter() - start
    outer = result.diagnostics.get("outer", [])
    trace = _adapt_trace_summary(list(outer), monotonic_tol=monotonic_tol)
    return {
        "ok": True,
        "energy": float(result.energy),
        "elapsed_s": float(elapsed),
        "ansatz_len": int(len(result.operators)),
        "trial_params": dict(trial_params),
        "details": {
            "stop_reason": result.diagnostics.get("stop_reason"),
            "pool_mode": str(pool_mode),
            "selection_mode": str(selection_mode),
            "pool_size": result.diagnostics.get("pool_size", [None])[0],
            "grouped_term_order": result.diagnostics.get("grouped_term_order"),
            "grouped_trotter_order": result.diagnostics.get("grouped_trotter_order"),
            "grouped_trotter_reps": result.diagnostics.get("grouped_trotter_reps"),
            "debug_semantics": result.diagnostics.get("debug_semantics"),
            "trace": trace,
        },
    }


def _wrap_trial(method: str, trial: dict[str, Any], exact_energy: float) -> dict[str, Any]:
    if not trial.get("ok", False):
        return {
            "method": str(method),
            "ok": False,
            "error": str(trial.get("error", "unknown trial failure")),
            "trial_params": dict(trial.get("trial_params", {})),
        }
    e = float(trial["energy"])
    return {
        "method": str(method),
        "ok": True,
        "energy": e,
        "exact_energy": float(exact_energy),
        "delta_e": float(e - exact_energy),
        "abs_delta_e": float(abs(e - exact_energy)),
        "elapsed_s": float(trial["elapsed_s"]),
        "ansatz_len": int(trial["ansatz_len"]),
        "trial_params": dict(trial["trial_params"]),
        "details": dict(trial.get("details", {})),
    }


def _select_best(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok_trials = [t for t in trials if t.get("ok", False)]
    if not ok_trials:
        return None
    return min(ok_trials, key=lambda t: (float(t["energy"]), float(t["elapsed_s"])))


def _run_sweep(
    *,
    method: str,
    trial_confs: list[dict[str, Any]],
    per_method_max_time_s: float,
    run_trial_fn,
    exact_energy: float,
) -> dict[str, Any]:
    method_start = time.perf_counter()
    wrapped_trials: list[dict[str, Any]] = []
    for idx, trial_conf in enumerate(trial_confs):
        elapsed_method = time.perf_counter() - method_start
        if elapsed_method >= float(per_method_max_time_s):
            break
        try:
            raw = run_trial_fn(idx, dict(trial_conf), max(0.0, float(per_method_max_time_s) - elapsed_method))
            wrapped = _wrap_trial(method, raw, exact_energy=exact_energy)
        except Exception as exc:  # pragma: no cover - runtime guard
            wrapped = _wrap_trial(
                method,
                {
                    "ok": False,
                    "error": repr(exc),
                    "trial_params": dict(trial_conf),
                },
                exact_energy=exact_energy,
            )
        wrapped_trials.append(wrapped)
    best = _select_best(wrapped_trials)
    return {
        "method": str(method),
        "elapsed_s_total": float(time.perf_counter() - method_start),
        "num_trials": int(len(wrapped_trials)),
        "trials": wrapped_trials,
        "best": best,
    }


def _gap_closure(hf_energy: float, exact_energy: float, candidate_energy: float) -> float | None:
    denom = float(hf_energy - exact_energy)
    if abs(denom) < 1e-14:
        return None
    return float((hf_energy - candidate_energy) / denom)


def _gate_from_best(
    *,
    method: str,
    l_value: int,
    best: dict[str, Any] | None,
    hf_energy: float,
    exact_energy: float,
    gate: GateTarget,
) -> dict[str, Any]:
    if best is None:
        return {"pass": False, "reason": "no_successful_trial"}

    abs_de = float(best["abs_delta_e"])
    details = best.get("details", {})
    trace = details.get("trace", {}) if isinstance(details, dict) else {}
    monotonic_violations = int(trace.get("monotonic_violations", 0))
    max_abs_n_err = float(trace.get("max_abs_N_err", 0.0))
    max_abs_sz_err = float(trace.get("max_abs_Sz_err", 0.0))
    diag_ok = (
        monotonic_violations == 0
        and max_abs_n_err <= float(gate.sector_err_tol)
        and max_abs_sz_err <= float(gate.sector_err_tol)
    )
    closure = _gap_closure(float(hf_energy), float(exact_energy), float(best["energy"]))

    if method in {"hardcoded_vqe", "qiskit_vqe"}:
        thr = _threshold_for_l(int(l_value), gate.vqe_l2, gate.vqe_l3)
        ok = abs_de <= thr
        return {
            "pass": bool(ok),
            "threshold": float(thr),
            "abs_delta_e": float(abs_de),
            "reason": "ok" if ok else "abs_delta_e_above_threshold",
        }

    if method == "adapt_uccsd_fixed":
        thr = _threshold_for_l(int(l_value), gate.adapt_uccsd_fixed_l2, gate.adapt_uccsd_fixed_l3)
        sem = details.get("debug_semantics") if isinstance(details, dict) else None
        sem_ok = bool(isinstance(sem, dict) and float(sem.get("min_fidelity", 0.0)) >= 1.0 - 1e-10)
        ok = abs_de <= thr and diag_ok and sem_ok
        return {
            "pass": bool(ok),
            "threshold": float(thr),
            "abs_delta_e": float(abs_de),
            "diagnostics_ok": bool(diag_ok),
            "semantics_ok": bool(sem_ok),
            "reason": "ok" if ok else "failed_fixed_uccsd_gate",
        }

    if method == "adapt_uccsd_adapt":
        thr = _threshold_for_l(int(l_value), gate.adapt_uccsd_adapt_l2, gate.adapt_uccsd_adapt_l3)
        ok = abs_de <= thr and diag_ok
        return {
            "pass": bool(ok),
            "threshold": float(thr),
            "abs_delta_e": float(abs_de),
            "diagnostics_ok": bool(diag_ok),
            "reason": "ok" if ok else "failed_adaptive_uccsd_gate",
        }

    if method == "adapt_cse_adapt":
        closure_ok = closure is not None and float(closure) >= float(gate.cse_gap_closure_min)
        ok = bool(closure_ok and diag_ok)
        return {
            "pass": bool(ok),
            "gap_closure": float(closure) if closure is not None else None,
            "gap_closure_min": float(gate.cse_gap_closure_min),
            "diagnostics_ok": bool(diag_ok),
            "reason": "ok" if ok else "failed_cse_relative_improvement_gate",
        }

    if method == "hf":
        return {"pass": True, "reason": "reference_only"}
    if method == "exact":
        return {"pass": True, "reason": "reference_only"}
    return {"pass": False, "reason": "unknown_method"}


def _best_energy_or_none(method_block: dict[str, Any]) -> float | None:
    best = method_block.get("best")
    if not isinstance(best, dict):
        return None
    if not best.get("ok", False):
        return None
    return float(best["energy"])


def _best_elapsed_or_none(method_block: dict[str, Any]) -> float | None:
    best = method_block.get("best")
    if not isinstance(best, dict):
        return None
    if not best.get("ok", False):
        return None
    return float(best["elapsed_s"])


def _apply_report_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
        }
    )


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_energy(value: float | None) -> str:
    if value is None:
        return "NA"
    v = float(value)
    if not np.isfinite(v):
        return "NA"
    if abs(v) >= 1e4 or (0.0 < abs(v) < 1e-4):
        return f"{v:.6e}"
    return f"{v:.12f}"


def _fmt_error(value: float | None) -> str:
    if value is None:
        return "NA"
    v = float(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.3e}"


def _fmt_runtime(value: float | None) -> str:
    if value is None:
        return "NA"
    v = float(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.2f}"


def _best_block(entry: dict[str, Any], method: str) -> dict[str, Any] | None:
    method_block = entry["methods"].get(method, {})
    best = method_block.get("best")
    if not isinstance(best, dict) or not best.get("ok", False):
        return None
    return best


def _short_method_label(method: str) -> str:
    return {
        "hardcoded_vqe": "Hardcoded",
        "qiskit_vqe": "Qiskit",
        "adapt_uccsd_fixed": "UCCSD fixed",
        "adapt_uccsd_adapt": "UCCSD adapt",
        "adapt_cse_adapt": "CSE adapt",
        "hf": "HF",
        "exact": "Exact",
    }.get(method, method)


def _derive_fail_reason(method: str, gate_block: dict[str, Any], best: dict[str, Any] | None) -> str:
    if best is None:
        return "no_success"
    details = best.get("details", {}) if isinstance(best, dict) else {}
    stop_reason = details.get("stop_reason") if isinstance(details, dict) else None
    if isinstance(stop_reason, str) and stop_reason.startswith("budget:"):
        return "timeout/cap"

    if bool(gate_block.get("pass", False)):
        return "ok"

    reason = str(gate_block.get("reason", "fail"))
    if method == "adapt_cse_adapt":
        gap = gate_block.get("gap_closure")
        gap_min = gate_block.get("gap_closure_min")
        if gap is None or (gap_min is not None and float(gap) < float(gap_min)):
            return "gap"
        if gate_block.get("diagnostics_ok") is False:
            return "diag"
    else:
        thr = gate_block.get("threshold")
        abs_de = gate_block.get("abs_delta_e")
        if thr is not None and abs_de is not None and float(abs_de) > float(thr):
            return "tol"
        if gate_block.get("diagnostics_ok") is False:
            return "diag"
        if gate_block.get("semantics_ok") is False:
            return "semantics"
    return reason.replace("_", "/")


def _primary_metric_cell(
    method: str,
    l_value: int,
    gate: GateTarget,
    gate_block: dict[str, Any],
    best: dict[str, Any] | None,
) -> str:
    if best is None:
        return "metric=NA"

    if method in {"hardcoded_vqe", "qiskit_vqe"}:
        thr = _threshold_for_l(l_value, gate.vqe_l2, gate.vqe_l3)
        return f"|dE|={_fmt_error(best.get('abs_delta_e'))} <= {_fmt_error(thr)}"
    if method == "adapt_uccsd_fixed":
        thr = _threshold_for_l(l_value, gate.adapt_uccsd_fixed_l2, gate.adapt_uccsd_fixed_l3)
        return f"|dE|={_fmt_error(best.get('abs_delta_e'))} <= {_fmt_error(thr)}"
    if method == "adapt_uccsd_adapt":
        thr = _threshold_for_l(l_value, gate.adapt_uccsd_adapt_l2, gate.adapt_uccsd_adapt_l3)
        return f"|dE|={_fmt_error(best.get('abs_delta_e'))} <= {_fmt_error(thr)}"
    if method == "adapt_cse_adapt":
        closure = gate_block.get("gap_closure")
        return f"gap={_fmt_float(closure, 3)} >= {gate.cse_gap_closure_min:.2f}"
    return "metric=NA"


def _collect_report_warnings(results: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    cap = float(args.per_method_max_time_s)
    for entry in results:
        l_value = int(entry["L"])
        if not np.isfinite(float(entry.get("exact_energy", float("nan")))):
            warnings.append(f"L={l_value}: exact_energy is missing or non-finite.")
        for method in BENCHMARK_METHODS:
            best = _best_block(entry, method)
            if best is None:
                warnings.append(f"L={l_value} {method}: no successful trial.")
                continue
            elapsed = best.get("elapsed_s")
            if elapsed is not None and float(elapsed) > cap + 1e-9:
                warnings.append(
                    f"L={l_value} {method}: best runtime {_fmt_runtime(float(elapsed))}s exceeds cap {_fmt_runtime(cap)}s."
                )
            details = best.get("details", {})
            stop_reason = details.get("stop_reason") if isinstance(details, dict) else None
            if isinstance(stop_reason, str) and stop_reason.startswith("budget:"):
                warnings.append(f"L={l_value} {method}: stopped due to {stop_reason}.")
            abs_de = best.get("abs_delta_e")
            if abs_de is None or not np.isfinite(float(abs_de)):
                warnings.append(f"L={l_value} {method}: abs error missing/non-finite.")
    if len(warnings) > 12:
        extra = len(warnings) - 12
        warnings = warnings[:12] + [f"... and {extra} more warnings."]
    return warnings


def _draw_table(
    ax,
    *,
    col_labels: list[str],
    rows: list[list[str]],
    fontsize: int = 8,
    scale_y: float = 1.35,
    row_colors: list[str] | None = None,
) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, scale_y)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#F2F4F7")
            cell.set_text_props(weight="bold")
        elif row_colors is not None and 0 <= row_idx - 1 < len(row_colors):
            cell.set_facecolor(row_colors[row_idx - 1])


def _dashboard_matrix_rows(results: list[dict[str, Any]], gate: GateTarget) -> tuple[list[str], list[list[str]], list[str]]:
    l_values = [int(entry["L"]) for entry in results]
    col_labels = ["Method"] + [f"L={l_value}" for l_value in l_values]
    rows: list[list[str]] = []
    row_colors: list[str] = []
    for method in BENCHMARK_METHODS:
        row = [METHOD_LABELS[method]]
        pass_all = True
        for entry in results:
            l_value = int(entry["L"])
            gate_block = entry["gates"].get(method, {})
            best = _best_block(entry, method)
            pass_flag = bool(gate_block.get("pass", False))
            pass_all = pass_all and pass_flag
            status = "PASS" if pass_flag else "FAIL"
            metric = _primary_metric_cell(method, l_value, gate, gate_block, best)
            reason = _derive_fail_reason(method, gate_block, best)
            row.append(f"{status}\n{metric}\nreason={reason}")
        rows.append(row)
        row_colors.append("#ECFDF3" if pass_all else "#FEF3F2")
    return col_labels, rows, row_colors


def _summary_table_rows_for_l(entry: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    exact = float(entry["exact_energy"])
    hf = float(entry["hf_energy"])
    rows: list[list[str]] = []

    rows.append(
        [
            "Exact (Sector)",
            _fmt_energy(exact),
            _fmt_error(0.0),
            "NA",
            "0.00",
            "PASS",
            "reference",
        ]
    )
    rows.append(
        [
            "Hartree-Fock",
            _fmt_energy(hf),
            _fmt_error(abs(hf - exact)),
            _fmt_float(0.0, 3),
            "0.00",
            "PASS",
            "reference",
        ]
    )
    for method in BENCHMARK_METHODS:
        best = _best_block(entry, method)
        gate_block = entry["gates"].get(method, {})
        if best is None:
            rows.append(
                [
                    METHOD_LABELS[method],
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "FAIL",
                    _derive_fail_reason(method, gate_block, best),
                ]
            )
            continue
        closure = _gap_closure(hf, exact, float(best["energy"]))
        rows.append(
            [
                METHOD_LABELS[method],
                _fmt_energy(best.get("energy")),
                _fmt_error(best.get("abs_delta_e")),
                _fmt_float(closure, 3),
                _fmt_runtime(best.get("elapsed_s")),
                "PASS" if bool(gate_block.get("pass", False)) else "FAIL",
                _derive_fail_reason(method, gate_block, best),
            ]
        )
    col_labels = ["Method", "E_best", "|dE|", "gap", "Runtime(s)", "Gate", "Reason"]
    return col_labels, rows


def _render_dashboard_page(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    gate: GateTarget,
    generated_utc: str,
    warnings: list[str],
) -> plt.Figure:
    fig = plt.figure(figsize=REPORT_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.5, 2.0], width_ratios=[1.0, 1.0])
    ax_meta = fig.add_subplot(gs[0, :])
    ax_matrix = fig.add_subplot(gs[1, :])
    ax_tbl_left = fig.add_subplot(gs[2, 0])
    ax_tbl_right = fig.add_subplot(gs[2, 1])

    ax_meta.axis("off")
    meta_lines = [
        f"Generated (UTC): {generated_utc}",
        f"Sites: {args.sites} | boundary={args.boundary} | ordering={args.ordering} | seed={args.seed}",
        f"ADAPT allow_repeats: {bool(getattr(args, 'adapt_allow_repeats', True))}",
        (
            "Caps: "
            f"per_method={_fmt_runtime(args.per_method_max_time_s)}s, "
            f"adapt_trial={_fmt_runtime(args.adapt_trial_max_time_s)}s, "
            f"adapt_fixed_trial={_fmt_runtime(args.adapt_fixed_trial_max_time_s)}s"
        ),
        (
            "Gates: "
            f"VQE(L2<={gate.vqe_l2:.1e}, L3<={gate.vqe_l3:.1e}) | "
            f"UCCSD-fixed(L2<={gate.adapt_uccsd_fixed_l2:.1e}, L3<={gate.adapt_uccsd_fixed_l3:.1e}) | "
            f"UCCSD-adapt(L2<={gate.adapt_uccsd_adapt_l2:.1e}, L3<={gate.adapt_uccsd_adapt_l3:.1e}) | "
            f"CSE(gap>={gate.cse_gap_closure_min:.2f})"
        ),
    ]
    if warnings:
        meta_lines.append("Warnings: " + " | ".join(warnings[:3]))
    ax_meta.text(0.01, 0.98, "\n".join(meta_lines), va="top", ha="left", fontsize=9)
    ax_meta.set_title("Summary Dashboard", loc="left", pad=6)

    matrix_cols, matrix_rows, matrix_colors = _dashboard_matrix_rows(results, gate)
    _draw_table(
        ax_matrix,
        col_labels=matrix_cols,
        rows=matrix_rows,
        fontsize=8,
        scale_y=1.45,
        row_colors=matrix_colors,
    )
    ax_matrix.set_title("PASS/FAIL Matrix (Method x L)", loc="left", pad=6)

    entries = list(results)
    if entries:
        cols, rows = _summary_table_rows_for_l(entries[0])
        _draw_table(ax_tbl_left, col_labels=cols, rows=rows, fontsize=7, scale_y=1.26)
        ax_tbl_left.set_title(f"L={int(entries[0]['L'])} Best Results", loc="left", pad=4)
    else:
        ax_tbl_left.axis("off")
    if len(entries) > 1:
        cols, rows = _summary_table_rows_for_l(entries[1])
        _draw_table(ax_tbl_right, col_labels=cols, rows=rows, fontsize=7, scale_y=1.26)
        ax_tbl_right.set_title(f"L={int(entries[1]['L'])} Best Results", loc="left", pad=4)
    else:
        ax_tbl_right.axis("off")
    return fig


def _render_detailed_results_page(results: list[dict[str, Any]]) -> plt.Figure:
    fig = plt.figure(figsize=REPORT_FIGSIZE, constrained_layout=True)
    ax = fig.add_subplot(111)
    rows: list[list[str]] = []
    row_colors: list[str] = []
    for entry in results:
        l_value = int(entry["L"])
        exact = float(entry["exact_energy"])
        hf = float(entry["hf_energy"])
        for method in BENCHMARK_METHODS:
            best = _best_block(entry, method)
            gate_block = entry["gates"].get(method, {})
            pass_flag = bool(gate_block.get("pass", False))
            if best is None:
                rows.append(
                    [
                        f"L={l_value}",
                        METHOD_LABELS[method],
                        "NA",
                        "NA",
                        "NA",
                        "NA",
                        "FAIL",
                        _derive_fail_reason(method, gate_block, best),
                    ]
                )
            else:
                closure = _gap_closure(hf, exact, float(best["energy"]))
                rows.append(
                    [
                        f"L={l_value}",
                        METHOD_LABELS[method],
                        _fmt_energy(best.get("energy")),
                        _fmt_error(best.get("abs_delta_e")),
                        _fmt_float(closure, 3),
                        _fmt_runtime(best.get("elapsed_s")),
                        "PASS" if pass_flag else "FAIL",
                        _derive_fail_reason(method, gate_block, best),
                    ]
                )
            row_colors.append("#ECFDF3" if pass_flag else "#FEF3F2")
    _draw_table(
        ax,
        col_labels=["L", "Method", "E_best", "|dE|", "gap", "Runtime(s)", "Gate", "Reason"],
        rows=rows,
        fontsize=8,
        scale_y=1.35,
        row_colors=row_colors,
    )
    ax.set_title("Detailed Best-Run Table", loc="left", pad=6)
    return fig


def _plot_energy_best(results: list[dict[str, Any]]) -> plt.Figure:
    n_cols = max(1, len(results))
    fig, axes = plt.subplots(1, n_cols, figsize=REPORT_FIGSIZE, constrained_layout=True)
    if n_cols == 1:
        axes = [axes]
    y_methods = BENCHMARK_METHODS + ["hf"]
    y_labels = [_short_method_label(method) for method in y_methods]
    color_map = plt.get_cmap("tab10")

    for ax, entry in zip(axes, results):
        exact = float(entry["exact_energy"])
        hf = float(entry["hf_energy"])
        y = np.arange(len(y_methods), dtype=float)
        x = []
        for method in y_methods:
            if method == "hf":
                x.append(hf)
            else:
                best = _best_block(entry, method)
                x.append(float(best["energy"]) if best is not None else float("nan"))
        for idx, val in enumerate(x):
            ax.scatter(val, y[idx], s=45, color=color_map(idx % 10), zorder=3)
        ax.axvline(exact, color="black", linestyle="--", linewidth=1.3, label="Exact")
        ax.axvline(hf, color="gray", linestyle=":", linewidth=1.1, label="HF")
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.grid(alpha=0.25, axis="x")
        ax.set_xlabel("Energy")
        ax.set_title(f"L={int(entry['L'])} Energy Positions")

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="tab:blue", markersize=6, label="Best method energy"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.3, label="Exact"),
        Line2D([], [], color="gray", linestyle=":", linewidth=1.1, label="HF"),
    ]
    axes[0].legend(handles=handles, loc="lower left")
    fig.suptitle("Energy Overview (Table Companion)")
    return fig


def _plot_abs_error_best(results: list[dict[str, Any]], gate: GateTarget) -> plt.Figure:
    methods = BENCHMARK_METHODS + ["hf"]
    labels = [_short_method_label(method) for method in methods]
    n_cols = max(1, len(results))
    fig, axes = plt.subplots(1, n_cols, figsize=REPORT_FIGSIZE, constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for ax, entry in zip(axes, results):
        l_value = int(entry["L"])
        exact = float(entry["exact_energy"])
        hf = float(entry["hf_energy"])
        vals = []
        for method in methods:
            if method == "hf":
                vals.append(max(abs(hf - exact), 1e-16))
            else:
                best = _best_block(entry, method)
                vals.append(max(float(best["abs_delta_e"]), 1e-16) if best is not None else float("nan"))
        x = np.arange(len(methods), dtype=float)
        ax.bar(x, vals, color="#7BAFD4", edgecolor="#3A6F91")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, axis="y", which="both")
        ax.set_ylabel("|E - E_exact|")
        ax.set_title(f"L={l_value} Absolute Error")
        ax.axhline(
            _threshold_for_l(l_value, gate.vqe_l2, gate.vqe_l3),
            color="tab:green",
            linestyle="--",
            linewidth=1.1,
            label="VQE tol",
        )
        ax.axhline(
            _threshold_for_l(l_value, gate.adapt_uccsd_fixed_l2, gate.adapt_uccsd_fixed_l3),
            color="tab:orange",
            linestyle=":",
            linewidth=1.1,
            label="UCCSD fixed tol",
        )
        ax.axhline(
            _threshold_for_l(l_value, gate.adapt_uccsd_adapt_l2, gate.adapt_uccsd_adapt_l3),
            color="tab:red",
            linestyle="-.",
            linewidth=1.1,
            label="UCCSD adapt tol",
        )
    axes[0].legend(loc="upper left")
    fig.suptitle("Correctness Plot: Absolute Error vs Exact")
    return fig


def _plot_runtime_best(results: list[dict[str, Any]], cap_s: float) -> plt.Figure:
    methods = BENCHMARK_METHODS
    labels = [_short_method_label(method) for method in methods]
    fig, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=REPORT_FIGSIZE, constrained_layout=True)
    x = np.arange(len(methods), dtype=float)
    width = 0.32 if len(results) > 1 else 0.55

    all_vals: list[float] = []
    for ridx, entry in enumerate(results):
        vals = []
        for method in methods:
            elapsed = _best_elapsed_or_none(entry["methods"][method])
            vals.append(float(elapsed) if elapsed is not None else np.nan)
        all_vals.extend([v for v in vals if np.isfinite(v)])
        offset = (ridx - (len(results) - 1) / 2.0) * width
        bar_x = x + offset
        ax_log.bar(bar_x, vals, width=width, label=f"L={int(entry['L'])}")
        ax_lin.bar(bar_x, vals, width=width, label=f"L={int(entry['L'])}")

    ax_log.set_yscale("log")
    ax_log.axhline(cap_s, color="tab:red", linestyle="--", linewidth=1.2, label=f"cap={cap_s:.0f}s")
    ax_log.set_ylabel("Runtime (s, log)")
    ax_log.set_xticks(x)
    ax_log.set_xticklabels(labels, rotation=18, ha="right")
    ax_log.grid(alpha=0.25, axis="y", which="both")
    ax_log.set_title("Runtime by Method (All Scales)")

    ymax = 0.0
    if all_vals:
        ymax = float(np.percentile(np.asarray(all_vals, dtype=float), 80))
    ymax = max(ymax * 1.35, 1.0)
    ax_lin.axhline(cap_s, color="tab:red", linestyle="--", linewidth=1.2)
    ax_lin.set_ylim(0.0, min(cap_s, ymax))
    ax_lin.set_ylabel("Runtime (s, zoom)")
    ax_lin.set_xticks(x)
    ax_lin.set_xticklabels(labels, rotation=18, ha="right")
    ax_lin.grid(alpha=0.25, axis="y")
    ax_lin.set_title("Runtime Zoom (small runtimes visible)")
    ax_log.legend(loc="upper left")
    return fig


def _plot_gap_closure(results: list[dict[str, Any]], gate: GateTarget) -> plt.Figure:
    methods = BENCHMARK_METHODS
    labels = [_short_method_label(method) for method in methods]
    fig, ax = plt.subplots(figsize=REPORT_FIGSIZE, constrained_layout=True)
    x = np.arange(len(methods), dtype=float)
    offsets = np.linspace(-0.16, 0.16, num=max(1, len(results)))
    for ridx, entry in enumerate(results):
        exact = float(entry["exact_energy"])
        hf = float(entry["hf_energy"])
        ys = []
        for method in methods:
            best = _best_block(entry, method)
            if best is None:
                ys.append(np.nan)
            else:
                ys.append(_gap_closure(hf, exact, float(best["energy"])))
        ax.scatter(x + offsets[ridx], ys, s=50, label=f"L={int(entry['L'])}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Exact closure")
    ax.axhline(gate.cse_gap_closure_min, color="tab:gray", linestyle=":", linewidth=1.2, label="CSE gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Gap Closure vs HF")
    ax.set_title("Gap Closure by Method")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left")
    return fig


def _convergence_rows(
    *,
    method: str,
    best: dict[str, Any] | None,
    exact: float,
) -> tuple[int, list[float], list[str]]:
    notes: list[str] = []
    if best is None:
        notes.append("no successful trial")
        return 0, [], notes
    details = best.get("details", {})
    trace = details.get("trace", {}) if isinstance(details, dict) else {}
    energies = list(trace.get("outer_energies", []))
    if not energies:
        energies = [float(best["energy"])]
    n_iter = len(energies)
    if n_iter < 2:
        notes.append("stopped after 1 iteration")
    stop_reason = details.get("stop_reason") if isinstance(details, dict) else None
    if stop_reason is not None:
        notes.append(f"stop={stop_reason}")
    return n_iter, [max(abs(float(e) - exact), 1e-16) for e in energies], notes


def _draw_short_run_table(ax, rows: list[list[str]]) -> None:
    _draw_table(
        ax,
        col_labels=["Method", "N_iter", "Final |dE|", "Runtime(s)", "Stop/Note"],
        rows=rows,
        fontsize=8,
        scale_y=1.25,
    )


def _plot_adapt_uccsd_convergence(entry: dict[str, Any]) -> plt.Figure:
    l_value = int(entry["L"])
    exact = float(entry["exact_energy"])
    fig = plt.figure(figsize=REPORT_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])

    plotted = False
    short_rows: list[list[str]] = []
    max_iter = 0
    for method in ["adapt_uccsd_adapt", "adapt_uccsd_fixed"]:
        best = _best_block(entry, method)
        n_iter, errors, notes = _convergence_rows(method=method, best=best, exact=exact)
        if n_iter >= 2:
            xs = np.arange(1, n_iter + 1, dtype=int)
            ax.plot(xs, errors, marker="o", label=METHOD_LABELS[method])
            plotted = True
            max_iter = max(max_iter, n_iter)
        else:
            runtime = _fmt_runtime(best.get("elapsed_s")) if best is not None else "NA"
            final_err = _fmt_error(best.get("abs_delta_e")) if best is not None else "NA"
            short_rows.append(
                [METHOD_LABELS[method], str(n_iter), final_err, runtime, "; ".join(notes) if notes else "short run"]
            )

    if plotted:
        ax.set_yscale("log")
        ax.set_xticks(np.arange(1, max_iter + 1, dtype=int))
        ax.set_xlabel("Outer Iteration")
        ax.set_ylabel("|E - E_exact|")
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="best")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No multi-iteration UCCSD convergence traces.", ha="center", va="center")
    ax.set_title(f"ADAPT-UCCSD Convergence (L={l_value})")

    if short_rows:
        _draw_short_run_table(ax_tbl, short_rows)
    else:
        ax_tbl.axis("off")
        ax_tbl.text(0.02, 0.95, "All shown traces have >=2 iterations.", va="top", ha="left", fontsize=9)
    return fig


def _plot_adapt_cse_convergence(entry: dict[str, Any]) -> plt.Figure:
    l_value = int(entry["L"])
    exact = float(entry["exact_energy"])
    fig = plt.figure(figsize=REPORT_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])

    best = _best_block(entry, "adapt_cse_adapt")
    n_iter, errors, notes = _convergence_rows(method="adapt_cse_adapt", best=best, exact=exact)
    if n_iter >= 2:
        xs = np.arange(1, n_iter + 1, dtype=int)
        ax.plot(xs, errors, marker="o", label=METHOD_LABELS["adapt_cse_adapt"])
        ax.set_yscale("log")
        ax.set_xticks(np.arange(1, n_iter + 1, dtype=int))
        ax.set_xlabel("Outer Iteration")
        ax.set_ylabel("|E - E_exact|")
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="best")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No multi-iteration CSE convergence trace.", ha="center", va="center")
    ax.set_title(f"ADAPT-CSE Convergence (L={l_value})")

    runtime = _fmt_runtime(best.get("elapsed_s")) if best is not None else "NA"
    final_err = _fmt_error(best.get("abs_delta_e")) if best is not None else "NA"
    _draw_short_run_table(
        ax_tbl,
        [[METHOD_LABELS["adapt_cse_adapt"], str(n_iter), final_err, runtime, "; ".join(notes) if notes else "ok"]],
    )
    return fig


def _render_appendix_page(results: list[dict[str, Any]], warnings: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=REPORT_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.6])
    ax_table = fig.add_subplot(gs[0, 0])
    ax_warn = fig.add_subplot(gs[1, 0])

    rows: list[list[str]] = []
    for entry in results:
        l_value = int(entry["L"])
        exact = float(entry["exact_energy"])
        hf = float(entry["hf_energy"])
        for method in BENCHMARK_METHODS:
            best = _best_block(entry, method)
            if best is None:
                rows.append([f"L={l_value}", _short_method_label(method), "NA", "NA", "NA", "NA"])
            else:
                closure = _gap_closure(hf, exact, float(best["energy"]))
                rows.append(
                    [
                        f"L={l_value}",
                        _short_method_label(method),
                        _fmt_energy(best.get("energy")),
                        _fmt_error(best.get("abs_delta_e")),
                        _fmt_float(closure, 3),
                        _fmt_runtime(best.get("elapsed_s")),
                    ]
                )
    _draw_table(
        ax_table,
        col_labels=["L", "Method", "E_best", "|dE|", "gap", "Runtime(s)"],
        rows=rows,
        fontsize=8,
        scale_y=1.2,
    )
    ax_table.set_title("Appendix: Raw Best Values", loc="left", pad=6)

    ax_warn.axis("off")
    warn_lines = warnings if warnings else ["No report-side warnings detected."]
    text = "\n".join(f"- {line}" for line in warn_lines)
    ax_warn.text(0.01, 0.98, text, va="top", ha="left", fontsize=9)
    ax_warn.set_title("Appendix: Warnings", loc="left", pad=6)
    return fig


def _apply_page_chrome(fig: plt.Figure, *, generated_utc: str, page_no: int, page_total: int) -> None:
    fig.text(0.01, 0.992, REPORT_TITLE, ha="left", va="top", fontsize=10, weight="bold")
    fig.text(0.99, 0.992, f"Generated UTC: {generated_utc}", ha="right", va="top", fontsize=8)
    fig.text(0.99, 0.012, f"Page {page_no}/{page_total}", ha="right", va="bottom", fontsize=8)


def _rows_for_best_csv(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in results:
        l_value = int(entry["L"])
        exact = float(entry["exact_energy"])
        hf = float(entry["hf_energy"])
        for method in METHOD_ORDER:
            if method == "exact":
                rows.append(
                    {
                        "L": l_value,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "energy": exact,
                        "exact_energy": exact,
                        "abs_delta_e": 0.0,
                        "elapsed_s": 0.0,
                        "gap_closure": None,
                        "gate_pass": True,
                        "gate_reason": "reference_only",
                    }
                )
                continue
            if method == "hf":
                rows.append(
                    {
                        "L": l_value,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "energy": hf,
                        "exact_energy": exact,
                        "abs_delta_e": abs(hf - exact),
                        "elapsed_s": 0.0,
                        "gap_closure": 0.0,
                        "gate_pass": True,
                        "gate_reason": "reference_only",
                    }
                )
                continue
            block = entry["methods"][method]
            best = block.get("best")
            gate = entry["gates"].get(method, {})
            if not isinstance(best, dict) or not best.get("ok", False):
                rows.append(
                    {
                        "L": l_value,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "energy": None,
                        "exact_energy": exact,
                        "abs_delta_e": None,
                        "elapsed_s": None,
                        "gap_closure": None,
                        "gate_pass": bool(gate.get("pass", False)),
                        "gate_reason": _derive_fail_reason(method, gate, None),
                    }
                )
                continue
            closure = _gap_closure(float(hf), float(exact), float(best["energy"]))
            rows.append(
                {
                    "L": l_value,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "energy": float(best["energy"]),
                    "exact_energy": exact,
                    "abs_delta_e": float(best["abs_delta_e"]),
                    "elapsed_s": float(best["elapsed_s"]),
                    "gap_closure": float(closure) if closure is not None else None,
                    "gate_pass": bool(gate.get("pass", False)),
                    "gate_reason": _derive_fail_reason(method, gate, best),
                }
            )
    return rows


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep confidence campaign for ADAPT-CSE/UCCSD vs hardcoded/Qiskit VQE baselines."
    )
    parser.add_argument("--sites", type=int, nargs="+", default=[2, 3], help="List of L values.")
    parser.add_argument("--n-up", type=int, default=None)
    parser.add_argument("--n-down", type=int, default=None)
    parser.add_argument("--odd-policy", choices=["min_sz", "restrict", "dope_sz0"], default="min_sz")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument("--boundary", choices=["open", "periodic"], default="open")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--per-method-max-time-s", type=float, default=1200.0)
    parser.add_argument("--adapt-trial-max-time-s", type=float, default=300.0)
    parser.add_argument("--adapt-fixed-trial-max-time-s", type=float, default=600.0)
    parser.add_argument("--sweep-profile", choices=["deep", "fast"], default="deep")
    parser.set_defaults(adapt_allow_repeats=True)
    parser.add_argument(
        "--adapt-allow-repeats",
        dest="adapt_allow_repeats",
        action="store_true",
        help="Allow ADAPT to reselect previously chosen operators (default: enabled).",
    )
    parser.add_argument(
        "--adapt-no-repeats",
        dest="adapt_allow_repeats",
        action="store_false",
        help="Disable ADAPT operator reuse.",
    )

    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/adapt_confidence_campaign_L2_L3"))
    return parser.parse_args()


def _build_trial_lists(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    if args.sweep_profile == "fast":
        vqe_restarts = [1, 2]
        vqe_maxiter = [200, 600]
        adapt_depth = [8, 12]
        adapt_steps = [250, 600]
        adapt_restarts = [1, 3]
        fixed_steps = [600, 1200]
        fixed_restarts = [1, 3]
    else:
        vqe_restarts = [1, 3, 5, 10]
        vqe_maxiter = [600, 2000, 5000]
        adapt_depth = [8, 12, 20, 32]
        adapt_steps = [250, 600, 1200]
        adapt_restarts = [1, 3, 5]
        fixed_steps = [600, 1200, 2000]
        fixed_restarts = [1, 3, 5, 10]

    hardcoded_trials = [
        {"reps": 2, "restarts": int(r), "maxiter": int(mi)}
        for r, mi in itertools.product(vqe_restarts, vqe_maxiter)
    ]
    qiskit_trials = [
        {"reps": 2, "restarts": int(r), "maxiter": int(mi)}
        for r, mi in itertools.product(vqe_restarts, vqe_maxiter)
    ]
    adapt_uccsd_trials = [
        {
            "max_depth": int(d),
            "inner_steps": int(s),
            "lbfgs_restarts": int(r),
        }
        for d, s, r in itertools.product(adapt_depth, adapt_steps, adapt_restarts)
    ]
    adapt_cse_trials = [
        {
            "max_depth": int(d),
            "inner_steps": int(s),
            "lbfgs_restarts": int(r),
        }
        for d, s, r in itertools.product(adapt_depth, adapt_steps, adapt_restarts)
    ]
    adapt_fixed_trials = [
        {
            "max_depth": 1,
            "inner_steps": int(s),
            "lbfgs_restarts": int(r),
            "debug_semantics_trials": 10,
            "debug_semantics_theta_std": 0.3,
        }
        for s, r in itertools.product(fixed_steps, fixed_restarts)
    ]
    return {
        "hardcoded_vqe": hardcoded_trials,
        "qiskit_vqe": qiskit_trials,
        "adapt_uccsd_adapt": adapt_uccsd_trials,
        "adapt_cse_adapt": adapt_cse_trials,
        "adapt_uccsd_fixed": adapt_fixed_trials,
    }


def main() -> None:
    args = parse_args()
    args.sites = [int(x) for x in args.sites]
    gate = GateTarget()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_lists = _build_trial_lists(args)
    all_results: list[dict[str, Any]] = []

    for n_sites in args.sites:
        n_up, n_down = _resolve_sector(int(n_sites), args.n_up, args.n_down, args.odd_policy)
        ferm_op, qubit_op, mapper, reference, h_poly, v_spec = _build_problem(
            n_sites=int(n_sites),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            ordering=str(args.ordering),
            boundary=str(args.boundary),
            n_up=int(n_up),
            n_down=int(n_down),
        )
        exact = exact_ground_energy_sector(
            qubit_op,
            int(n_sites),
            int(n_up) + int(n_down),
            0.5 * (int(n_up) - int(n_down)),
        )
        hf_energy = _statevector_energy(qubit_op, reference)

        methods: dict[str, dict[str, Any]] = {}

        methods["hardcoded_vqe"] = _run_sweep(
            method="hardcoded_vqe",
            trial_confs=trial_lists["hardcoded_vqe"],
            per_method_max_time_s=float(args.per_method_max_time_s),
            exact_energy=float(exact),
            run_trial_fn=lambda idx, conf, _remaining: _run_hardcoded_vqe_trial(
                n_sites=int(n_sites),
                n_up=int(n_up),
                n_down=int(n_down),
                h_poly=h_poly,
                ordering=str(args.ordering),
                trial_params=conf,
                seed=int(args.seed + idx),
            ),
        )

        methods["qiskit_vqe"] = _run_sweep(
            method="qiskit_vqe",
            trial_confs=trial_lists["qiskit_vqe"],
            per_method_max_time_s=float(args.per_method_max_time_s),
            exact_energy=float(exact),
            run_trial_fn=lambda idx, conf, _remaining: _run_qiskit_vqe_trial(
                n_sites=int(n_sites),
                n_up=int(n_up),
                n_down=int(n_down),
                qubit_op=qubit_op,
                mapper=mapper,
                trial_params=conf,
                seed=int(args.seed + idx),
            ),
        )

        methods["adapt_uccsd_fixed"] = _run_sweep(
            method="adapt_uccsd_fixed",
            trial_confs=trial_lists["adapt_uccsd_fixed"],
            per_method_max_time_s=float(args.per_method_max_time_s),
            exact_energy=float(exact),
            run_trial_fn=lambda idx, conf, remaining: _run_adapt_trial(
                qubit_op=qubit_op,
                reference=reference,
                ferm_op=ferm_op,
                mapper=mapper,
                n_sites=int(n_sites),
                n_up=int(n_up),
                n_down=int(n_down),
                trial_params={
                    **conf,
                    "max_time_s": min(float(args.adapt_fixed_trial_max_time_s), float(remaining)),
                    "eps_grad": 1e-6,
                    "eps_energy": 1e-8,
                },
                pool_mode="hardcoded_uccsd_onzots",
                selection_mode="fixed_sequence",
                seed=int(args.seed + idx),
                ordering=str(args.ordering),
                monotonic_tol=float(gate.monotonic_tol),
            ),
        )

        methods["adapt_uccsd_adapt"] = _run_sweep(
            method="adapt_uccsd_adapt",
            trial_confs=trial_lists["adapt_uccsd_adapt"],
            per_method_max_time_s=float(args.per_method_max_time_s),
            exact_energy=float(exact),
            run_trial_fn=lambda idx, conf, remaining: _run_adapt_trial(
                qubit_op=qubit_op,
                reference=reference,
                ferm_op=ferm_op,
                mapper=mapper,
                n_sites=int(n_sites),
                n_up=int(n_up),
                n_down=int(n_down),
                trial_params={
                    **conf,
                    "max_time_s": min(float(args.adapt_trial_max_time_s), float(remaining)),
                    "eps_grad": 1e-6,
                    "eps_energy": 1e-8,
                    "allow_repeats": bool(args.adapt_allow_repeats),
                },
                pool_mode="hardcoded_uccsd_onzots",
                selection_mode="adapt",
                seed=int(args.seed + idx),
                ordering=str(args.ordering),
                monotonic_tol=float(gate.monotonic_tol),
            ),
        )

        methods["adapt_cse_adapt"] = _run_sweep(
            method="adapt_cse_adapt",
            trial_confs=trial_lists["adapt_cse_adapt"],
            per_method_max_time_s=float(args.per_method_max_time_s),
            exact_energy=float(exact),
            run_trial_fn=lambda idx, conf, remaining: _run_adapt_trial(
                qubit_op=qubit_op,
                reference=reference,
                ferm_op=ferm_op,
                mapper=mapper,
                n_sites=int(n_sites),
                n_up=int(n_up),
                n_down=int(n_down),
                trial_params={
                    **conf,
                    "max_time_s": min(float(args.adapt_trial_max_time_s), float(remaining)),
                    "eps_grad": 1e-6,
                    "eps_energy": 1e-8,
                    "allow_repeats": bool(args.adapt_allow_repeats),
                },
                pool_mode="cse_density_ops",
                selection_mode="adapt",
                seed=int(args.seed + idx),
                ordering=str(args.ordering),
                monotonic_tol=float(gate.monotonic_tol),
            ),
        )

        gates = {
            "hardcoded_vqe": _gate_from_best(
                method="hardcoded_vqe",
                l_value=int(n_sites),
                best=methods["hardcoded_vqe"].get("best"),
                hf_energy=float(hf_energy),
                exact_energy=float(exact),
                gate=gate,
            ),
            "qiskit_vqe": _gate_from_best(
                method="qiskit_vqe",
                l_value=int(n_sites),
                best=methods["qiskit_vqe"].get("best"),
                hf_energy=float(hf_energy),
                exact_energy=float(exact),
                gate=gate,
            ),
            "adapt_uccsd_fixed": _gate_from_best(
                method="adapt_uccsd_fixed",
                l_value=int(n_sites),
                best=methods["adapt_uccsd_fixed"].get("best"),
                hf_energy=float(hf_energy),
                exact_energy=float(exact),
                gate=gate,
            ),
            "adapt_uccsd_adapt": _gate_from_best(
                method="adapt_uccsd_adapt",
                l_value=int(n_sites),
                best=methods["adapt_uccsd_adapt"].get("best"),
                hf_energy=float(hf_energy),
                exact_energy=float(exact),
                gate=gate,
            ),
            "adapt_cse_adapt": _gate_from_best(
                method="adapt_cse_adapt",
                l_value=int(n_sites),
                best=methods["adapt_cse_adapt"].get("best"),
                hf_energy=float(hf_energy),
                exact_energy=float(exact),
                gate=gate,
            ),
            "exact": {"pass": True, "reason": "reference_only"},
            "hf": {"pass": True, "reason": "reference_only"},
        }

        all_results.append(
            {
                "L": int(n_sites),
                "n_up": int(n_up),
                "n_down": int(n_down),
                "exact_energy": float(exact),
                "hf_energy": float(hf_energy),
                "hamiltonian": {
                    "t": float(args.t),
                    "u": float(args.u),
                    "dv_cli": float(args.dv),
                    "boundary": str(args.boundary),
                    "ordering": str(args.ordering),
                    "site_potential": v_spec,
                },
                "methods": methods,
                "gates": gates,
            }
        )

    all_results.sort(key=lambda r: int(r["L"]))
    gate_summary: dict[str, Any] = {}
    for method in METHOD_ORDER:
        if method in {"exact", "hf"}:
            continue
        per_l = [bool(entry["gates"][method]["pass"]) for entry in all_results]
        gate_summary[method] = {
            "pass_all_L": bool(all(per_l)),
            "pass_count": int(sum(1 for x in per_l if x)),
            "num_L": int(len(per_l)),
        }

    generated_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_utc": generated_utc,
        "settings": {
            "sites": list(args.sites),
            "n_up": args.n_up,
            "n_down": args.n_down,
            "odd_policy": str(args.odd_policy),
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "seed": int(args.seed),
            "per_method_max_time_s": float(args.per_method_max_time_s),
            "adapt_trial_max_time_s": float(args.adapt_trial_max_time_s),
            "adapt_fixed_trial_max_time_s": float(args.adapt_fixed_trial_max_time_s),
            "adapt_allow_repeats": bool(args.adapt_allow_repeats),
            "sweep_profile": str(args.sweep_profile),
            "gate_targets": vars(gate),
        },
        "results": all_results,
        "gate_summary": gate_summary,
    }

    json_path = out_dir / "campaign_results.json"
    csv_path = out_dir / "campaign_best.csv"
    pdf_path = out_dir / "campaign_report.pdf"
    energy_png = out_dir / "energy_vs_exact_best.png"
    abs_err_png = out_dir / "abs_error_vs_exact_best_log.png"
    runtime_png = out_dir / "runtime_best.png"
    gap_png = out_dir / "gap_closure_vs_method.png"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = _rows_for_best_csv(all_results)
    _save_csv(csv_path, rows)

    _apply_report_style()

    fig_energy = _plot_energy_best(all_results)
    fig_abs = _plot_abs_error_best(all_results, gate)
    fig_runtime = _plot_runtime_best(all_results, cap_s=float(args.per_method_max_time_s))
    fig_gap = _plot_gap_closure(all_results, gate)

    fig_energy.savefig(energy_png, dpi=180)
    fig_abs.savefig(abs_err_png, dpi=180)
    fig_runtime.savefig(runtime_png, dpi=180)
    fig_gap.savefig(gap_png, dpi=180)

    per_l_plots: list[tuple[str, plt.Figure]] = []
    for entry in all_results:
        l_value = int(entry["L"])
        fig_u = _plot_adapt_uccsd_convergence(entry)
        fig_c = _plot_adapt_cse_convergence(entry)
        path_u = out_dir / f"adapt_uccsd_convergence_L{l_value}.png"
        path_c = out_dir / f"adapt_cse_convergence_L{l_value}.png"
        fig_u.savefig(path_u, dpi=180)
        fig_c.savefig(path_c, dpi=180)
        per_l_plots.append((str(path_u), fig_u))
        per_l_plots.append((str(path_c), fig_c))

    warnings = _collect_report_warnings(all_results, args)
    fig_dashboard = _render_dashboard_page(
        all_results,
        args=args,
        gate=gate,
        generated_utc=generated_utc,
        warnings=warnings,
    )
    fig_detail = _render_detailed_results_page(all_results)
    fig_appendix = _render_appendix_page(all_results, warnings)
    report_pages: list[plt.Figure] = [fig_dashboard, fig_detail, fig_abs, fig_runtime, fig_gap, fig_energy]
    report_pages.extend([fig for _path, fig in per_l_plots])
    report_pages.append(fig_appendix)

    with PdfPages(pdf_path) as pdf:
        total_pages = len(report_pages)
        for idx, fig in enumerate(report_pages, start=1):
            _apply_page_chrome(fig, generated_utc=generated_utc, page_no=idx, page_total=total_pages)
            pdf.savefig(fig)

    plt.close(fig_dashboard)
    plt.close(fig_detail)
    plt.close(fig_energy)
    plt.close(fig_abs)
    plt.close(fig_runtime)
    plt.close(fig_gap)
    for _path, fig in per_l_plots:
        plt.close(fig)
    plt.close(fig_appendix)

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote PDF:  {pdf_path}")
    print(f"Wrote PNG:  {energy_png}")
    print(f"Wrote PNG:  {abs_err_png}")
    print(f"Wrote PNG:  {runtime_png}")
    print(f"Wrote PNG:  {gap_png}")
    for entry in all_results:
        l_value = int(entry["L"])
        print(f"Wrote PNG:  {out_dir / f'adapt_uccsd_convergence_L{l_value}.png'}")
        print(f"Wrote PNG:  {out_dir / f'adapt_cse_convergence_L{l_value}.png'}")


if __name__ == "__main__":
    main()
