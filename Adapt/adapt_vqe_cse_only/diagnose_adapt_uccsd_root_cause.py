#!/usr/bin/env python3
"""Root-cause diagnostics for ADAPT-UCCSD: selection vs inner optimization/budget.

This script runs a deterministic matrix and writes a machine-readable summary plus
CSV/PDF artifacts so we can decide whether the dominant failure mode is:
- selection
- optimization_budget
- mixed
"""

from __future__ import annotations

import argparse
import csv
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


@dataclass
class Problem:
    n_sites: int
    n_up: int
    n_down: int
    exact_energy: float
    qubit_op: Any
    mapper: Any
    reference: Any
    ferm_op: Any
    ordering: str
    h_poly: Any


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "adapt_max_depth": 8,
        "adapt_inner_steps": 40,
        "adapt_lbfgs_restarts": 1,
        "baseline_max_time_s": 120.0,
        "budget_ladder": [30.0, 60.0, 120.0],
        "generous_inner_steps": 180,
        "generous_lbfgs_restarts": 2,
        "matched_depth": 5,
        "prefix_depths": [3, 5],
        "l3_max_depth": 6,
        "l3_max_time_s": 90.0,
    },
    "practical": {
        "adapt_max_depth": 14,
        "adapt_inner_steps": 90,
        "adapt_lbfgs_restarts": 1,
        "baseline_max_time_s": 1800.0,
        "budget_ladder": [300.0, 900.0, 1800.0],
        "generous_inner_steps": 500,
        "generous_lbfgs_restarts": 3,
        "matched_depth": 8,
        "prefix_depths": [3, 5, 8],
        "l3_max_depth": 10,
        "l3_max_time_s": 600.0,
    },
    "heavy": {
        "adapt_max_depth": 23,
        "adapt_inner_steps": 120,
        "adapt_lbfgs_restarts": 1,
        "baseline_max_time_s": 12000.0,
        "budget_ladder": [300.0, 1200.0, 3600.0, 12000.0],
        "generous_inner_steps": 1200,
        "generous_lbfgs_restarts": 4,
        "matched_depth": 12,
        "prefix_depths": [3, 5, 8, 12],
        "l3_max_depth": 12,
        "l3_max_time_s": 1200.0,
    },
}


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
    n_up: int,
    n_down: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
) -> Problem:
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

    exact = exact_ground_energy_sector(
        qubit_op,
        int(n_sites),
        int(n_up) + int(n_down),
        0.5 * (int(n_up) - int(n_down)),
    )
    return Problem(
        n_sites=int(n_sites),
        n_up=int(n_up),
        n_down=int(n_down),
        exact_energy=float(exact),
        qubit_op=qubit_op,
        mapper=mapper,
        reference=reference,
        ferm_op=ferm_op,
        ordering=str(ordering),
        h_poly=h_poly,
    )


def _tail_var_h(result) -> float | None:
    outer = result.diagnostics.get("outer", [])
    if outer and isinstance(outer[-1], dict):
        v = outer[-1].get("VarH")
        if v is not None:
            return float(v)
    return None


def _outer_energies(result) -> list[float]:
    out: list[float] = []
    for entry in result.diagnostics.get("outer", []):
        if isinstance(entry, dict) and entry.get("energy") is not None:
            out.append(float(entry["energy"]))
    return out


def _sequence_depth(result) -> int:
    return int(len(getattr(result, "operators", [])))


def _serialize_sequence(operators: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, spec in enumerate(operators):
        if isinstance(spec, dict):
            paulis = []
            for item in spec.get("paulis", []):
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    paulis.append([str(item[0]), float(item[1])])
            out.append({"index": int(idx), "name": str(spec.get("name", f"op_{idx}")), "paulis": paulis})
        else:
            out.append({"index": int(idx), "name": str(spec), "paulis": []})
    return out


def _summarize_run(label: str, result, exact_energy: float) -> dict[str, Any]:
    energy = float(result.energy)
    abs_delta = float(abs(energy - float(exact_energy)))
    return {
        "label": str(label),
        "energy": energy,
        "exact_energy": float(exact_energy),
        "abs_delta_e": abs_delta,
        "delta_e": float(energy - float(exact_energy)),
        "ansatz_len": int(_sequence_depth(result)),
        "depth": int(_sequence_depth(result)),
        "VarH": _tail_var_h(result),
        "outer_iters": int(len(result.diagnostics.get("outer", []))),
        "stop_reason": result.diagnostics.get("stop_reason"),
        "t_total_s": float(result.diagnostics.get("t_total_s", float("nan"))),
        "outer_energies": _outer_energies(result),
    }


def _run_adapt_uccsd(
    problem: Problem,
    *,
    grouped_selection_gradient: str,
    grouped_selection_fd_delta: float,
    max_depth: int,
    inner_steps: int,
    lbfgs_restarts: int,
    max_time_s: float | None,
    seed: int,
    eps_grad: float,
    eps_energy: float,
    verbose: bool,
):
    estimator = _make_estimator()
    t_start = time.perf_counter()
    result = run_meta_adapt_vqe(
        problem.qubit_op,
        problem.reference,
        estimator,
        pool_mode="hardcoded_uccsd_onzots",
        ferm_op=problem.ferm_op,
        mapper=problem.mapper,
        n_sites=int(problem.n_sites),
        n_up=int(problem.n_up),
        n_down=int(problem.n_down),
        enforce_sector=True,
        selection_mode="adapt",
        selection_strategy="max_gradient",
        inner_optimizer="lbfgs",
        max_depth=int(max_depth),
        inner_steps=int(inner_steps),
        lbfgs_maxiter=int(inner_steps),
        lbfgs_restarts=int(lbfgs_restarts),
        eps_grad=float(eps_grad),
        eps_energy=float(eps_energy),
        max_time_s=float(max_time_s) if max_time_s is not None else None,
        theta_bound="pi",
        seed=int(seed),
        allow_repeats=True,
        compute_var_h=True,
        hardcoded_indexing=str(problem.ordering),
        grouped_term_order="canonical_sorted",
        grouped_generator_scale=1.0,
        grouped_selection_gradient=str(grouped_selection_gradient),
        grouped_selection_fd_delta=float(grouped_selection_fd_delta),
        verbose=bool(verbose),
    )
    summary = _summarize_run("adapt_uccsd", result, problem.exact_energy)
    summary["elapsed_wall_s"] = float(time.perf_counter() - t_start)
    return result, summary


def _run_fixed_sequence(
    problem: Problem,
    sequence_ops: list[Any],
    *,
    inner_optimizer: str,
    inner_steps: int,
    lbfgs_restarts: int,
    max_time_s: float | None,
    seed: int,
    verbose: bool,
):
    estimator = _make_estimator()
    t_start = time.perf_counter()
    result = run_meta_adapt_vqe(
        problem.qubit_op,
        problem.reference,
        estimator,
        pool=list(sequence_ops),
        pool_mode="hardcoded_uccsd_onzots",
        mapper=problem.mapper,
        n_sites=int(problem.n_sites),
        n_up=int(problem.n_up),
        n_down=int(problem.n_down),
        enforce_sector=True,
        selection_mode="fixed_sequence",
        inner_optimizer=str(inner_optimizer),
        theta_bound="pi",
        inner_steps=int(inner_steps),
        lbfgs_restarts=int(lbfgs_restarts),
        lbfgs_maxiter=int(inner_steps),
        max_time_s=float(max_time_s) if max_time_s is not None else None,
        seed=int(seed),
        allow_repeats=True,
        compute_var_h=True,
        hardcoded_indexing=str(problem.ordering),
        grouped_term_order="canonical_sorted",
        grouped_generator_scale=1.0,
        verbose=bool(verbose),
    )
    summary = _summarize_run("fixed_sequence", result, problem.exact_energy)
    summary["elapsed_wall_s"] = float(time.perf_counter() - t_start)
    return result, summary


def _run_hardcoded_vqe(problem: Problem, *, reps: int, restarts: int, maxiter: int, seed: int) -> dict[str, Any]:
    t_start = time.perf_counter()
    num_particles = (int(problem.n_up), int(problem.n_down))
    hf_bits = str(
        hartree_fock_bitstring(
            n_sites=int(problem.n_sites),
            num_particles=num_particles,
            indexing=str(problem.ordering),
        )
    )
    psi_ref = np.asarray(basis_state(2 * int(problem.n_sites), hf_bits), dtype=complex)
    ansatz = HardcodedUCCSDAnsatz(
        dims=int(problem.n_sites),
        num_particles=num_particles,
        reps=int(reps),
        repr_mode="JW",
        indexing=str(problem.ordering),
        include_singles=True,
        include_doubles=True,
    )
    res = vqe_minimize(
        problem.h_poly,
        ansatz,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        method="SLSQP",
        maxiter=int(maxiter),
        bounds=(-math.pi, math.pi),
    )
    energy = float(res.energy)
    return {
        "energy": energy,
        "abs_delta_e": float(abs(energy - problem.exact_energy)),
        "ansatz_len": int(ansatz.num_parameters),
        "elapsed_s": float(time.perf_counter() - t_start),
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "nit": int(res.nit),
    }


def _prefix(sequence_ops: list[Any], depth: int) -> list[Any]:
    k = max(0, min(int(depth), len(sequence_ops)))
    return list(sequence_ops[:k])


def _classify(summary: dict[str, Any]) -> dict[str, Any]:
    baseline_gap = float(summary["l4"]["baseline"]["abs_delta_e"])

    frozen = summary["l4"].get("frozen_reopt", {})
    frozen_gaps = [float(v.get("abs_delta_e", float("inf"))) for v in frozen.values()]
    best_frozen_gap = min(frozen_gaps) if frozen_gaps else float("inf")
    closure = 0.0
    if baseline_gap > 0 and math.isfinite(best_frozen_gap):
        closure = max(0.0, min(1.0, (baseline_gap - best_frozen_gap) / baseline_gap))

    matched = summary["l4"].get("matched_depth_compare", {})
    current_gap = matched.get("current", {}).get("abs_delta_e")
    reference_gap = matched.get("reference", {}).get("abs_delta_e")
    selection_margin = None
    if current_gap is not None and reference_gap is not None:
        selection_margin = float(current_gap) - float(reference_gap)

    ladder = summary["l4"].get("budget_ladder", [])
    budget_gain = 0.0
    if len(ladder) >= 2:
        first = float(ladder[0]["abs_delta_e"])
        best = min(float(x["abs_delta_e"]) for x in ladder)
        if first > 0:
            budget_gain = max(0.0, min(1.0, (first - best) / first))

    if closure >= 0.70 and (selection_margin is None or abs(selection_margin) < 0.02) and budget_gain >= 0.25:
        label = "optimization_budget"
        reason = "Frozen-sequence re-optimization closes most error and quality improves strongly with budget."
    elif closure < 0.30 and selection_margin is not None and selection_margin >= 0.05 and budget_gain < 0.15:
        label = "selection"
        reason = "Frozen-sequence optimization does not recover much, while directional-FD selection outperforms current."
    else:
        label = "mixed"
        reason = "Signals indicate both selection quality and optimization budget likely contribute."

    return {
        "label": label,
        "reason": reason,
        "metrics": {
            "baseline_gap": baseline_gap,
            "best_frozen_gap": best_frozen_gap,
            "frozen_gap_closure": closure,
            "selection_margin_current_minus_reference": selection_margin,
            "budget_gain": budget_gain,
        },
        "criteria": {
            "selection_margin_threshold": 0.05,
            "frozen_gap_closure_threshold_high": 0.70,
            "frozen_gap_closure_threshold_low": 0.30,
            "budget_gain_threshold_high": 0.25,
            "budget_gain_threshold_low": 0.15,
        },
    }


def _flatten_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _push(group: str, name: str, payload: dict[str, Any]) -> None:
        rows.append(
            {
                "group": group,
                "name": name,
                "energy": payload.get("energy"),
                "abs_delta_e": payload.get("abs_delta_e"),
                "VarH": payload.get("VarH"),
                "depth": payload.get("depth", payload.get("ansatz_len")),
                "stop_reason": payload.get("stop_reason"),
                "t_total_s": payload.get("t_total_s", payload.get("elapsed_s")),
            }
        )

    l4 = summary["l4"]
    _push("l4", "baseline", l4["baseline"])
    _push("l4", "hardcoded_vqe", l4["hardcoded_vqe"])

    for regime, payload in l4.get("frozen_reopt", {}).items():
        _push("l4_frozen_reopt", regime, payload)

    for item in l4.get("budget_ladder", []):
        _push("l4_budget_ladder", f"max_time_s={item['max_time_s']}", item)

    for depth_key, payload in l4.get("prefix_compare", {}).items():
        _push("l4_prefix_current", depth_key, payload.get("current", {}))
        _push("l4_prefix_reference", depth_key, payload.get("reference", {}))

    l3 = summary.get("l3_control", {})
    for key, payload in l3.items():
        if isinstance(payload, dict) and "energy" in payload:
            _push("l3_control", key, payload)

    return rows


def _text_lines(summary: dict[str, Any]) -> list[str]:
    cls = summary["classification"]
    lines = [
        "ADAPT Root-Cause Diagnostic Report",
        f"Generated UTC: {summary['generated_utc']}",
        "",
        f"Classification: {cls['label']}",
        f"Reason: {cls['reason']}",
        "",
        "Key metrics:",
    ]
    for key, val in cls["metrics"].items():
        lines.append(f"- {key}: {val}")

    l4 = summary["l4"]
    lines += [
        "",
        "L4 baseline:",
        f"- energy={l4['baseline']['energy']:.10f}",
        f"- abs_delta_e={l4['baseline']['abs_delta_e']:.6e}",
        f"- depth={l4['baseline']['depth']}",
        f"- stop_reason={l4['baseline']['stop_reason']}",
        "",
        "L4 frozen-sequence best:",
    ]
    best_name = None
    best_gap = float("inf")
    for name, payload in l4.get("frozen_reopt", {}).items():
        gap = float(payload["abs_delta_e"])
        if gap < best_gap:
            best_gap = gap
            best_name = name
    if best_name is not None:
        p = l4["frozen_reopt"][best_name]
        lines.append(f"- {best_name}: abs_delta_e={p['abs_delta_e']:.6e}, energy={p['energy']:.10f}")

    lines += [
        "",
        "Assumptions:",
        "- Sector exact energy is the reference ground truth.",
        "- Current selection policy is the configured 'current' policy.",
        "- Reference selection policy is directional FD.",
    ]
    return lines


def _render_text_page(pdf: PdfPages, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.98
    step = 0.03
    for line in lines:
        ax.text(0.02, y, line, fontsize=10, va="top", family="monospace")
        y -= step
        if y <= 0.03:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(11.0, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.98
    pdf.savefig(fig)
    plt.close(fig)


def _plot_budget_ladder(summary: dict[str, Any]):
    data = summary["l4"].get("budget_ladder", [])
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if data:
        xs = [float(x["max_time_s"]) for x in data]
        ys = [float(x["abs_delta_e"]) for x in data]
        ax.plot(xs, ys, marker="o", linewidth=2.0)
    ax.set_title("L4 Budget Ladder (ADAPT-UCCSD)")
    ax.set_xlabel("max_time_s")
    ax.set_ylabel("abs error vs exact")
    ax.grid(alpha=0.3)
    return fig


def _plot_frozen_regimes(summary: dict[str, Any]):
    data = summary["l4"].get("frozen_reopt", {})
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if data:
        names = list(data.keys())
        ys = [float(data[k]["abs_delta_e"]) for k in names]
        ax.bar(names, ys)
    ax.set_title("L4 Frozen-Sequence Reoptimization")
    ax.set_ylabel("abs error vs exact")
    ax.grid(axis="y", alpha=0.3)
    return fig


def _plot_prefix_compare(summary: dict[str, Any]):
    data = summary["l4"].get("prefix_compare", {})
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if data:
        depths = sorted(int(k) for k in data.keys())
        y_cur = [float(data[str(d)]["current"]["abs_delta_e"]) for d in depths if "current" in data[str(d)]]
        y_ref = [float(data[str(d)]["reference"]["abs_delta_e"]) for d in depths if "reference" in data[str(d)]]
        x = np.arange(len(depths), dtype=float)
        width = 0.35
        if len(y_cur) == len(depths):
            ax.bar(x - width / 2, y_cur, width=width, label="current")
        if len(y_ref) == len(depths):
            ax.bar(x + width / 2, y_ref, width=width, label="reference")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in depths])
    ax.set_title("L4 Prefix Quality (Reoptimized)")
    ax.set_xlabel("prefix depth")
    ax.set_ylabel("abs error vs exact")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose ADAPT-UCCSD failures: selection vs optimizer/budget bottlenecks."
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS.keys()), default="smoke")
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--n-up", type=int, default=None)
    parser.add_argument("--n-down", type=int, default=None)
    parser.add_argument("--odd-policy", choices=["min_sz", "restrict", "dope_sz0"], default="min_sz")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument("--boundary", choices=["open", "periodic"], default="open")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--policy-current", choices=["probe_legacy", "directional_fd"], default="probe_legacy")
    parser.add_argument("--policy-reference", choices=["probe_legacy", "directional_fd"], default="directional_fd")
    parser.add_argument("--grouped-selection-fd-delta", type=float, default=1e-5)

    parser.add_argument("--adapt-max-depth", type=int, default=None)
    parser.add_argument("--adapt-inner-steps", type=int, default=None)
    parser.add_argument("--adapt-lbfgs-restarts", type=int, default=None)
    parser.add_argument("--baseline-max-time-s", type=float, default=None)
    parser.add_argument("--generous-inner-steps", type=int, default=None)
    parser.add_argument("--generous-lbfgs-restarts", type=int, default=None)
    parser.add_argument("--matched-depth", type=int, default=None)
    parser.add_argument("--prefix-depths", type=int, nargs="+", default=None)
    parser.add_argument("--budget-ladder", type=float, nargs="+", default=None)
    parser.add_argument("--l3-max-depth", type=int, default=None)
    parser.add_argument("--l3-max-time-s", type=float, default=None)
    parser.add_argument("--skip-l3-control", action="store_true")

    parser.add_argument("--eps-grad", type=float, default=1e-6)
    parser.add_argument("--eps-energy", type=float, default=1e-8)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/adapt_root_cause_L4"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    prof = PROFILE_DEFAULTS[str(args.profile)]
    if args.adapt_max_depth is None:
        args.adapt_max_depth = int(prof["adapt_max_depth"])
    if args.adapt_inner_steps is None:
        args.adapt_inner_steps = int(prof["adapt_inner_steps"])
    if args.adapt_lbfgs_restarts is None:
        args.adapt_lbfgs_restarts = int(prof["adapt_lbfgs_restarts"])
    if args.baseline_max_time_s is None:
        args.baseline_max_time_s = float(prof["baseline_max_time_s"])
    if args.generous_inner_steps is None:
        args.generous_inner_steps = int(prof["generous_inner_steps"])
    if args.generous_lbfgs_restarts is None:
        args.generous_lbfgs_restarts = int(prof["generous_lbfgs_restarts"])
    if args.matched_depth is None:
        args.matched_depth = int(prof["matched_depth"])
    if args.prefix_depths is None:
        args.prefix_depths = [int(x) for x in prof["prefix_depths"]]
    if args.budget_ladder is None:
        args.budget_ladder = [float(x) for x in prof["budget_ladder"]]
    if args.l3_max_depth is None:
        args.l3_max_depth = int(prof["l3_max_depth"])
    if args.l3_max_time_s is None:
        args.l3_max_time_s = float(prof["l3_max_time_s"])


def main() -> None:
    args = parse_args()
    _apply_profile_defaults(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_up, n_down = _resolve_sector(int(args.sites), args.n_up, args.n_down, args.odd_policy)
    problem_l4 = _build_problem(
        n_sites=int(args.sites),
        n_up=int(n_up),
        n_down=int(n_down),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
    )

    # L4 baseline with "current" policy.
    baseline_result, baseline = _run_adapt_uccsd(
        problem_l4,
        grouped_selection_gradient=str(args.policy_current),
        grouped_selection_fd_delta=float(args.grouped_selection_fd_delta),
        max_depth=int(args.adapt_max_depth),
        inner_steps=int(args.adapt_inner_steps),
        lbfgs_restarts=int(args.adapt_lbfgs_restarts),
        max_time_s=float(args.baseline_max_time_s),
        seed=int(args.seed),
        eps_grad=float(args.eps_grad),
        eps_energy=float(args.eps_energy),
        verbose=bool(args.verbose),
    )
    baseline_sequence = list(getattr(baseline_result, "operators", []))

    hardcoded = _run_hardcoded_vqe(problem_l4, reps=2, restarts=1, maxiter=1000, seed=int(args.seed))

    # Frozen-sequence reoptimization regimes.
    frozen_reopt: dict[str, dict[str, Any]] = {}
    regimes = {
        "A_current_budget": {
            "inner_optimizer": "lbfgs",
            "inner_steps": int(args.adapt_inner_steps),
            "lbfgs_restarts": int(args.adapt_lbfgs_restarts),
            "max_time_s": float(args.baseline_max_time_s),
        },
        "B_unbounded_lbfgs": {
            "inner_optimizer": "lbfgs",
            "inner_steps": int(args.generous_inner_steps),
            "lbfgs_restarts": int(args.generous_lbfgs_restarts),
            "max_time_s": None,
        },
        "C_multistart_slsqp": {
            "inner_optimizer": "slsqp",
            "inner_steps": int(args.generous_inner_steps),
            "lbfgs_restarts": max(2, int(args.generous_lbfgs_restarts)),
            "max_time_s": None,
        },
    }
    for name, cfg in regimes.items():
        _res, _sum = _run_fixed_sequence(
            problem_l4,
            baseline_sequence,
            inner_optimizer=str(cfg["inner_optimizer"]),
            inner_steps=int(cfg["inner_steps"]),
            lbfgs_restarts=int(cfg["lbfgs_restarts"]),
            max_time_s=cfg["max_time_s"],
            seed=int(args.seed),
            verbose=bool(args.verbose),
        )
        frozen_reopt[name] = _sum

    # Selection-policy runs with generous optimization to generate comparable prefixes.
    selection_depth = max([int(args.matched_depth)] + [int(x) for x in args.prefix_depths])
    seq_runs: dict[str, Any] = {}
    seq_summaries: dict[str, dict[str, Any]] = {}
    for label, policy in (("current", args.policy_current), ("reference", args.policy_reference)):
        r, s = _run_adapt_uccsd(
            problem_l4,
            grouped_selection_gradient=str(policy),
            grouped_selection_fd_delta=float(args.grouped_selection_fd_delta),
            max_depth=int(selection_depth),
            inner_steps=int(args.generous_inner_steps),
            lbfgs_restarts=int(args.generous_lbfgs_restarts),
            max_time_s=None,
            seed=int(args.seed),
            eps_grad=0.0,
            eps_energy=0.0,
            verbose=bool(args.verbose),
        )
        seq_runs[label] = r
        seq_summaries[label] = s

    # Matched-depth comparison (reopt prefixes under identical generous settings).
    matched_depth_compare: dict[str, dict[str, Any]] = {}
    for label in ("current", "reference"):
        ops = list(getattr(seq_runs[label], "operators", []))
        pref = _prefix(ops, int(args.matched_depth))
        _res, _sum = _run_fixed_sequence(
            problem_l4,
            pref,
            inner_optimizer="lbfgs",
            inner_steps=int(args.generous_inner_steps),
            lbfgs_restarts=int(args.generous_lbfgs_restarts),
            max_time_s=None,
            seed=int(args.seed),
            verbose=bool(args.verbose),
        )
        matched_depth_compare[label] = _sum

    # Prefix quality curve.
    prefix_compare: dict[str, dict[str, dict[str, Any]]] = {}
    for depth in [int(x) for x in args.prefix_depths]:
        key = str(depth)
        prefix_compare[key] = {}
        for label in ("current", "reference"):
            ops = list(getattr(seq_runs[label], "operators", []))
            pref = _prefix(ops, int(depth))
            _res, _sum = _run_fixed_sequence(
                problem_l4,
                pref,
                inner_optimizer="lbfgs",
                inner_steps=int(args.generous_inner_steps),
                lbfgs_restarts=int(args.generous_lbfgs_restarts),
                max_time_s=None,
                seed=int(args.seed),
                verbose=bool(args.verbose),
            )
            prefix_compare[key][label] = _sum

    # Budget ladder sensitivity.
    budget_ladder_rows: list[dict[str, Any]] = []
    for max_time in [float(x) for x in args.budget_ladder]:
        _res, _sum = _run_adapt_uccsd(
            problem_l4,
            grouped_selection_gradient=str(args.policy_current),
            grouped_selection_fd_delta=float(args.grouped_selection_fd_delta),
            max_depth=int(args.adapt_max_depth),
            inner_steps=int(args.adapt_inner_steps),
            lbfgs_restarts=int(args.adapt_lbfgs_restarts),
            max_time_s=float(max_time),
            seed=int(args.seed),
            eps_grad=float(args.eps_grad),
            eps_energy=float(args.eps_energy),
            verbose=bool(args.verbose),
        )
        _sum["max_time_s"] = float(max_time)
        budget_ladder_rows.append(_sum)

    l3_control: dict[str, Any]
    if bool(args.skip_l3_control):
        l3_control = {"skipped": True}
    else:
        # L3 control: condensed matrix to check scale behavior.
        l3_n_up, l3_n_down = _resolve_sector(3, None, None, args.odd_policy)
        problem_l3 = _build_problem(
            n_sites=3,
            n_up=int(l3_n_up),
            n_down=int(l3_n_down),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            ordering=str(args.ordering),
            boundary=str(args.boundary),
        )
        l3_baseline_res, l3_baseline = _run_adapt_uccsd(
            problem_l3,
            grouped_selection_gradient=str(args.policy_current),
            grouped_selection_fd_delta=float(args.grouped_selection_fd_delta),
            max_depth=int(args.l3_max_depth),
            inner_steps=max(1, int(args.adapt_inner_steps)),
            lbfgs_restarts=int(args.adapt_lbfgs_restarts),
            max_time_s=float(args.l3_max_time_s),
            seed=int(args.seed),
            eps_grad=float(args.eps_grad),
            eps_energy=float(args.eps_energy),
            verbose=bool(args.verbose),
        )
        l3_seq = list(getattr(l3_baseline_res, "operators", []))
        _l3_fixed_a_res, l3_fixed_a = _run_fixed_sequence(
            problem_l3,
            l3_seq,
            inner_optimizer="lbfgs",
            inner_steps=max(1, int(args.adapt_inner_steps)),
            lbfgs_restarts=int(args.adapt_lbfgs_restarts),
            max_time_s=float(args.l3_max_time_s),
            seed=int(args.seed),
            verbose=bool(args.verbose),
        )
        _l3_fixed_b_res, l3_fixed_b = _run_fixed_sequence(
            problem_l3,
            l3_seq,
            inner_optimizer="lbfgs",
            inner_steps=max(1, int(args.generous_inner_steps)),
            lbfgs_restarts=max(1, int(args.generous_lbfgs_restarts)),
            max_time_s=None,
            seed=int(args.seed),
            verbose=bool(args.verbose),
        )
        l3_control = {
            "baseline": l3_baseline,
            "fixed_sequence_current_budget": l3_fixed_a,
            "fixed_sequence_generous": l3_fixed_b,
        }

    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "profile": str(args.profile),
            "seed": int(args.seed),
            "sites": int(args.sites),
            "n_up": args.n_up,
            "n_down": args.n_down,
            "odd_policy": str(args.odd_policy),
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "policy_current": str(args.policy_current),
            "policy_reference": str(args.policy_reference),
            "grouped_selection_fd_delta": float(args.grouped_selection_fd_delta),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_inner_steps": int(args.adapt_inner_steps),
            "adapt_lbfgs_restarts": int(args.adapt_lbfgs_restarts),
            "baseline_max_time_s": float(args.baseline_max_time_s),
            "generous_inner_steps": int(args.generous_inner_steps),
            "generous_lbfgs_restarts": int(args.generous_lbfgs_restarts),
            "matched_depth": int(args.matched_depth),
            "prefix_depths": [int(x) for x in args.prefix_depths],
            "budget_ladder": [float(x) for x in args.budget_ladder],
            "l3_max_depth": int(args.l3_max_depth),
            "l3_max_time_s": float(args.l3_max_time_s),
            "skip_l3_control": bool(args.skip_l3_control),
        },
        "l4": {
            "exact_energy": float(problem_l4.exact_energy),
            "baseline": baseline,
            "baseline_selected_sequence": _serialize_sequence(baseline_sequence),
            "hardcoded_vqe": hardcoded,
            "frozen_reopt": frozen_reopt,
            "selection_sequence_runs": seq_summaries,
            "matched_depth_compare": matched_depth_compare,
            "prefix_compare": prefix_compare,
            "budget_ladder": budget_ladder_rows,
        },
        "l3_control": l3_control,
    }

    summary["classification"] = _classify(summary)

    json_path = out_dir / "diagnostic_summary.json"
    csv_path = out_dir / "diagnostic_table.csv"
    pdf_path = out_dir / "diagnostic_report.pdf"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = _flatten_rows(summary)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "name", "energy", "abs_delta_e", "VarH", "depth", "stop_reason", "t_total_s"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with PdfPages(pdf_path) as pdf:
        _render_text_page(pdf, _text_lines(summary))

        fig1 = _plot_frozen_regimes(summary)
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = _plot_budget_ladder(summary)
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = _plot_prefix_compare(summary)
        fig3.tight_layout()
        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote PDF:  {pdf_path}")


if __name__ == "__main__":
    main()
