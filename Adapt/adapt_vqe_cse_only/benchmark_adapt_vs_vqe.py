#!/usr/bin/env python3
"""Benchmark ADAPT-CSE and ADAPT-UCCSD against hardcoded/Qiskit UCCSD VQE."""

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


METHOD_ORDER = ["exact", "adapt_cse", "adapt_uccsd", "hardcoded_vqe", "qiskit_vqe"]
METHOD_LABELS = {
    "exact": "Exact (Sector)",
    "adapt_cse": "ADAPT-CSE",
    "adapt_uccsd": "ADAPT-UCCSD",
    "hardcoded_vqe": "Hardcoded VQE (UCCSD)",
    "qiskit_vqe": "Qiskit VQE (UCCSD)",
}


@dataclass
class MethodResult:
    method: str
    energy: float
    exact_energy: float
    delta_e: float
    abs_delta_e: float
    elapsed_s: float
    ansatz_len: int
    details: dict[str, Any]


def _selected_method_order(*, include_qiskit_vqe: bool) -> list[str]:
    methods = ["exact", "adapt_cse", "adapt_uccsd", "hardcoded_vqe"]
    if bool(include_qiskit_vqe):
        methods.append("qiskit_vqe")
    return methods


def _resolve_sector(n_sites: int, n_up: int | None, n_down: int | None, odd_policy: str) -> tuple[int, int]:
    if n_up is None and n_down is None:
        return half_filling_sector(int(n_sites), odd_policy=str(odd_policy))
    if n_up is None or n_down is None:
        raise ValueError("Set both --n-up and --n-down together, or set neither.")
    return int(n_up), int(n_down)


def _site_potential_default(n_sites: int, dv: float) -> list[float] | None:
    # Preserve run_adapt_cse.py defaults: only L=2 gets a staggered +/-dv/2 potential.
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


def _pack_method(
    *,
    method: str,
    energy: float,
    exact_energy: float,
    elapsed_s: float,
    ansatz_len: int,
    details: dict[str, Any] | None = None,
) -> MethodResult:
    return MethodResult(
        method=str(method),
        energy=float(energy),
        exact_energy=float(exact_energy),
        delta_e=float(energy - exact_energy),
        abs_delta_e=float(abs(energy - exact_energy)),
        elapsed_s=float(elapsed_s),
        ansatz_len=int(ansatz_len),
        details=dict(details or {}),
    )


def _run_adapt(
    *,
    method_name: str,
    pool_mode: str,
    qubit_op,
    reference,
    ferm_op,
    mapper,
    n_sites: int,
    n_up: int,
    n_down: int,
    exact_energy: float,
    args: argparse.Namespace,
) -> MethodResult:
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
        "selection_mode": "adapt",
        "selection_strategy": "max_gradient",
        "inner_optimizer": "lbfgs",
        "max_depth": int(args.adapt_max_depth),
        "inner_steps": int(args.adapt_inner_steps),
        "lbfgs_maxiter": int(args.adapt_inner_steps),
        "lbfgs_restarts": int(args.adapt_lbfgs_restarts),
        "eps_grad": float(args.adapt_eps_grad),
        "eps_energy": float(args.adapt_eps_energy),
        "max_time_s": float(args.adapt_max_time_s) if args.adapt_max_time_s is not None else None,
        "theta_bound": "pi",
        "seed": int(args.adapt_seed),
        "allow_repeats": bool(args.adapt_allow_repeats),
        "compute_var_h": True,
        "verbose": bool(args.adapt_verbose),
    }
    if str(pool_mode) in {"hardcoded_uccsd_onzots", "uccsd_onzots"}:
        kwargs.update(
            {
                "hardcoded_indexing": str(args.ordering),
                "grouped_generator_scale": 1.0,
                "grouped_term_order": "canonical_sorted",
            }
        )

    result = run_meta_adapt_vqe(qubit_op, reference, estimator, **kwargs)
    elapsed = time.perf_counter() - start
    outer = result.diagnostics.get("outer", [])
    final_var_h = None
    if outer and isinstance(outer[-1], dict):
        final_var_h = outer[-1].get("VarH")

    details = {
        "stop_reason": result.diagnostics.get("stop_reason"),
        "pool_mode": str(pool_mode),
        "outer_iters": int(len(outer)),
        "final_var_h": final_var_h,
        "pool_size": result.diagnostics.get("pool_size", [None])[0],
    }
    return _pack_method(
        method=method_name,
        energy=float(result.energy),
        exact_energy=float(exact_energy),
        elapsed_s=elapsed,
        ansatz_len=len(result.operators),
        details=details,
    )


def _run_hardcoded_vqe(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    h_poly,
    ordering: str,
    exact_energy: float,
    args: argparse.Namespace,
) -> MethodResult:
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
        reps=int(args.vqe_reps),
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    result = vqe_minimize(
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(args.vqe_restarts),
        seed=int(args.vqe_seed),
        method="SLSQP",
        maxiter=int(args.vqe_maxiter),
        bounds=(-math.pi, math.pi),
    )

    elapsed = time.perf_counter() - start
    details = {
        "success": bool(result.success),
        "message": str(result.message),
        "best_restart": int(result.best_restart),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "num_parameters": int(ansatz.num_parameters),
    }
    return _pack_method(
        method="hardcoded_vqe",
        energy=float(result.energy),
        exact_energy=float(exact_energy),
        elapsed_s=elapsed,
        ansatz_len=int(ansatz.num_parameters),
        details=details,
    )


def _run_qiskit_vqe(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    qubit_op,
    mapper,
    exact_energy: float,
    args: argparse.Namespace,
) -> MethodResult:
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
        reps=int(args.vqe_reps),
    )

    rng = np.random.default_rng(int(args.vqe_seed))
    best_energy = float("inf")
    best_restart = -1
    best_result = None

    for restart in range(max(1, int(args.vqe_restarts))):
        x0 = 0.3 * rng.normal(size=ansatz.num_parameters)
        optimizer = SLSQP(maxiter=max(1, int(args.vqe_maxiter)))
        solver = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=x0,
        )
        result = solver.compute_minimum_eigenvalue(qubit_op)
        energy = float(np.real(result.eigenvalue))
        if energy < best_energy:
            best_energy = energy
            best_restart = restart
            best_result = result

    elapsed = time.perf_counter() - start
    nit = None
    nfev = None
    if best_result is not None and getattr(best_result, "optimizer_result", None) is not None:
        opt_res = best_result.optimizer_result
        nit = getattr(opt_res, "nit", None)
        nfev = getattr(opt_res, "nfev", None)

    details = {
        "success": True,
        "method": "qiskit_vqe_uccsd",
        "num_parameters": int(ansatz.num_parameters),
        "best_restart": int(best_restart),
        "nfev": int(nfev) if nfev is not None else None,
        "nit": int(nit) if nit is not None else None,
        "effective_maxiter": int(args.vqe_maxiter),
    }
    return _pack_method(
        method="qiskit_vqe",
        energy=float(best_energy),
        exact_energy=float(exact_energy),
        elapsed_s=elapsed,
        ansatz_len=int(ansatz.num_parameters),
        details=details,
    )


def _rows_for_csv(payload_results: list[dict[str, Any]], method_order: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload_results:
        l_value = int(entry["L"])
        methods = entry["methods"]
        for method in method_order:
            block = methods[method]
            details = block.get("details") or {}
            rows.append(
                {
                    "L": l_value,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "energy": float(block["energy"]),
                    "exact_energy": float(block["exact_energy"]),
                    "delta_e": float(block["delta_e"]),
                    "abs_delta_e": float(block["abs_delta_e"]),
                    "elapsed_s": float(block["elapsed_s"]),
                    "ansatz_len": int(block["ansatz_len"]),
                    "stop_reason": details.get("stop_reason"),
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


def _plot_energy(results: list[dict[str, Any]], method_order: list[str]) -> plt.Figure:
    sites = [int(r["L"]) for r in results]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for method in method_order:
        ys = [float(r["methods"][method]["energy"]) for r in results]
        ax.plot(sites, ys, marker="o", linewidth=2.0, label=METHOD_LABELS[method])
    ax.set_xlabel("L (Number of Sites)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Comparison by System Size")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def _plot_abs_error(results: list[dict[str, Any]], method_order: list[str]) -> plt.Figure:
    methods = [m for m in method_order if m != "exact"]
    sites = [int(r["L"]) for r in results]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for method in methods:
        ys = [max(float(r["methods"][method]["abs_delta_e"]), 1e-16) for r in results]
        ax.plot(sites, ys, marker="o", linewidth=2.0, label=METHOD_LABELS[method])
    ax.set_xlabel("L (Number of Sites)")
    ax.set_ylabel("Absolute Energy Error")
    ax.set_yscale("log")
    ax.set_title("Absolute Error vs Exact Sector Energy")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    return fig


def _plot_runtime(results: list[dict[str, Any]], method_order: list[str]) -> plt.Figure:
    methods = [m for m in method_order if m != "exact"]
    sites = [int(r["L"]) for r in results]
    x = np.arange(len(sites), dtype=float)
    width = 0.8 / max(1, len(methods))
    center = 0.5 * (len(methods) - 1)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    for idx, method in enumerate(methods):
        ys = [float(r["methods"][method]["elapsed_s"]) for r in results]
        ax.bar(x + (idx - center) * width, ys, width=width, label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={l}" for l in sites])
    ax.set_xlabel("System Size")
    ax.set_ylabel("Wall-Clock Runtime (s)")
    ax.set_title("Runtime by Method")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best")
    return fig


def _summary_lines(results: list[dict[str, Any]], args: argparse.Namespace, method_order: list[str]) -> list[str]:
    lines = [
        "ADAPT/VQE Benchmark Summary",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "Settings:",
        (
            f"  sites={args.sites} t={args.t} u={args.u} dv={args.dv} "
            f"boundary={args.boundary} ordering={args.ordering} odd_policy={args.odd_policy}"
        ),
        (
            f"  ADAPT(inner_optimizer=lbfgs, inner_steps={args.adapt_inner_steps}, "
            f"max_depth={args.adapt_max_depth}, max_time_s={args.adapt_max_time_s}, "
            f"allow_repeats={args.adapt_allow_repeats})"
        ),
        (
            f"  VQE(reps={args.vqe_reps}, restarts={args.vqe_restarts}, "
            f"maxiter={args.vqe_maxiter}, seed={args.vqe_seed})"
        ),
        f"  include_qiskit_vqe={bool(args.include_qiskit_vqe)}",
        "",
    ]
    for entry in results:
        lines.append(f"L={entry['L']}  (n_up={entry['n_up']}, n_down={entry['n_down']})")
        for method in method_order:
            block = entry["methods"][method]
            lines.append(
                "  "
                + f"{METHOD_LABELS[method]:24s} "
                + f"E={float(block['energy']): .12f} "
                + f"|dE|={float(block['abs_delta_e']):.3e} "
                + f"t={float(block['elapsed_s']):.2f}s"
            )
        lines.append("")
    return lines


def _render_text_page(pdf: PdfPages, lines: list[str], *, fontsize: int = 10, step: float = 0.03) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.98
    for line in lines:
        ax.text(
            0.03,
            y,
            line,
            transform=ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=fontsize,
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ADAPT-CSE/ADAPT-UCCSD vs hardcoded and Qiskit UCCSD VQE."
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

    parser.add_argument("--adapt-max-depth", type=int, default=12)
    parser.add_argument("--adapt-inner-steps", type=int, default=250)
    parser.add_argument("--adapt-lbfgs-restarts", type=int, default=1)
    parser.add_argument("--adapt-max-time-s", type=float, default=300.0)
    parser.add_argument("--adapt-eps-grad", type=float, default=1e-6)
    parser.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    parser.add_argument("--adapt-seed", type=int, default=7)
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
    parser.add_argument("--adapt-verbose", action="store_true")

    parser.add_argument("--vqe-reps", type=int, default=1)
    parser.add_argument("--vqe-restarts", type=int, default=1)
    parser.add_argument("--vqe-maxiter", type=int, default=250)
    parser.add_argument("--vqe-seed", type=int, default=7)
    parser.set_defaults(include_qiskit_vqe=True)
    parser.add_argument(
        "--include-qiskit-vqe",
        dest="include_qiskit_vqe",
        action="store_true",
        help="Include Qiskit VQE baseline (default: enabled).",
    )
    parser.add_argument(
        "--skip-qiskit-vqe",
        dest="include_qiskit_vqe",
        action="store_false",
        help="Skip Qiskit VQE baseline.",
    )

    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/adapt_benchmark_l2_l3"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sites = [int(x) for x in args.sites]
    method_order = _selected_method_order(include_qiskit_vqe=bool(args.include_qiskit_vqe))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        methods: dict[str, MethodResult] = {
            "exact": _pack_method(
                method="exact",
                energy=float(exact),
                exact_energy=float(exact),
                elapsed_s=0.0,
                ansatz_len=0,
                details={"note": "sector-filtered exact energy"},
            )
        }

        methods["adapt_cse"] = _run_adapt(
            method_name="adapt_cse",
            pool_mode="cse_density_ops",
            qubit_op=qubit_op,
            reference=reference,
            ferm_op=ferm_op,
            mapper=mapper,
            n_sites=int(n_sites),
            n_up=int(n_up),
            n_down=int(n_down),
            exact_energy=float(exact),
            args=args,
        )
        methods["adapt_uccsd"] = _run_adapt(
            method_name="adapt_uccsd",
            pool_mode="hardcoded_uccsd_onzots",
            qubit_op=qubit_op,
            reference=reference,
            ferm_op=ferm_op,
            mapper=mapper,
            n_sites=int(n_sites),
            n_up=int(n_up),
            n_down=int(n_down),
            exact_energy=float(exact),
            args=args,
        )
        methods["hardcoded_vqe"] = _run_hardcoded_vqe(
            n_sites=int(n_sites),
            n_up=int(n_up),
            n_down=int(n_down),
            h_poly=h_poly,
            ordering=str(args.ordering),
            exact_energy=float(exact),
            args=args,
        )
        if bool(args.include_qiskit_vqe):
            methods["qiskit_vqe"] = _run_qiskit_vqe(
                n_sites=int(n_sites),
                n_up=int(n_up),
                n_down=int(n_down),
                qubit_op=qubit_op,
                mapper=mapper,
                exact_energy=float(exact),
                args=args,
            )

        all_results.append(
            {
                "L": int(n_sites),
                "n_up": int(n_up),
                "n_down": int(n_down),
                "hamiltonian": {
                    "t": float(args.t),
                    "u": float(args.u),
                    "dv_cli": float(args.dv),
                    "boundary": str(args.boundary),
                    "ordering": str(args.ordering),
                    "site_potential": v_spec,
                },
                "methods": {k: vars(v) for k, v in methods.items()},
            }
        )

    all_results.sort(key=lambda item: int(item["L"]))
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
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
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_inner_steps": int(args.adapt_inner_steps),
            "adapt_lbfgs_restarts": int(args.adapt_lbfgs_restarts),
            "adapt_max_time_s": float(args.adapt_max_time_s) if args.adapt_max_time_s is not None else None,
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_seed": int(args.adapt_seed),
            "adapt_allow_repeats": bool(args.adapt_allow_repeats),
            "vqe_reps": int(args.vqe_reps),
            "vqe_restarts": int(args.vqe_restarts),
            "vqe_maxiter": int(args.vqe_maxiter),
            "vqe_seed": int(args.vqe_seed),
            "include_qiskit_vqe": bool(args.include_qiskit_vqe),
        },
        "results": all_results,
    }

    json_path = out_dir / "benchmark_results.json"
    csv_path = out_dir / "benchmark_results.csv"
    pdf_path = out_dir / "benchmark_report.pdf"
    energy_png = out_dir / "energy_vs_exact.png"
    abs_err_png = out_dir / "abs_error_vs_exact.png"
    runtime_png = out_dir / "runtime_by_method.png"

    rows = _rows_for_csv(all_results, method_order)
    _save_csv(csv_path, rows)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig_energy = _plot_energy(all_results, method_order)
    fig_abs = _plot_abs_error(all_results, method_order)
    fig_runtime = _plot_runtime(all_results, method_order)
    fig_energy.tight_layout()
    fig_abs.tight_layout()
    fig_runtime.tight_layout()

    fig_energy.savefig(energy_png, dpi=180)
    fig_abs.savefig(abs_err_png, dpi=180)
    fig_runtime.savefig(runtime_png, dpi=180)

    with PdfPages(pdf_path) as pdf:
        _render_text_page(pdf, _summary_lines(all_results, args, method_order), fontsize=9, step=0.028)
        pdf.savefig(fig_energy)
        pdf.savefig(fig_abs)
        pdf.savefig(fig_runtime)

    plt.close(fig_energy)
    plt.close(fig_abs)
    plt.close(fig_runtime)

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote PDF:  {pdf_path}")
    print(f"Wrote PNG:  {energy_png}")
    print(f"Wrote PNG:  {abs_err_png}")
    print(f"Wrote PNG:  {runtime_png}")


if __name__ == "__main__":
    main()
