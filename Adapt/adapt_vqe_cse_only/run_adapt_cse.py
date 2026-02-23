#!/usr/bin/env python3
"""Run ADAPT-VQE with the CSE operator pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Qiskit >=1.0
    from qiskit.primitives import StatevectorEstimator

    def _make_estimator():
        return StatevectorEstimator()

except ImportError:  # Qiskit 0.x fallback
    from qiskit.primitives import Estimator

    def _make_estimator():
        return Estimator()

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


def _resolve_sector(n_sites: int, n_up: int | None, n_down: int | None, odd_policy: str) -> tuple[int, int]:
    if n_up is None and n_down is None:
        return half_filling_sector(int(n_sites), odd_policy=str(odd_policy))
    if n_up is None or n_down is None:
        raise ValueError("Set both --n-up and --n-down together, or set neither.")
    return int(n_up), int(n_down)


def main() -> None:
    ap = argparse.ArgumentParser(description="ADAPT-VQE with CSE pool on 1D Hubbard.")
    ap.add_argument("--sites", type=int, default=2)
    ap.add_argument("--n-up", type=int, default=None)
    ap.add_argument("--n-down", type=int, default=None)
    ap.add_argument("--odd-policy", type=str, choices=["min_sz", "restrict", "dope_sz0"], default="min_sz")
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--u", type=float, default=4.0)
    ap.add_argument("--dv", type=float, default=0.5)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--inner-steps", type=int, default=25)
    ap.add_argument("--eps-grad", type=float, default=1e-4)
    ap.add_argument("--eps-energy", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.set_defaults(adapt_allow_repeats=True)
    ap.add_argument(
        "--adapt-allow-repeats",
        dest="adapt_allow_repeats",
        action="store_true",
        help="Allow ADAPT to reselect previously chosen operators (default: enabled).",
    )
    ap.add_argument(
        "--adapt-no-repeats",
        dest="adapt_allow_repeats",
        action="store_false",
        help="Disable ADAPT operator reuse.",
    )
    ap.add_argument("--budget-k", type=float, default=2000.0)
    ap.add_argument("--no-budget", action="store_true")
    ap.add_argument("--max-pauli-terms", type=int, default=None)
    ap.add_argument("--max-circuits", type=int, default=None)
    ap.add_argument("--max-time-s", type=float, default=None)
    ap.add_argument("--out", type=str, default="runs/adapt_cse_result.json")
    args = ap.parse_args()

    n_sites = int(args.sites)
    n_up, n_down = _resolve_sector(n_sites, args.n_up, args.n_down, args.odd_policy)

    ferm_op = build_fermionic_hubbard(
        n_sites=n_sites,
        t=float(args.t),
        u=float(args.u),
        edges=default_1d_chain_edges(n_sites, periodic=False),
        v=[-float(args.dv) / 2.0, float(args.dv) / 2.0] if n_sites == 2 else None,
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)

    exact = exact_ground_energy_sector(
        qubit_op,
        n_sites,
        int(n_up) + int(n_down),
        0.5 * (int(n_up) - int(n_down)),
    )

    occupations = jw_reference_occupations_from_particles(n_sites, n_up, n_down)
    reference = build_reference_state(qubit_op.num_qubits, occupations)

    if args.no_budget:
        max_pauli_terms = None
        max_circuits = None
        max_time_s = None
    else:
        max_pauli_terms = (
            int(args.max_pauli_terms)
            if args.max_pauli_terms is not None
            else int(float(args.budget_k) * (n_sites**2))
        )
        max_circuits = int(args.max_circuits) if args.max_circuits is not None else None
        max_time_s = float(args.max_time_s) if args.max_time_s is not None else None

    estimator = _make_estimator()
    result = run_meta_adapt_vqe(
        qubit_op,
        reference,
        estimator,
        pool_mode="cse_density_ops",
        ferm_op=ferm_op,
        mapper=mapper,
        n_sites=n_sites,
        n_up=n_up,
        n_down=n_down,
        enforce_sector=True,
        inner_optimizer="lbfgs",
        max_depth=int(args.max_depth),
        inner_steps=int(args.inner_steps),
        lbfgs_maxiter=int(args.inner_steps),
        lbfgs_restarts=1,
        eps_grad=float(args.eps_grad),
        eps_energy=float(args.eps_energy),
        allow_repeats=bool(args.adapt_allow_repeats),
        max_pauli_terms_measured=max_pauli_terms,
        max_circuits_executed=max_circuits,
        max_time_s=max_time_s,
        seed=int(args.seed),
        verbose=False,
    )

    operators = []
    for op in result.operators:
        if isinstance(op, dict) and "name" in op:
            operators.append(str(op["name"]))
        else:
            operators.append(str(op))

    payload = {
        "method": "adapt_cse_hybrid",
        "sites": n_sites,
        "n_up": int(n_up),
        "n_down": int(n_down),
        "ham_params": {"t": float(args.t), "u": float(args.u), "dv": float(args.dv)},
        "energy": float(result.energy),
        "exact_sector": float(exact),
        "delta_e": float(result.energy - exact),
        "abs_delta_e": float(abs(result.energy - exact)),
        "ansatz_len": int(len(result.operators)),
        "operators": operators,
        "stop_reason": result.diagnostics.get("stop_reason"),
        "budget": {
            "enabled": bool(not args.no_budget),
            "budget_k": None if args.no_budget else float(args.budget_k),
            "max_pauli_terms_measured": max_pauli_terms,
            "max_circuits_executed": max_circuits,
            "max_time_s": max_time_s,
        },
        "adapt_allow_repeats": bool(args.adapt_allow_repeats),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Exact sector energy: {exact:.10f}")
    print(f"ADAPT-CSE energy:    {float(result.energy):.10f}")
    print(f"Abs error:           {abs(float(result.energy) - exact):.6e}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
