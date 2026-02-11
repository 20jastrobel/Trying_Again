#!/usr/bin/env python3
import argparse
import os
import time
import traceback

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from pydephasing.quantum.ansatz import build_ansatz
from pydephasing.quantum.hamiltonian.exact import exact_ground_energy
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.vqe.run_store import JsonRunStore, SqliteRunStore


def run_vqe(
    qubit_op: SparsePauliOp,
    mapper,
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    maxiter: int,
    reps: int,
    seed: int,
    store=None,
    run_id: str | None = None,
    exact_energy: float | None = None,
) -> tuple[float, float]:
    estimator = StatevectorEstimator()
    ansatz = build_ansatz(
        "uccsd",
        qubit_op.num_qubits,
        reps,
        mapper,
        n_sites=n_sites,
        num_particles=(n_up, n_down),
    )
    optimizer = COBYLA(maxiter=maxiter)
    initial_point = np.zeros(ansatz.num_parameters, dtype=float)

    t0 = time.perf_counter()
    t_last = t0

    def _callback(eval_count, params, energy, _meta):
        nonlocal t_last
        if store is None or run_id is None:
            return
        now = time.perf_counter()
        row = {
            "energy": float(energy),
            "t_iter_s": float(now - t_last),
            "t_cum_s": float(now - t0),
            "ansatz": "uccsd",
            "n_params": int(ansatz.num_parameters),
        }
        if exact_energy is not None:
            row["delta_e"] = float(energy) - float(exact_energy)
        t_last = now
        store.log_step(str(run_id), int(eval_count), row)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=_callback,
    )

    start = time.perf_counter()
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    elapsed = time.perf_counter() - start
    vqe_energy = float(np.real(result.eigenvalue))
    return vqe_energy, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local UCCSD-VQE vs exact ground-state energy for 2-site Hubbard (4 qubits)."
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-up", type=int, default=1)
    parser.add_argument("--n-down", type=int, default=1)
    parser.add_argument(
        "--store",
        type=str,
        choices=["sqlite", "json"],
        default=os.environ.get("RUN_STORE", "sqlite"),
        help="Run storage backend (env: RUN_STORE).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.environ.get("RUNS_DB_PATH", "data/runs.db"),
        help="SQLite DB path when --store sqlite (env: RUNS_DB_PATH).",
    )
    args = parser.parse_args()

    store_kind = str(args.store)
    if store_kind == "sqlite":
        store = SqliteRunStore(args.db)
    elif store_kind == "json":
        store = JsonRunStore(base_dir="runs")
    else:
        raise ValueError(f"Unknown store kind: {store_kind}")

    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=args.t,
        u=args.u,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-args.dv / 2, args.dv / 2],
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    exact = exact_ground_energy(qubit_op)
    run_config = {
        "run_id": f"vqe_local_{int(time.time())}",
        "entrypoint": "scripts/run_vqe_local.py",
        "seed": int(args.seed),
        "sites": 2,
        "n_up": int(args.n_up),
        "n_down": int(args.n_down),
        "system": {"sites": 2, "n_up": int(args.n_up), "n_down": int(args.n_down)},
        "ansatz": {"kind": "uccsd", "reps": int(args.reps)},
        "optimizer": {"name": "COBYLA", "maxiter": int(args.maxiter)},
        "cli": vars(args),
        "exact_energy": float(exact),
    }
    run_id = store.start_run(run_config)
    status = "completed"
    summary: dict[str, object] = {}
    try:
        vqe_energy, elapsed = run_vqe(
            qubit_op,
            mapper,
            n_sites=2,
            n_up=args.n_up,
            n_down=args.n_down,
            maxiter=args.maxiter,
            reps=args.reps,
            seed=args.seed,
            store=store,
            run_id=run_id,
            exact_energy=exact,
        )
        summary = {
            "energy": float(vqe_energy),
            "exact_energy": float(exact),
            "abs_delta_e": float(abs(vqe_energy - exact)),
            "elapsed_s": float(elapsed),
        }
    except Exception as exc:
        status = "error"
        summary = {"error": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        try:
            store.finish_run(run_id, status=status, summary_metrics=summary)
        finally:
            if hasattr(store, "close"):
                store.close()

    diff = vqe_energy - exact

    print("Local VQE (no IBM runtime)")
    print(f"t={args.t}, U={args.u}, dv={args.dv}")
    print(f"Exact ground energy: {exact:.8f}")
    print(f"VQE energy:          {vqe_energy:.8f}")
    print(f"Abs diff:            {abs(diff):.8e}")
    print(f"Elapsed:             {elapsed:.2f}s")


if __name__ == "__main__":
    main()
