"""Exact energy evaluation utilities."""

from __future__ import annotations

import numpy as np

try:
    from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
except Exception:  # pragma: no cover
    from qiskit_algorithms import NumPyMinimumEigensolver


def exact_ground_energy(qubit_op) -> float:
    try:
        solver = NumPyMinimumEigensolver()
        result = solver.compute_minimum_eigenvalue(qubit_op)
        return float(np.real(result.eigenvalue))
    except Exception:
        mat = qubit_op.to_matrix()
        evals = np.linalg.eigvalsh(mat)
        return float(np.min(np.real(evals)))
