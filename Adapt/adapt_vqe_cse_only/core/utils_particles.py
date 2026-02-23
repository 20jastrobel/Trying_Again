"""Particle-count and JW-reference occupation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.quantum.hartree_fock_reference_state import hartree_fock_occupied_qubits
from src.quantum.vqe_latex_python_pairs import half_filled_num_particles


def half_filling_sector(n_sites: int, *, odd_policy: str = "min_sz") -> tuple[int, int]:
    policy = str(odd_policy).strip().lower()
    if policy not in {"min_sz", "restrict", "dope_sz0"}:
        raise ValueError(f"Unknown odd_policy: {odd_policy}")

    n_up, n_down = half_filled_num_particles(int(n_sites))
    return int(n_up), int(n_down)


def jw_reference_occupations_from_particles(
    n_sites: int,
    n_up: int,
    n_down: int,
    *,
    indexing: str = "blocked",
) -> list[int]:
    return list(
        hartree_fock_occupied_qubits(
            int(n_sites),
            (int(n_up), int(n_down)),
            indexing=str(indexing),
        )
    )
