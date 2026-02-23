"""Ansatz builders for the focused ADAPT-vs-VQE benchmark."""

from __future__ import annotations

def build_uccsd_ansatz(
    *,
    n_sites: int,
    num_particles: tuple[int, int],
    reps: int,
    qubit_mapper,
):
    try:
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    except ImportError:  # pragma: no cover
        from qiskit_nature.circuit.library import HartreeFock, UCCSD

    particles = (int(num_particles[0]), int(num_particles[1]))

    try:
        initial_state = HartreeFock(
            num_spatial_orbitals=int(n_sites),
            num_particles=particles,
            qubit_mapper=qubit_mapper,
        )
    except TypeError:
        initial_state = HartreeFock(int(n_sites), particles, qubit_mapper)

    try:
        return UCCSD(
            num_spatial_orbitals=int(n_sites),
            num_particles=particles,
            qubit_mapper=qubit_mapper,
            reps=int(reps),
            initial_state=initial_state,
        )
    except TypeError:
        return UCCSD(
            int(n_sites),
            particles,
            qubit_mapper,
            reps=int(reps),
            initial_state=initial_state,
        )

ANSATZ_BUILDERS = {
    "uccsd": build_uccsd_ansatz,
}


def build_ansatz(
    kind: str,
    num_qubits: int,
    reps: int,
    mapper,
    *,
    n_sites: int | None = None,
    num_particles: int | tuple[int, int] | None = None,
):
    if kind not in ANSATZ_BUILDERS:
        raise ValueError(f"Unknown ansatz: {kind}")

    if kind == "uccsd":
        if n_sites is None:
            if num_qubits % 2 != 0:
                raise ValueError("UCCSD requires an even number of qubits for spinful mapping.")
            n_sites = num_qubits // 2

        if num_particles is None:
            if n_sites % 2 != 0:
                raise ValueError(
                    "Default UCCSD num_particles assumes even n_sites. "
                    "Pass num_particles explicitly for odd n_sites."
                )
            num_particles = (n_sites // 2, n_sites // 2)

        return ANSATZ_BUILDERS[kind](
            n_sites=n_sites,
            num_particles=num_particles,
            reps=reps,
            qubit_mapper=mapper,
        )

    raise ValueError(f"Unsupported ansatz kind: {kind}")
