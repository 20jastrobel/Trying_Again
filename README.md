<!-- README.md -->

# Hubbard (JW) + VQE (Hardcoded) Playground

This repository is a small, dependency-light codebase for building fermionic Hamiltonians (starting with the 1D spinful Fermi–Hubbard model), mapping them to qubit Pauli operators via Jordan–Wigner (JW), and then solving for ground-state energies using VQE.

The design goal is **operator-first**:
- define a clean algebra for Pauli “letters” → Pauli “words” → Pauli “polynomials”
- build fermionic operators and Hamiltonians in that algebra
- implement algorithms (VQE now, QPE/Suzuki–Trotter later) *on top* of the operator layer without heavy frameworks

A Qiskit implementation is included **only** as a validation reference.

---

## Core math conventions (important)

### Pauli “letters”
We use symbols:
- `e` = identity
- `x`, `y`, `z` = Pauli X/Y/Z

### Pauli-string ordering
A Pauli string `ps` is interpreted as acting on qubits in the order:

- **left-to-right = q_(n-1) ... q_0**
- equivalently: **qubit 0 is the rightmost character** in the string

Example: for `nq=4`, `"exzx"` means:
- q3: `e`
- q2: `x`
- q1: `z`
- q0: `x`

This matches how JW strings are constructed in `fermion_plus_operator` / `fermion_minus_operator`.

### Spin-orbital ordering before JW
Two supported orderings:

- **interleaved**: `(α0, β0, α1, β1, ..., α_{L-1}, β_{L-1})`
- **blocked**: `(α0, α1, ..., α_{L-1}, β0, β1, ..., β_{L-1})`

The mapping from (site, spin) → mode index depends on this choice.

---

## Repository layout (current)

### Operator algebra layer
- `pauli_letters_module.py`  
  Defines `PauliLetter` and the **Symbol Product Map** (Pauli multiplication table with phases).

- `qubitization_module.py`  
  Defines `PauliTerm`: a coefficient times a Pauli-word (tensor product of PauliLetters).

- `pauli_polynomial_class.py`  
  Defines `PauliPolynomial`: a sum of PauliTerms, with multiplication, reduction, etc.  
  Also defines JW-mapped ladder operators:
  - `fermion_plus_operator`  (creation operator)
  - `fermion_minus_operator` (annihilation operator)

- `pauli_words.py`  
  A standalone copy of `PauliTerm` with fallbacks (useful outside the full package).  
  In the packaged code, prefer **one** PauliTerm class consistently to avoid type-mismatch issues.

### Hubbard builder (literate / LaTeX-Python pairing)
- `hubbard_latex_python_pairs.py`  
  Builds the spinful Hubbard Hamiltonian as a `PauliPolynomial`:
  - kinetic term `H_t`
  - onsite term `H_U`
  - potential term `H_v`
  Includes a helper to render LaTeX and print the corresponding code.

- `hubbard_latex_python_pairs.ipynb`  
  Notebook version of the same “LaTeX above Python” style.

### Reference data / validation
- `export_hubbard_jw_reference.py`  
  Exports canonical `{PauliString: coefficient}` dictionaries as JSON for regression testing.

- `hubbard_jw_*.json`  
  Precomputed reference Hamiltonians for various L, boundary conditions, and indexing.

### Qiskit baseline for comparison
- `Hubbard_VQE_Qiskit.py`
- `Hubbard_VQE_Qiskit.ipynb`
- `Hubbard_VQE_Qiskit_hamiltonian_report*.json`

These build the same Fermi–Hubbard model using Qiskit Nature, map via `JordanWignerMapper`, and run VQE(UCCSD) + exact filtered diagonalization for comparison.

### Current eigensolver stub
- `quantum_eigensolver.py`  
  Currently a stub. This is where the hardcoded VQE pipeline should live (or where it should call into a dedicated VQE module).

---

## Typical workflows

### 1) Build a Hubbard Hamiltonian (JW PauliPolynomial)
In Python (inside the `pydephasing.quantum` package context):

```python
from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian

H = build_hubbard_hamiltonian(
    dims=2,      # L=2 dimer
    t=1.0,
    U=4.0,
    indexing="blocked",
    edges=[(0,1)],  # explicit edge-set
    pbc=True,
)
