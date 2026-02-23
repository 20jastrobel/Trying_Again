# ADAPT-VQE UCCSD Deep Research Brief (for Claude Code)

## 1) Snapshot (Current Conclusion)

This is **not** a parameter-budget insufficiency issue anymore.  
At this point, evidence strongly indicates a **core implementation/correctness issue** in grouped ADAPT behavior.

Key proof:
- Increasing ADAPT runtime/depth/restarts at `L=4` made results remain very poor (often worse), while canonical hardcoded VQE remains near the sector baseline.
- Even a heavy fixed-sequence comparator path (inside ADAPT core) still produced nonsense values under current budget/gradient handling.

## 2) Scope Constraints (Strict)

- Only modify files under: `Adapt/adapt_vqe_cse_only/core/`
- Do **not** edit canonical repo code (`src/` and pipeline repo are read-only references).
- Keep public core APIs compatible unless impossible; if impossible, document and justify.

## 3) Canonical Source of Truth

Primary canonical module:
- `src/quantum/vqe_latex_python_pairs.py`
  - `HardcodedUCCSDAnsatz`
  - `apply_exp_pauli_polynomial(sort_terms=True)`
  - `vqe_minimize` (SLSQP, bounded, no jac)

Reference HF helpers:
- `src/quantum/hartree_fock_reference_state.py`
- `src/quantum/vqe_latex_python_pairs.py::hartree_fock_bitstring`

## 4) Relevant ADAPT Core File

- `Adapt/adapt_vqe_cse_only/core/adapt_vqe_meta.py`

Important current points:
- Grouped pool modes include hardcoded ONZOT aliases.
- `grouped_term_order`, `selection_mode`, `selection_strategy`, validators, and `slsqp` path were added.
- Budget is still enforced via `BudgetExceeded` thrown inside energy callbacks (`_energy_obj`, `_energy_grad`), which can interrupt inner minimization.

## 5) Empirical Evidence

## 5.1 L=2/L=3 recap (already known)

- `L=2` can be exact in patched ADAPT grouped runs.
- `L=3` best grouped ADAPT remains far from exact and high `Var(H)`; many runs stop due `budget:max_time_s`.

## 5.2 L=4: new decisive evidence (sector exact comparison)

Model for all below:
- 1D Hubbard, open boundary
- `L=4`, `t=1.0`, `u=2.0`, blocked ordering
- Half-filling `(n_up,n_down)=(2,2)`
- **Use sector exact**, not full Hilbert exact

Sector exact target:
- `E_exact_sector = -2.8759428090050623`

Hardcoded references:
- `artifacts/hardcoded_pipeline_L4_u2_quick.json`:
  - `E_hardcoded = -2.7589387957551037`
  - gap to sector exact: `+0.11700401324995857`
- `artifacts/l4_exact_vs_hardcoded_vs_adapt_quick.json` (alternate hardcoded config):
  - `E_hardcoded = -2.7647746702964495`
  - gap: `+0.11116813870861275`

ADAPT grouped runs (all bad):
- `artifacts/l4_exact_vs_hardcoded_vs_adapt_quick.json`
  - `E_adapt = -0.5268251381932941`
  - gap to exact: `+2.349117670811768`
  - depth `4`, `Var(H)~1.96588`, stop `budget:max_time_s`
- `artifacts/l4_adapt_vs_exact_vs_hardcoded_quickcompare.json`
  - `E_adapt = 1.155858862540`
  - gap to exact: `+4.031801671545`
  - depth `5`, stop `budget:max_time_s`
- `artifacts/l4_adapt_heavy_600s_maxgrad.json` (10 min ADAPT):
  - `E_adapt = 1.2837219068350916`
  - gap to exact: `+4.159664715840154`
  - depth `6`, `Var(H)~1.75734`, stop `budget:max_time_s`
- `artifacts/l4_heavy_sector_sweep.json`:
  - `fixed_sequence_heavy`: `E=0.2682541331790904`, gap `+3.144196942184153`, depth `26`, stop `budget:max_time_s`
  - `adapt_maxgrad_heavy`: `E=1.2993114032820139`, gap `+4.175254212287076`, depth `4`, stop `budget:max_time_s`
  - `adapt_lookahead_heavy`: `E=5.0690359909566496e-8`, gap `+2.875942859695422`, depth `3`, stop `budget:max_time_s`

Interpretation:
- Heavier settings do not recover.
- This is incompatible with a pure “not enough depth/time” explanation.

## 5.3 Important exact-energy trap

`hardcoded_pipeline_L4_u2_quick.json` also contains full-space exact:
- `ground_state.exact_energy = -3.0695353593424146`

Do **not** compare half-filled ADAPT/UCCSD runs against this full-space value.
Use sector exact `-2.8759428090050623`.

## 6) High-Probability Root Causes (Ranked)

1. **Grouped outer selection gradient is still not physically consistent enough for ADAPT ranking.**
   - Current grouped selection still relies on probe-based assembly (`_compute_grouped_pool_gradients`).
   - Existing diagnostic runs showed large mismatch vs directional finite-difference (order ~`0.5` relative error in quick checks).

2. **Budget enforcement is breaking inner optimization semantics.**
   - `BudgetExceeded` is raised inside objective/gradient callbacks.
   - This can interrupt SciPy minimization mid-run and return partially optimized/noisy points.
   - This directly pollutes both ADAPT and fixed-sequence comparator outcomes.

3. **Fixed-sequence comparator is not currently a clean canonical-equivalence benchmark under budget pressure.**
   - Even if unitary semantics are close, repeated time aborts during optimization invalidate apples-to-apples conclusions.

4. **ADAPT path dependence remains severe once selection gradients are noisy.**
   - Bad selected operators at early depth poison later trajectory.

## 7) Implementation Tasks for Claude Code (Concrete)

Implement in `Adapt/adapt_vqe_cse_only/core/adapt_vqe_meta.py` (or minimally within `core/` only):

1. Replace grouped outer selection gradient with exact directional finite difference.
   - For each candidate grouped spec:
     - Build trial circuit with `ops + [candidate]`.
     - Evaluate `E(theta_aug + delta*e_last)` and `E(theta_aug - delta*e_last)` with current `theta` fixed and appended parameter around 0.
     - Use centered diff `(E+ - E-) / (2*delta)` for ranking.
   - Keep old probe method only as optional debug mode (not default for grouped pools).

2. Fix budget semantics so inner optimizers are not torn down mid-objective call.
   - Remove `BudgetExceeded` throws from objective callbacks passed into SciPy.
   - Enforce global budget at safe boundaries:
     - before outer iteration
     - before inner solve
     - after inner solve
   - If early-stop inside minimize is required, use controlled callback/exception with best-so-far state capture, not arbitrary objective abort.

3. Harden canonical fixed-sequence mode as benchmark contract.
   - For hardcoded pool:
     - `grouped_term_order = canonical_sorted`
     - `grouped_generator_scale = 1.0`
     - `grouped_trotter_order=1`, `grouped_trotter_reps=1`
   - Run SLSQP bounded (`[-pi,pi]`) without jac.
   - Add deterministic restarts and report best run stats.
   - Ensure no budget-induced partial optimizer exits unless explicitly requested and clearly labeled.

4. Keep/extend validators and make them gate correctness.
   - Semantics validator:
     - hardcoded grouped circuit vs canonical `HardcodedUCCSDAnsatz.prepare_state(sort_terms=True)`
     - require fidelity `>= 1 - 1e-10` for L=3 random trials.
   - Outer-gradient validator:
     - compare ADAPT selection gradient vs directional FD
     - require max relative error `<= 1e-3` in deterministic checks.

5. Improve diagnostics needed for forensic debugging.
   - Per outer iteration log:
     - top-k candidate gradients
     - selected operator index/name
     - pre/post inner-opt energy
     - energy drop achieved
     - cumulative wall time
   - Log explicit reason for stopping inner optimization.

## 8) Acceptance Criteria (Must Pass)

1. **Canonical semantics equivalence**
   - L=3 hardcoded semantics validator passes with min fidelity `>= 1 - 1e-10`.

2. **Grouped outer gradient correctness**
   - Validator max relative error `<= 1e-3` at HF and at one random theta point.

3. **Fixed-sequence canonical match sanity**
   - L=3 fixed-sequence mode (same optimizer contract as canonical) should track canonical hardcoded VQE energy closely (target difference `<= 1e-6` on deterministic seed/budget settings, or explain residual numerical mismatch).

4. **L=4 no longer pathological**
   - Under heavy but finite budget, ADAPT should stop producing obviously non-physical high energies with `Var(H)~O(1-2)` at tiny depth.
   - It does not need to be exact immediately, but should move toward hardcoded baseline rather than diverge.

## 9) Repro Artifacts to Read First

- `artifacts/l4_heavy_sector_sweep.json`
- `artifacts/l4_adapt_heavy_600s_maxgrad.json`
- `artifacts/l4_exact_vs_hardcoded_vs_adapt_quick.json`
- `artifacts/l4_adapt_vs_exact_vs_hardcoded_quickcompare.json`
- `artifacts/hardcoded_pipeline_L4_u2_quick.json`
- Prior L=3 context:
  - `artifacts/l3_onzot_prior_best_cfg_after_scale_fix.json`
  - `artifacts/l3_onzot_heavy_bounded_sweep.json`
  - `artifacts/adapt_cse_heavy_L2_L3_groupfix.json`

## 10) Known Runner Issue (Non-Core)

- `Adapt/adapt_vqe_cse_only/run_adapt_cse.py` has stale import expectations around removed cost-counter symbols.
- This is runner-layer, not core algorithm logic.
- Core debugging should proceed via direct `run_meta_adapt_vqe` calls.

## 11) Explicit Ask to Claude Code

Please implement the above fixes in ADAPT core (no canonical edits), then run a deterministic validation matrix:
- L=3 semantics validator
- L=3 outer-gradient validator
- L=3 fixed-sequence canonical comparator
- L=4 exact-vs-hardcoded-vs-ADAPT sector comparison

Return:
- patch summary
- exact commands run
- resulting energies/gaps/Var(H)
- whether the issue is resolved or narrowed.
