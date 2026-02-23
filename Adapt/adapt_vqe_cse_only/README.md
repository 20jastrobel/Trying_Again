# ADAPT-VQE CSE Only

This folder is the minimal workflow for:
- ADAPT-VQE with the CSE operator pool
- Benchmarking ADAPT-CSE against regular UCCSD-VQE

## Install

```bash
python -m pip install -U pip
python -m pip install numpy scipy qiskit qiskit-aer qiskit-nature qiskit-algorithms
```

## Run ADAPT-CSE only

```bash
python adapt_vqe_cse_only/run_adapt_cse.py --sites 2 --out runs/adapt_cse_result.json
```

Notes:
- ADAPT operator reuse is enabled by default in runners (`allow_repeats=True`).
- To disable reuse, pass `--adapt-no-repeats`.

## Benchmark ADAPT-CSE vs UCCSD VQE

```bash
python adapt_vqe_cse_only/benchmark_adapt_vs_vqe.py --sites 2 3 4 --out-dir runs/compare_adapt_vs_vqe_only
```

Notes:
- ADAPT reuse is on by default; use `--adapt-no-repeats` to force no reuse.

Outputs:
- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_report.pdf`
- plots: `energy_vs_exact.png`, `abs_error_vs_exact.png`, `runtime_by_method.png`

## Confidence Campaign (Deep/Bounded Sweeps)

```bash
python adapt_vqe_cse_only/run_adapt_confidence_campaign.py \
  --sites 2 3 \
  --sweep-profile deep \
  --per-method-max-time-s 1200 \
  --adapt-trial-max-time-s 300 \
  --adapt-fixed-trial-max-time-s 600 \
  --out-dir runs/adapt_confidence_campaign_L2_L3
```

Notes:
- ADAPT adaptive modes use operator reuse by default; disable with `--adapt-no-repeats`.

Outputs:
- `campaign_results.json` (all trials + diagnostics + gates)
- `campaign_best.csv` (best-per-method summary)
- `campaign_report.pdf`
- plots:
  - `energy_vs_exact_best.png`
  - `abs_error_vs_exact_best_log.png`
  - `runtime_best.png`
  - `gap_closure_vs_method.png`
  - `adapt_uccsd_convergence_L{L}.png`
  - `adapt_cse_convergence_L{L}.png`

## Root-Cause Diagnostics (Selection vs Optimization/Budget)

Run the structured ADAPT-UCCSD diagnostic matrix:

```bash
python adapt_vqe_cse_only/diagnose_adapt_uccsd_root_cause.py \
  --profile practical \
  --sites 4 \
  --policy-current probe_legacy \
  --policy-reference directional_fd \
  --out-dir artifacts/adapt_root_cause_L4
```

Quick smoke check:

```bash
python adapt_vqe_cse_only/diagnose_adapt_uccsd_root_cause.py \
  --profile smoke \
  --skip-l3-control \
  --out-dir artifacts/adapt_root_cause_smoke
```

Outputs:
- `diagnostic_summary.json` (full matrix + automatic classification)
- `diagnostic_table.csv` (flat table)
- `diagnostic_report.pdf` (summary + plots)
