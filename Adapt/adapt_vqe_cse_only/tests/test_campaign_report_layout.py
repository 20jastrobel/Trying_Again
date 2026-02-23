from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_adapt_confidence_campaign as report_mod


def _best(
    *,
    energy: float,
    exact: float,
    elapsed_s: float,
    stop_reason: str | None,
    outer_energies: list[float],
) -> dict:
    return {
        "ok": True,
        "energy": float(energy),
        "exact_energy": float(exact),
        "delta_e": float(energy - exact),
        "abs_delta_e": float(abs(energy - exact)),
        "elapsed_s": float(elapsed_s),
        "ansatz_len": int(len(outer_energies)),
        "trial_params": {},
        "details": {
            "stop_reason": stop_reason,
            "trace": {
                "outer_energies": list(outer_energies),
                "outer_iters": len(outer_energies),
                "monotonic_violations": 0,
                "max_abs_N_err": 0.0,
                "max_abs_Sz_err": 0.0,
            },
        },
    }


def _fake_results() -> list[dict]:
    exact_l2 = -0.8360571181551808
    exact_l3 = -1.2360679774997887
    return [
        {
            "L": 2,
            "n_up": 1,
            "n_down": 1,
            "exact_energy": exact_l2,
            "hf_energy": 4.5,
            "methods": {
                "hardcoded_vqe": {"best": _best(energy=exact_l2 + 1e-9, exact=exact_l2, elapsed_s=0.2, stop_reason=None, outer_energies=[exact_l2 + 1e-9])},
                "qiskit_vqe": {"best": _best(energy=exact_l2 + 5e-10, exact=exact_l2, elapsed_s=0.4, stop_reason=None, outer_energies=[exact_l2 + 5e-10])},
                "adapt_uccsd_fixed": {"best": _best(energy=exact_l2 + 1e-2, exact=exact_l2, elapsed_s=1.2, stop_reason="budget:max_time_s", outer_energies=[exact_l2 + 1e-1])},
                "adapt_uccsd_adapt": {"best": _best(energy=exact_l2 + 1e-3, exact=exact_l2, elapsed_s=2.0, stop_reason=None, outer_energies=[exact_l2 + 1e-1, exact_l2 + 1e-2, exact_l2 + 1e-3])},
                "adapt_cse_adapt": {"best": _best(energy=exact_l2 + 2e-3, exact=exact_l2, elapsed_s=1.8, stop_reason=None, outer_energies=[exact_l2 + 2e-2])},
            },
            "gates": {
                "hardcoded_vqe": {"pass": True, "threshold": 1e-8, "abs_delta_e": 1e-9, "reason": "ok"},
                "qiskit_vqe": {"pass": True, "threshold": 1e-8, "abs_delta_e": 5e-10, "reason": "ok"},
                "adapt_uccsd_fixed": {"pass": False, "threshold": 1e-8, "abs_delta_e": 1e-2, "diagnostics_ok": True, "semantics_ok": True, "reason": "failed_fixed_uccsd_gate"},
                "adapt_uccsd_adapt": {"pass": False, "threshold": 1e-7, "abs_delta_e": 1e-3, "diagnostics_ok": True, "reason": "failed_adaptive_uccsd_gate"},
                "adapt_cse_adapt": {"pass": True, "gap_closure": 0.95, "gap_closure_min": 0.9, "diagnostics_ok": True, "reason": "ok"},
            },
        },
        {
            "L": 3,
            "n_up": 2,
            "n_down": 1,
            "exact_energy": exact_l3,
            "hf_energy": 4.0,
            "methods": {
                "hardcoded_vqe": {"best": _best(energy=exact_l3 + 1e-7, exact=exact_l3, elapsed_s=6.0, stop_reason=None, outer_energies=[exact_l3 + 1e-7])},
                "qiskit_vqe": {"best": _best(energy=exact_l3 + 1e-7, exact=exact_l3, elapsed_s=8.0, stop_reason=None, outer_energies=[exact_l3 + 1e-7])},
                "adapt_uccsd_fixed": {"best": _best(energy=3.0, exact=exact_l3, elapsed_s=5.0, stop_reason="budget:max_time_s", outer_energies=[3.0])},
                "adapt_uccsd_adapt": {"best": _best(energy=2.8, exact=exact_l3, elapsed_s=4.0, stop_reason="budget:max_time_s", outer_energies=[3.1, 2.9, 2.8])},
                "adapt_cse_adapt": {"best": _best(energy=2.9, exact=exact_l3, elapsed_s=3.5, stop_reason="budget:max_time_s", outer_energies=[2.9])},
            },
            "gates": {
                "hardcoded_vqe": {"pass": True, "threshold": 1e-6, "abs_delta_e": 1e-7, "reason": "ok"},
                "qiskit_vqe": {"pass": True, "threshold": 1e-6, "abs_delta_e": 1e-7, "reason": "ok"},
                "adapt_uccsd_fixed": {"pass": False, "threshold": 1e-6, "abs_delta_e": 4.2, "diagnostics_ok": True, "semantics_ok": False, "reason": "failed_fixed_uccsd_gate"},
                "adapt_uccsd_adapt": {"pass": False, "threshold": 1e-5, "abs_delta_e": 4.0, "diagnostics_ok": True, "reason": "failed_adaptive_uccsd_gate"},
                "adapt_cse_adapt": {"pass": False, "gap_closure": 0.2, "gap_closure_min": 0.9, "diagnostics_ok": True, "reason": "failed_cse_relative_improvement_gate"},
            },
        },
    ]


def test_campaign_report_pdf_layout(tmp_path: Path) -> None:
    results = _fake_results()
    gate = report_mod.GateTarget()
    args = SimpleNamespace(
        sites=[2, 3],
        boundary="open",
        ordering="blocked",
        seed=7,
        per_method_max_time_s=1200.0,
        adapt_trial_max_time_s=300.0,
        adapt_fixed_trial_max_time_s=600.0,
    )
    generated_utc = "2026-02-21T00:00:00+00:00"

    report_mod._apply_report_style()
    warnings = report_mod._collect_report_warnings(results, args)
    pages = [
        report_mod._render_dashboard_page(results, args, gate, generated_utc, warnings),
        report_mod._render_detailed_results_page(results),
        report_mod._plot_abs_error_best(results, gate),
        report_mod._plot_runtime_best(results, cap_s=float(args.per_method_max_time_s)),
        report_mod._plot_gap_closure(results, gate),
        report_mod._plot_energy_best(results),
    ]
    for entry in results:
        pages.append(report_mod._plot_adapt_uccsd_convergence(entry))
        pages.append(report_mod._plot_adapt_cse_convergence(entry))
    pages.append(report_mod._render_appendix_page(results, warnings))

    pdf_path = tmp_path / "campaign_report.pdf"
    with PdfPages(pdf_path) as pdf:
        total = len(pages)
        for idx, fig in enumerate(pages, start=1):
            report_mod._apply_page_chrome(fig, generated_utc=generated_utc, page_no=idx, page_total=total)
            pdf.savefig(fig)
    for fig in pages:
        plt.close(fig)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0

    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) >= 6
    sizes = []
    for page in reader.pages:
        box = page.mediabox
        sizes.append((round(float(box.width), 2), round(float(box.height), 2)))
    assert len(set(sizes)) == 1

    page0 = reader.pages[0].extract_text() or ""
    assert "ADAPT Confidence Campaign" in page0
    assert "PASS/FAIL Matrix" in page0
    assert "Summary Dashboard" in page0
