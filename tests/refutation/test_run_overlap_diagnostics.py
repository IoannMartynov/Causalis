import numpy as np
import pandas as pd

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def test_run_overlap_diagnostics_returns_summary_and_meta():
    rng = np.random.default_rng(42)
    n = 300
    m = np.clip(rng.beta(2.0, 2.0, size=n), 1e-6, 1.0 - 1e-6)
    d = rng.integers(0, 2, size=n)
    data, estimate = make_overlap_data_and_estimate(
        m_hat=m,
        d=d,
        normalize_ipw=True,
        overlap_threshold=0.10,
    )

    report = run_overlap_diagnostics(data, estimate)

    assert isinstance(report, dict)
    assert report["n"] == n
    assert "flags" in report and isinstance(report["flags"], dict)
    assert "summary" in report and isinstance(report["summary"], pd.DataFrame)
    assert "ate_tails" in report and isinstance(report["ate_tails"], dict)
    assert "att_weights" in report and isinstance(report["att_weights"], dict)
    assert report["meta"]["use_hajek"] is True
    assert report["meta"]["n_bins"] == 10
    assert {"tails_w1_q99/med", "tails_w0_q99/med", "ATT_identity_relerr"}.issubset(set(report["summary"]["metric"]))


def test_run_overlap_diagnostics_falls_back_to_data_when_diag_d_missing():
    rng = np.random.default_rng(7)
    n = 250
    m = np.clip(rng.uniform(0.01, 0.99, size=n), 1e-6, 1.0 - 1e-6)
    d = rng.integers(0, 2, size=n)
    data, estimate = make_overlap_data_and_estimate(
        m_hat=m,
        d=d,
        include_d_in_diag=False,
    )

    report = run_overlap_diagnostics(data, estimate)

    assert report["n"] == n
    assert report["n_treated"] == int(np.sum(d))
    assert isinstance(report.get("summary"), pd.DataFrame)
