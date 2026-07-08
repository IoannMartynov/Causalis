import numpy as np

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def test_report_contains_expected_top_level_keys():
    rng = np.random.default_rng(123)
    n = 400
    m = np.clip(rng.beta(2.0, 2.0, size=n), 1e-6, 1.0 - 1e-6)
    d = rng.binomial(1, m)
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)

    report = run_overlap_diagnostics(data, estimate)

    expected = {"edge_mass", "ks", "auc", "ate_ess", "ate_tails", "att_weights", "clipping", "calibration", "flags", "meta", "summary"}
    assert expected.issubset(set(report.keys()))


def test_hajek_autodetected_from_diagnostic_data():
    rng = np.random.default_rng(9)
    m = np.clip(rng.uniform(0.05, 0.95, size=240), 1e-6, 1.0 - 1e-6)
    d = rng.integers(0, 2, size=240)
    data, estimate = make_overlap_data_and_estimate(
        m_hat=m,
        d=d,
        normalize_ipw=True,
        overlap_threshold=0.1,
    )

    report = run_overlap_diagnostics(data, estimate)

    assert report["meta"]["use_hajek"] is True
    assert report["flags"]["clip_m"] in {"GREEN", "YELLOW", "RED"}


def test_clip_flag_is_na_without_overlap_threshold():
    rng = np.random.default_rng(77)
    m = np.clip(rng.uniform(0.1, 0.9, size=200), 1e-6, 1.0 - 1e-6)
    d = rng.integers(0, 2, size=200)
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d, overlap_threshold=None)

    report = run_overlap_diagnostics(data, estimate)

    assert report["flags"]["clip_m"] == "NA"
    assert np.isnan(report["clipping"]["m_clip_lower"])
    assert np.isnan(report["clipping"]["m_clip_upper"])


def test_auc_flip_guard_flags_anti_predictive_scores():
    rng = np.random.default_rng(11)
    n = 200
    d = np.array([0, 1] * (n // 2))
    m = np.clip(1.0 - d + rng.normal(0.0, 0.01, size=n), 1e-6, 1.0 - 1e-6)
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)

    report = run_overlap_diagnostics(data, estimate)

    assert report["flags"]["auc"] == "RED"
    assert report["flags"].get("auc_flip_suspected") == "YELLOW"
