import numpy as np

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def test_overlap_metrics_have_valid_ranges():
    rng = np.random.default_rng(0)
    n = 500
    m = np.clip(rng.beta(2.0, 2.0, size=n), 1e-6, 1.0 - 1e-6)
    d = rng.binomial(1, m)
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)

    report = run_overlap_diagnostics(data, estimate)

    edge = report["edge_mass"]
    assert 0.0 <= edge["share_below_001"] <= 1.0
    assert 0.0 <= edge["share_above_001"] <= 1.0
    assert 0.0 <= edge["share_below_002"] <= 1.0
    assert 0.0 <= edge["share_above_002"] <= 1.0

    if np.isfinite(report["ks"]):
        assert 0.0 <= report["ks"] <= 1.0
    if np.isfinite(report["auc"]):
        assert 0.0 <= report["auc"] <= 1.0

    ess = report["ate_ess"]
    assert set(["ess_w1", "ess_w0", "ess_ratio_w1", "ess_ratio_w0"]).issubset(ess.keys())


def test_nearly_separable_propensity_triggers_strong_flags():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.standard_normal(n)
    d = (x > 0.0).astype(int)
    m = 1.0 / (1.0 + np.exp(-5.0 * x))
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d, overlap_threshold=0.01)

    report = run_overlap_diagnostics(data, estimate)

    red_any = any(
        report["flags"].get(key) == "RED"
        for key in ["auc", "ks", "edge_mass_001", "edge_mass_002"]
    )
    assert red_any


def test_extreme_treated_weight_triggers_tails_red_flag():
    n_treated = 200
    n_control = 200
    d = np.concatenate([np.ones(n_treated, dtype=int), np.zeros(n_control, dtype=int)])
    m_treated = np.full(n_treated, 0.5, dtype=float)
    m_treated[-1] = 1e-6
    m_control = np.full(n_control, 0.5, dtype=float)
    m = np.concatenate([m_treated, m_control])

    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)
    report = run_overlap_diagnostics(data, estimate)

    assert report["flags"]["tails_w1"] == "RED"
    assert report["flags"]["tails_w0"] in {"GREEN", "YELLOW", "RED", "NA"}

    tails_w1 = report["ate_tails"]["w1"]
    med = float(tails_w1["median"])
    assert med > 0.0
    assert float(tails_w1["q999"]) / med > 100.0
    assert float(tails_w1["max"]) / med > 100.0


def test_att_identity_relerr_and_flag_are_reported():
    n_treated = 200
    n_control = 200
    d = np.concatenate([np.ones(n_treated, dtype=int), np.zeros(n_control, dtype=int)])
    m_treated = np.full(n_treated, 0.5, dtype=float)
    m_control = np.full(n_control, 0.9, dtype=float)
    m = np.concatenate([m_treated, m_control])

    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)
    report = run_overlap_diagnostics(data, estimate)

    assert "att_weights" in report
    assert report["att_weights"]["rhs_sum"] == float(n_treated)
    assert report["att_weights"]["lhs_sum"] > report["att_weights"]["rhs_sum"]
    assert report["att_weights"]["rel_err"] > 0.10
    assert report["flags"]["att_identity"] == "RED"

    row = report["summary"].loc[report["summary"]["metric"] == "ATT_identity_relerr"]
    assert not row.empty
    assert float(row["value"].iloc[0]) > 0.10
    assert row["flag"].iloc[0] == "RED"
