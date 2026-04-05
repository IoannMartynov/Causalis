import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_estimate(seed: int = 123):
    df = generate_rct(n=800, k=4, random_state=seed, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]
    data = CausalData(df=df, treatment="d", outcome="y", confounders=confs)
    estimate = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=400),
        n_folds=3,
        normalize_ipw=True,
        random_state=seed,
    ).fit().estimate(score="ATE")
    return data, estimate


def test_se_plugin_metric_present_and_finite():
    data, estimate = _make_estimate(seed=13)
    report = run_score_diagnostics(data, estimate, return_summary=True)

    assert np.isfinite(float(report["influence_diagnostics"]["se_plugin"]))
    se_row = report["summary"].loc[report["summary"]["metric"] == "se_plugin"]
    assert not se_row.empty
    assert se_row["flag"].iloc[0] == "NA"


def test_summary_contains_flag_column_and_expected_metrics():
    data, estimate = _make_estimate(seed=29)
    report = run_score_diagnostics(data, estimate, return_summary=True)

    summary = report["summary"]
    assert "flag" in summary.columns
    assert {"se_plugin", "psi_p99_over_med", "psi_kurtosis"}.issubset(set(summary["metric"]))
