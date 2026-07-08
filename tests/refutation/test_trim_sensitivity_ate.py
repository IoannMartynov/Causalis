import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_data(n: int = 300, seed: int = 321) -> CausalData:
    df = generate_rct(n=n, k=3, random_state=seed, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def test_score_diagnostics_overlap_threshold_is_respected_in_params():
    data = _make_data(n=240, seed=777)
    estimate = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=400),
        n_folds=3,
        random_state=777,
    ).fit().estimate(score="ATE")

    report_lo = run_score_diagnostics(data, estimate, overlap_threshold=1e-6)
    report_hi = run_score_diagnostics(data, estimate, overlap_threshold=0.10)

    assert report_lo["params"]["overlap_threshold"] == 1e-6
    assert report_hi["params"]["overlap_threshold"] == 0.10
    assert np.isfinite(float(report_lo["influence_diagnostics"]["se_plugin"]))
    assert np.isfinite(float(report_hi["influence_diagnostics"]["se_plugin"]))
