import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_data(n: int = 300, seed: int = 123) -> CausalData:
    df = generate_rct(n=n, k=2, random_state=seed, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def _make_estimate(data: CausalData):
    return IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=777,
    ).fit().estimate(score="ATE")


def test_score_diagnostics_is_deterministic_for_fixed_estimate():
    data = _make_data(n=240, seed=777)
    estimate = _make_estimate(data)

    out1 = run_score_diagnostics(data, estimate, overlap_threshold=0.01, n_basis_funcs=3, return_summary=True)
    out2 = run_score_diagnostics(data, estimate, overlap_threshold=0.01, n_basis_funcs=3, return_summary=True)

    assert out1["params"] == out2["params"]
    assert np.isclose(out1["influence_diagnostics"]["se_plugin"], out2["influence_diagnostics"]["se_plugin"])
    assert np.isclose(out1["influence_diagnostics"]["p99_over_med"], out2["influence_diagnostics"]["p99_over_med"])
    assert np.isclose(out1["influence_diagnostics"]["kurtosis"], out2["influence_diagnostics"]["kurtosis"])
