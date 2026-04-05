import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_data(n: int = 1200, k: int = 4, seed: int = 123) -> CausalData:
    df = generate_rct(n=n, k=k, random_state=seed, outcome_type="normal")
    confs = [column for column in df.columns if column.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def _make_estimate(data: CausalData):
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=400),
        n_folds=3,
        normalize_ipw=True,
        trimming_threshold=1e-3,
        random_state=19,
    ).fit()
    return model.estimate(score="ATE", alpha=0.10)


def test_run_score_diagnostics_with_causal_estimate_and_data():
    data = _make_data(seed=7)
    estimate = _make_estimate(data)

    report = run_score_diagnostics(data, estimate, return_summary=True)
    assert "influence_diagnostics" in report
    assert "orthogonality_derivatives" in report
    assert "oos_moment_test" in report
    assert "summary" in report
    assert isinstance(report["summary"], pd.DataFrame)
    assert report["meta"]["n"] == int(data.get_df().shape[0])
    assert report["params"]["score"] == "ATE"


def test_run_score_diagnostics_falls_back_to_causal_data_when_y_d_missing():
    data = _make_data(seed=11)
    estimate = _make_estimate(data)

    diag_without_yd = estimate.diagnostic_data.model_copy(update={"y": None, "d": None})
    estimate_without_yd = estimate.model_copy(update={"diagnostic_data": diag_without_yd})

    report = run_score_diagnostics(data, estimate_without_yd, return_summary=True)
    assert "oos_moment_test" in report
    assert "summary" in report
    assert isinstance(report["summary"], pd.DataFrame)
    assert report["meta"]["n"] == int(data.get_df().shape[0])
