import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
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
        overlap_threshold=1e-3,
        random_state=19,
    ).fit()
    return model.estimate(score="ATE", alpha=0.10)


def test_overlap_single_api_with_causal_estimate():
    data = _make_data(seed=17)
    estimate = _make_estimate(data)

    report = run_overlap_diagnostics(data, estimate)
    assert "summary" in report
    assert "ks" in report
    assert "auc" in report


def test_overlap_single_api_with_causal_estimate_and_causal_data_fallback():
    data = _make_data(seed=29)
    estimate = _make_estimate(data)

    # Fallback path: when `d` is missing in diagnostic_data, it is read from CausalData.
    diag_without_d = estimate.diagnostic_data.model_copy(update={"d": None})
    estimate_without_d = estimate.model_copy(update={"diagnostic_data": diag_without_d})

    report = run_overlap_diagnostics(data, estimate_without_d)
    assert report["n"] == int(data.get_df().shape[0])
    assert "summary" in report


def test_score_single_api_with_causal_estimate():
    data = _make_data(seed=41)
    estimate = _make_estimate(data)

    report = run_score_diagnostics(data, estimate)
    assert "summary" in report
    assert "orthogonality_derivatives" in report


def test_score_single_api_with_causal_estimate_and_causal_data_fallback():
    data = _make_data(seed=53)
    estimate = _make_estimate(data)

    # Fallback path: y and d missing in diagnostic_data, take them from provided CausalData.
    diag_without_yd = estimate.diagnostic_data.model_copy(update={"y": None, "d": None})
    estimate_without_yd = estimate.model_copy(update={"diagnostic_data": diag_without_yd})

    report = run_score_diagnostics(data, estimate_without_yd)
    assert "summary" in report
    assert isinstance(report["summary"], pd.DataFrame)
    assert report["meta"]["n"] == int(data.get_df().shape[0])
