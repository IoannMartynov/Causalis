import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_data(n: int = 1000, k: int = 4, seed: int = 123) -> CausalData:
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
        random_state=29,
    ).fit()
    return model.estimate(score="ATE", alpha=0.10)


def test_score_diagnostics_returns_flags_and_summary_with_flags_column():
    data = _make_data(seed=31)
    estimate = _make_estimate(data)

    report = run_score_diagnostics(data, estimate, return_summary=True)

    assert "flags" in report
    assert "overall_flag" in report
    assert "thresholds" in report
    assert report["overall_flag"] in {"GREEN", "YELLOW", "RED", "NA"}

    assert "summary" in report
    assert isinstance(report["summary"], pd.DataFrame)
    assert "flag" in report["summary"].columns


def test_score_diagnostics_flag_keys_are_present():
    data = _make_data(seed=37)
    estimate = _make_estimate(data)

    report = run_score_diagnostics(data, estimate, return_summary=True)
    flags = report["flags"]

    assert "psi_tail_ratio" in flags
    assert "psi_kurtosis" in flags
    assert "ortho_max_|t|_g1" in flags
    assert "ortho_max_|t|_g0" in flags
    assert "ortho_max_|t|_m" in flags
    assert "oos_moment" in flags
