import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_data(n: int = 800, k: int = 4, seed: int = 202) -> CausalData:
    df = generate_rct(n=n, k=k, random_state=seed, outcome_type="normal")
    confs = [column for column in df.columns if column.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def test_run_score_diagnostics_has_flags_by_default():
    data = _make_data()
    estimate = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=400),
        n_folds=3,
        normalize_ipw=True,
        trimming_threshold=1e-3,
        random_state=17,
    ).fit().estimate(score="ATE")

    report = run_score_diagnostics(data, estimate, return_summary=True)
    assert "flags" in report
    assert "overall_flag" in report
    assert report["overall_flag"] in {"GREEN", "YELLOW", "RED", "NA"}
    assert "summary" in report and isinstance(report["summary"], pd.DataFrame)
    assert "flag" in report["summary"].columns
