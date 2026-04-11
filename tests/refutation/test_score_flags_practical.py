from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_estimate(seed: int):
    df = generate_rct(n=1200, k=5, random_state=seed, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]
    data = CausalData(df=df, treatment="d", outcome="y", confounders=confs)
    estimate = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=400),
        n_folds=3,
        random_state=seed,
    ).fit().estimate(score="ATE")
    return data, estimate


def test_score_flags_and_thresholds_present():
    data, estimate = _make_estimate(seed=3)
    out = run_score_diagnostics(data, estimate, return_summary=True)

    assert set(
        [
            "psi_tail_ratio",
            "psi_kurtosis",
            "ortho_max_|t|_g1",
            "ortho_max_|t|_g0",
            "ortho_max_|t|_m",
            "oos_moment",
        ]
    ).issubset(set(out["flags"]))
    assert set(
        ["tail_ratio_warn", "tail_ratio_strong", "kurt_warn", "kurt_strong", "t_warn", "t_strong"]
    ).issubset(set(out["thresholds"]))
    assert out["overall_flag"] in {"GREEN", "YELLOW", "RED", "NA"}


def test_summary_flags_align_with_report_flags():
    data, estimate = _make_estimate(seed=9)
    out = run_score_diagnostics(data, estimate, return_summary=True)
    summary = out["summary"]

    row_tail = summary.loc[summary["metric"] == "psi_p99_over_med"]
    row_kurt = summary.loc[summary["metric"] == "psi_kurtosis"]
    row_oos = summary.loc[summary["metric"] == "oos_max_abs_t"]
    assert not row_tail.empty
    assert not row_kurt.empty
    assert not row_oos.empty
    assert row_tail["flag"].iloc[0] == out["flags"]["psi_tail_ratio"]
    assert row_kurt["flag"].iloc[0] == out["flags"]["psi_kurtosis"]
    assert row_oos["flag"].iloc[0] == out["flags"]["oos_moment"]
