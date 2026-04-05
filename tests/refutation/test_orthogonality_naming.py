import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _toy_data(n: int = 200, seed: int = 0) -> CausalData:
    df = generate_rct(n=n, k=2, random_state=seed, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def _fit_estimate(data: CausalData, score: str = "ATE"):
    return IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=2,
        random_state=5,
    ).fit().estimate(score=score)


def test_score_diagnostics_derivative_column_names():
    data = _toy_data(160)
    estimate = _fit_estimate(data, score="ATE")

    res = run_score_diagnostics(data, estimate, return_summary=True)
    ortho = res["orthogonality_derivatives"]
    assert set(["basis", "d_g1", "se_g1", "t_g1", "d_g0", "se_g0", "t_g0", "d_m", "se_m", "t_m"]).issubset(
        set(ortho.columns)
    )


def test_score_diagnostics_att_score_normalization_and_basic_outputs():
    data = _toy_data(180, seed=1)
    estimate = _fit_estimate(data, score="ATTE")

    res = run_score_diagnostics(data, estimate)
    assert res["params"]["score"] == "ATTE"
    assert np.isfinite(float(res["influence_diagnostics"]["se_plugin"]))
