import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_data(n: int = 200, seed: int = 1) -> CausalData:
    df = generate_rct(n=n, k=3, random_state=seed, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def _make_estimate(data: CausalData, score: str):
    return IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=19,
    ).fit().estimate(score=score)


def test_atte_derivative_wrt_g1_is_zero():
    data = _make_data(n=180, seed=42)
    estimate = _make_estimate(data, score="ATTE")

    report = run_score_diagnostics(data, estimate, n_basis_funcs=4)
    ortho = report["orthogonality_derivatives"]

    assert report["params"]["score"] == "ATTE"
    assert np.allclose(ortho["d_g1"].to_numpy(dtype=float), 0.0)
    assert np.allclose(ortho["t_g1"].to_numpy(dtype=float), 0.0)


def test_atte_has_expected_columns_and_finite_main_stats():
    data = _make_data(n=220, seed=7)
    estimate = _make_estimate(data, score="ATTE")

    report = run_score_diagnostics(data, estimate, return_summary=True)
    ortho = report["orthogonality_derivatives"]

    assert set(["d_g1", "se_g1", "t_g1", "d_g0", "se_g0", "t_g0", "d_m", "se_m", "t_m"]).issubset(
        set(ortho.columns)
    )
    assert np.isfinite(float(report["influence_diagnostics"]["se_plugin"]))
    assert "summary" in report
