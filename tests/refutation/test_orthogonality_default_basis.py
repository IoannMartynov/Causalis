from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_dataset(n: int = 80, p: int = 4, seed: int = 42) -> CausalData:
    df = generate_rct(n=n, k=p, random_state=seed, outcome_type="normal")
    confounders = [f"x{i+1}" for i in range(p)]
    return CausalData(df=df[["y", "d"] + confounders], treatment="d", outcome="y", confounders=confounders)


def _make_estimate(data: CausalData):
    return IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=400),
        n_folds=3,
        random_state=42,
    ).fit().estimate(score="ATE")


def test_default_basis_uses_all_confounders_plus_constant():
    data = _make_dataset(n=100, p=4)
    estimate = _make_estimate(data)

    res = run_score_diagnostics(data, estimate)
    ortho = res["orthogonality_derivatives"]
    assert len(ortho) == len(data.confounders) + 1


def test_explicit_n_basis_funcs_is_respected():
    data = _make_dataset(n=100, p=4)
    estimate = _make_estimate(data)

    res = run_score_diagnostics(data, estimate, n_basis_funcs=2)
    ortho = res["orthogonality_derivatives"]
    assert len(ortho) == 2
