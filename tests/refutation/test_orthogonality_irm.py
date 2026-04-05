from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from causalis.dgp import CausalDatasetGenerator
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def test_score_diagnostics_att_runs_and_returns_expected_columns():
    gen = CausalDatasetGenerator(seed=123)
    cd = gen.to_causal_data(n=300, confounders=["x1", "x2", "x3", "x4", "x5"])

    model = IRM(
        cd,
        ml_g=RandomForestRegressor(n_estimators=30, random_state=1),
        ml_m=RandomForestClassifier(n_estimators=30, random_state=2),
        n_folds=3,
    ).fit()
    estimate = model.estimate(score="ATTE")

    res = run_score_diagnostics(cd, estimate, return_summary=True)

    assert isinstance(res, dict)
    assert "orthogonality_derivatives" in res
    assert "influence_diagnostics" in res
    assert "summary" in res

    ortho = res["orthogonality_derivatives"]
    assert set(["d_g0", "d_m", "t_g0", "t_m"]).issubset(set(ortho.columns))
    assert "se_plugin" in res["influence_diagnostics"]
