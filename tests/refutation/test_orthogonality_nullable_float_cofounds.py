import numpy as np
import pandas as pd

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def make_nullable_float_data(n=80, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Create nullable Float64 pandas dtype columns
    x1_s = pd.Series(x1, dtype="Float64")
    x2_s = pd.Series(x2, dtype="Float64")
    g = 1 / (1 + np.exp(-(x1 - 0.5 * x2)))
    d = rng.binomial(1, p=np.clip(g, 0.1, 0.9)).astype(float)
    m0 = 0.3 * x1 - 0.2 * x2
    y = m0 + 0.8 * d + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({
        'y': pd.Series(y, dtype='Float64'),
        'd': pd.Series(d, dtype='Float64'),
        'x1': x1_s,
        'x2': x2_s,
    })
    return CausalData(df=df, treatment='d', outcome='y', confounders=['x1', 'x2'])


def test_refute_irm_orthogonality_with_nullable_float_confounders_runs():
    data = make_nullable_float_data(n=80, seed=13)
    estimate = IRM(data, n_folds=3, random_state=13).fit().estimate(score="ATTE")
    res = run_score_diagnostics(data, estimate, n_basis_funcs=3, return_summary=True)

    assert "orthogonality_derivatives" in res
    assert "influence_diagnostics" in res
    assert "summary" in res
    assert res["params"]["score"] == "ATTE"
