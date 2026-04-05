import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM


def _make_data(n: int = 300, seed: int = 123) -> CausalData:
    df = generate_rct(n=n, k=3, random_state=seed, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def test_parallel_cross_fit_matches_sequential():
    data = _make_data()

    kwargs = dict(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=7,
    )

    irm_seq = IRM(**kwargs, n_jobs=1).fit(store_diagnostics=False)
    irm_par = IRM(**kwargs, n_jobs=2).fit(store_diagnostics=False)

    np.testing.assert_allclose(irm_seq.g0_hat_, irm_par.g0_hat_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(irm_seq.g1_hat_, irm_par.g1_hat_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(irm_seq.m_hat_, irm_par.m_hat_, rtol=1e-10, atol=1e-10)

    est_seq = irm_seq.estimate(score="ATE")
    est_par = irm_par.estimate(score="ATE")

    assert np.isclose(est_seq.value, est_par.value)
