import numpy as np
import pandas as pd

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM

from sklearn.linear_model import LinearRegression, LogisticRegression


def make_synthetic(n=300, seed=123):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2])
    # Propensity driven by X
    logits = 0.5 * x1 - 0.5 * x2
    p = 1 / (1 + np.exp(-logits))
    d = rng.binomial(1, p)
    # Outcome: treatment effect ~ 2.0 plus some X signal
    y = 1.0 + 2.0 * d + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=1.0, size=n)

    df = pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })
    return df


def fit_irm(normalize_ipw: bool, seed=123):
    df = make_synthetic(seed=seed)
    cd = CausalData(df=df, treatment='d', outcome='y', confounders=['x1', 'x2'])
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000, solver='lbfgs')

    irm = IRM(data=cd, ml_g=ml_g, ml_m=ml_m, n_folds=5, normalize_ipw=normalize_ipw,
              overlap_threshold=1e-6, random_state=42)
    irm.fit().estimate(score="ATTE")
    return irm


def test_att_normalize_ipw_invariance():
    irm_no_norm = fit_irm(normalize_ipw=False)
    irm_norm = fit_irm(normalize_ipw=True)

    theta_no_norm = float(irm_no_norm.coef[0])
    theta_norm = float(irm_norm.coef[0])

    # With the special case removed, ATT estimate is no longer strictly invariant to IPW normalization,
    # but they should still be reasonably close in this synthetic example.
    assert np.isfinite(theta_no_norm) and np.isfinite(theta_norm)
    assert abs(theta_no_norm - theta_norm) < 0.01
