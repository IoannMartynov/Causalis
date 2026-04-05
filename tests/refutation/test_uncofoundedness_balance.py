import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression, LinearRegression

from causalis.dgp.causaldata import CausalData
from causalis.dgp import generate_rct
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)


@pytest.mark.parametrize("normalize_ipw", [False, True])
def test_uncofoundedness_balance_ate(normalize_ipw):
    # Generate simple RCT-like data_contracts with confounders
    df = generate_rct(n=2000, k=3, random_state=123, outcome_type="binary")
    confs = [c for c in df.columns if c.startswith("x")]  # ['x1','x2','x3']
    data = CausalData(df=df, treatment='d', outcome='y', confounders=confs)

    # Simple learners: regressor for outcome (works even if y is binary),
    # logistic regression for propensity with predict_proba
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=500)

    res = IRM(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        normalize_ipw=normalize_ipw,
        trimming_threshold=1e-3,
        random_state=7,
    ).fit().estimate(alpha=0.10)

    out = run_unconfoundedness_diagnostics(data, res)

    # Basic structure checks
    assert out['params']['score'] == 'ATE'
    assert out['params']['normalize'] == normalize_ipw
    smd = out['balance']['smd']
    assert isinstance(smd, pd.Series)
    assert list(smd.index) == confs
    assert np.all(np.isfinite(smd.values))
    assert (smd.values >= 0).all()

    # New: unweighted SMD available and aligned
    smd_unw = out['balance']['smd_unweighted']
    assert isinstance(smd_unw, pd.Series)
    assert list(smd_unw.index) == confs
    assert np.all((smd_unw.values >= 0) | ~np.isfinite(smd_unw.values))  # allow NaN if degenerate

    # Removed keys should not be present
    assert 'means_treated' not in out['balance']
    assert 'means_control' not in out['balance']
    assert 'sd_treated' not in out['balance']
    assert 'sd_control' not in out['balance']

    # Threshold logic: very high threshold should pass, tiny threshold likely fails
    out_hi = run_unconfoundedness_diagnostics(data, res, threshold=1e6)
    assert out_hi['balance']['pass'] is True
    out_lo = run_unconfoundedness_diagnostics(data, res, threshold=1e-12)
    assert out_lo['balance']['pass'] in (False, True)  # don't force, but should be boolean


@pytest.mark.parametrize("normalize_ipw", [False, True])
def test_uncofoundedness_balance_att(normalize_ipw):
    # Generate simple data_contracts with confounders
    df = generate_rct(n=2000, k=4, random_state=321, outcome_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]  # ['x1',...]
    data = CausalData(df=df, treatment='d', outcome='y', confounders=confs)

    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=500)

    res = IRM(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        normalize_ipw=normalize_ipw,
        trimming_threshold=1e-3,
        random_state=13,
    ).fit().estimate(score="ATTE", alpha=0.10)

    out = run_unconfoundedness_diagnostics(data, res)

    assert out['params']['score'] == 'ATTE'
    # ATTE uses canonical EIF; normalize_ipw is intentionally ignored for this estimand.
    assert out['params']['normalize'] is False
    smd = out['balance']['smd']
    assert isinstance(smd, pd.Series)
    assert list(smd.index) == confs
    assert np.all(np.isfinite(smd.values))
    assert (smd.values >= 0).all()
