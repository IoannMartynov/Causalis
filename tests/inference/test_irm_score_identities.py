import numpy as np
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import CausalDatasetGenerator
from causalis.scenarios.unconfoundedness.model import IRM


@pytest.mark.parametrize("normalize_ipw", [False, True])
def test_irm_atte_score_identities(normalize_ipw):
    # Build a dataset with cofounding so ATT is well-defined
    gen = CausalDatasetGenerator(
        theta=0.8,
        beta_y=np.array([0.7, -0.2, 0.3]),
        beta_d=np.array([0.9, 0.1, -0.1]),
        outcome_type="continuous",
        sigma_y=1.0,
        k=3,
        seed=101,
    )
    cd = gen.to_causal_data(n=2500)

    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=500)

    irm = IRM(
        cd,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        normalize_ipw=normalize_ipw,
        trimming_threshold=1e-3,
        random_state=3,
    ).fit()
    res = irm.estimate(score="ATTE", alpha=0.05)

    psi_a = irm.psi_a_
    psi = irm.psi_

    # Score identities
    assert np.isclose(np.mean(psi_a), -1.0, atol=1e-10)
    assert np.isclose(np.mean(psi), 0.0, atol=1e-8)
