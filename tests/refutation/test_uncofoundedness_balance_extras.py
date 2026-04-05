import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.dgp import generate_rct
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)


def test_uncofoundedness_balance_extras_outputs():
    df = generate_rct(n=1500, k=4, random_state=17, outcome_type="binary")
    confs = [c for c in df.columns if c.startswith("x")]
    data = CausalData(df=df, treatment='d', outcome='y', confounders=confs)

    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=400)

    res = IRM(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        normalize_ipw=True,
        trimming_threshold=1e-3,
        random_state=11,
    ).fit().estimate(alpha=0.10)

    out = run_unconfoundedness_diagnostics(data, res)["balance"]
    # New fields should be present and of correct types
    assert 'smd_max' in out
    assert 'worst_features' in out
    assert np.isfinite(float(out['smd_max'])) or np.isnan(float(out['smd_max']))

    worst = out['worst_features']
    assert isinstance(worst, pd.Series)
    # worst_features index must be subset of confounders
    assert set(worst.index).issubset(set(confs))
    # Values non-negative
    assert (worst.values >= 0).all()
