import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.model import MultiTreatmentIRM


def _make_multi_causal_data(n: int = 240, seed: int = 42) -> MultiCausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)

    labels = np.tile(np.array([0, 1, 2], dtype=int), int(np.ceil(n / 3)))[:n]
    rng.shuffle(labels)
    d = np.eye(3, dtype=int)[labels]

    effects = np.array([0.0, -0.5, 0.8], dtype=float)
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + effects[labels] + rng.normal(0.0, 0.1, size=n)

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "d0": d[:, 0],
            "d1": d[:, 1],
            "d2": d[:, 2],
        }
    )

    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1", "d2"],
        confounders=["x1", "x2"],
        control_treatment="d0",
    )


def test_parallel_cross_fit_matches_sequential():
    data = _make_multi_causal_data()

    kwargs = dict(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=7,
    )

    model_seq = MultiTreatmentIRM(**kwargs, n_jobs=1).fit()
    model_par = MultiTreatmentIRM(**kwargs, n_jobs=2).fit()

    np.testing.assert_allclose(model_seq.g_hat_, model_par.g_hat_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model_seq.m_hat_, model_par.m_hat_, rtol=1e-10, atol=1e-10)

    est_seq = model_seq.estimate(score="ATE", diagnostic_data=False)
    est_par = model_par.estimate(score="ATE", diagnostic_data=False)

    np.testing.assert_allclose(est_seq.value, est_par.value, rtol=1e-10, atol=1e-10)
