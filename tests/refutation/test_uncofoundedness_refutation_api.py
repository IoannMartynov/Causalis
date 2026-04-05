import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp import generate_rct
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)


def _make_data(n: int = 1500, k: int = 4, seed: int = 123) -> CausalData:
    df = generate_rct(n=n, k=k, random_state=seed, outcome_type="normal")
    confs = [column for column in df.columns if column.startswith("x")]
    return CausalData(df=df, treatment="d", outcome="y", confounders=confs)


def _make_estimate(data: CausalData, *, score: str = "ATE"):
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=400),
        n_folds=3,
        normalize_ipw=True,
        trimming_threshold=1e-3,
        random_state=7,
    ).fit()
    return model.estimate(score=score, alpha=0.10)


def test_uncofoundedness_single_api_with_causal_estimate():
    data = _make_data(seed=17)
    estimate = _make_estimate(data)

    report = run_unconfoundedness_diagnostics(data, estimate)

    assert "summary" in report
    assert report["params"]["score"] == "ATE"
    assert isinstance(report["balance"]["smd"], pd.Series)
    assert list(report["balance"]["smd"].index) == list(data.confounders)
    assert isinstance(report["balance"]["smd_unweighted"], pd.Series)
    assert "pass" in report["balance"]


def test_uncofoundedness_single_api_with_causal_estimate_and_causal_data_fallback():
    data = _make_data(seed=33)
    estimate = _make_estimate(data)

    diag_without_x = estimate.diagnostic_data.model_copy(update={"x": None})
    estimate_without_x = estimate.model_copy(update={"diagnostic_data": diag_without_x})

    report = run_unconfoundedness_diagnostics(data, estimate_without_x)

    assert report["meta"]["n"] == int(data.get_df().shape[0])
    assert list(report["balance"]["smd"].index) == list(data.confounders)


def test_atte_keeps_canonical_unnormalized_weights():
    data = _make_data(seed=71)
    estimate = _make_estimate(data, score="ATTE")

    report = run_unconfoundedness_diagnostics(data, estimate, normalize=True)
    assert report["params"]["score"] == "ATTE"
    assert report["params"]["normalize"] is False
