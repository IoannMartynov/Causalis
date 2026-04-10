"""Integration tests for current refutation diagnostics API."""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation import (
    love_plot,
    run_overlap_diagnostics,
    run_score_diagnostics,
    run_unconfoundedness_diagnostics,
)


def _make_data(n: int = 500, seed: int = 42) -> CausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    logits = 0.5 * x1 + 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, p)
    y = 2.0 + 0.5 * x1 + 0.3 * x2 + 1.5 * d + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2})
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])


def _make_estimate(data: CausalData):
    return IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=7,
    ).fit().estimate(score="ATE")


def test_refutation_namespace_exposes_current_diagnostics():
    import causalis.scenarios.unconfoundedness.refutation as ref

    assert hasattr(ref, "run_overlap_diagnostics")
    assert hasattr(ref, "run_score_diagnostics")
    assert hasattr(ref, "run_unconfoundedness_diagnostics")
    assert hasattr(ref, "love_plot")


def test_overlap_diagnostics_runs_with_causal_estimate():
    data = _make_data(seed=11)
    estimate = _make_estimate(data)

    report = run_overlap_diagnostics(data, estimate)
    assert "summary" in report
    assert "ks" in report
    assert "auc" in report


def test_score_diagnostics_runs_with_causal_estimate():
    data = _make_data(seed=13)
    estimate = _make_estimate(data)

    report = run_score_diagnostics(data, estimate, return_summary=True)
    assert "summary" in report
    assert "orthogonality_derivatives" in report
    assert "influence_diagnostics" in report
    assert report["meta"]["n"] == int(data.get_df().shape[0])


def test_unconfoundedness_diagnostics_runs_with_causal_estimate():
    data = _make_data(seed=17)
    estimate = _make_estimate(data)

    report = run_unconfoundedness_diagnostics(data, estimate)
    assert "summary" in report
    assert "balance" in report
    assert "overall_flag" in report
    assert "smd" in report["balance"]
    assert "smd_unweighted" in report["balance"]
    assert "pass" in report["balance"]


def test_love_plot_runs_with_causal_estimate():
    data = _make_data(seed=19)
    estimate = _make_estimate(data)

    fig = love_plot(data, estimate)
    assert fig is not None
    assert len(fig.axes) == 1


def test_score_diagnostics_fallback_when_y_d_missing():
    data = _make_data(seed=23)
    estimate = _make_estimate(data)

    diag_without_yd = estimate.diagnostic_data.model_copy(update={"y": None, "d": None})
    estimate_without_yd = estimate.model_copy(update={"diagnostic_data": diag_without_yd})

    report = run_score_diagnostics(data, estimate_without_yd, return_summary=True)
    assert "summary" in report
    assert report["meta"]["n"] == int(data.get_df().shape[0])
