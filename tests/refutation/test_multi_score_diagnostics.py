import numpy as np
import pandas as pd
import pytest

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.model import MultiTreatmentIRM
from causalis.scenarios.multi_unconfoundedness.refutation.score.score_validation import run_score_diagnostics


def _make_multi_causal_data(n: int = 180, seed: int = 42) -> MultiCausalData:
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
            "d_0": d[:, 0],
            "d_1": d[:, 1],
            "d_2": d[:, 2],
        }
    )

    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d_0", "d_1", "d_2"],
        confounders=["x1", "x2"],
        control_treatment="d_0",
    )


def _make_estimate(data: MultiCausalData, *, normalize_ipw: bool = False):
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        normalize_ipw=normalize_ipw,
        n_folds=3,
        random_state=1,
    ).fit()
    return model.estimate(score="ATE", diagnostic_data=True)


def test_multi_score_diagnostics_runs_and_returns_long_summary():
    data = _make_multi_causal_data(seed=17)
    estimate = _make_estimate(data)

    report = run_score_diagnostics(data, estimate, return_summary=True)

    assert "summary" in report
    summary = report["summary"]
    assert list(summary.columns) == ["comparison", "metric", "value", "flag"]
    assert {"d_1 vs d_0", "d_2 vs d_0"}.issubset(set(summary["comparison"]))
    assert {"se_plugin", "max_|t|", "oos_tstat_fold", "oos_tstat_strict"}.issubset(
        set(summary["metric"])
    )


def test_multi_score_diagnostics_exposes_core_blocks():
    data = _make_multi_causal_data(seed=33)
    estimate = _make_estimate(data)

    report = run_score_diagnostics(data, estimate, return_summary=True)

    assert "orthogonality_derivatives" in report
    assert "influence_diagnostics" in report
    assert "oos_moment_test" in report
    assert "flags" in report
    assert "flags_by_comparison" in report
    assert report["params"]["score"] == "ATE"


def test_multi_score_refutation_namespace_exposes_runner():
    import causalis.scenarios.multi_unconfoundedness.refutation as ref

    assert hasattr(ref, "run_score_diagnostics")


def test_multi_score_diagnostics_warns_and_disables_hajek_for_orthogonality():
    data = _make_multi_causal_data(seed=71)
    estimate = _make_estimate(data, normalize_ipw=True)

    with pytest.warns(RuntimeWarning, match="normalize_ipw=False"):
        report = run_score_diagnostics(data, estimate, return_summary=True)

    assert report["params"]["normalize_ipw"] is True
    assert report["params"]["orthogonality_normalize_ipw"] is False
    assert report["meta"]["orthogonality_derivatives_use_score_normalization"] is False
