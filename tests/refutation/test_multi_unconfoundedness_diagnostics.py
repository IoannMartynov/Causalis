import numpy as np
import pandas as pd
import pytest
import matplotlib
from typing import Optional

matplotlib.use("Agg")

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from causalis.data_contracts.causal_diagnostic_data import MultiUnconfoundednessDiagnosticData
from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.model import MultiTreatmentIRM
from causalis.scenarios.multi_unconfoundedness.refutation.overlap.overlap_validation import (
    run_overlap_diagnostics
)
from causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)


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


def _make_estimate(
    data: MultiCausalData,
    *,
    score: str = "ATE",
) -> MultiCausalEstimate:
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    ).fit()
    return model.estimate(score=score, diagnostic_data=True)


def _make_manual_multi_data(
    *,
    d: np.ndarray,
    treatment_names: list[str],
) -> MultiCausalData:
    d = np.asarray(d, dtype=int)
    n, k = d.shape
    if len(treatment_names) != k:
        raise ValueError("treatment_names length must match d.shape[1].")

    df_dict = {"y": np.linspace(0.0, 1.0, n)}
    for idx, name in enumerate(treatment_names):
        df_dict[str(name)] = d[:, idx]

    return MultiCausalData(
        df=pd.DataFrame(df_dict),
        outcome="y",
        treatment_names=treatment_names,
        confounders=[],
        control_treatment=treatment_names[0],
    )


def _make_manual_estimate(
    *,
    data: MultiCausalData,
    d: np.ndarray,
    m_hat: np.ndarray,
    m_hat_raw: Optional[np.ndarray] = None,
    normalize_ipw: bool = False,
    trimming_threshold: float = 0.01,
) -> MultiCausalEstimate:
    d = np.asarray(d, dtype=float)
    m_hat = np.asarray(m_hat, dtype=float)
    k = m_hat.shape[1]

    diag = MultiUnconfoundednessDiagnosticData(
        m_hat=m_hat,
        m_hat_raw=None if m_hat_raw is None else np.asarray(m_hat_raw, dtype=float),
        d=d,
        trimming_threshold=float(trimming_threshold),
        normalize_ipw=bool(normalize_ipw),
        score="ATE",
    )

    zeros = np.zeros(k - 1, dtype=float)
    return MultiCausalEstimate(
        estimand="ATE",
        model="MultiTreatmentIRM",
        model_options={
            "normalize_ipw": bool(normalize_ipw),
            "trimming_threshold": float(trimming_threshold),
        },
        value=zeros,
        ci_upper_absolute=zeros,
        ci_lower_absolute=zeros,
        alpha=0.05,
        p_value=np.ones(k - 1, dtype=float),
        is_significant=[False] * (k - 1),
        n_treated=int(np.sum(d[:, 1:] == 1.0)),
        n_control=int(np.sum(d[:, 0] == 1.0)),
        outcome=data.outcome,
        treatment=list(data.treatment_names),
        contrast_labels=[f"{name} vs {data.treatment_names[0]}" for name in data.treatment_names[1:]],
        diagnostic_data=diag,
    )


def test_multi_unconfoundedness_summary_is_long_with_comparisons():
    data = _make_multi_causal_data()
    estimate = _make_estimate(data)

    report = run_unconfoundedness_diagnostics(data, estimate)
    summary = report["summary"]

    assert list(summary.columns) == ["comparison", "metric", "value", "flag"]
    assert {"d_0 vs d_1", "d_0 vs d_2", "overall"}.issubset(set(summary["comparison"]))
    assert {"balance_max_smd", "balance_frac_violations", "balance_pass"}.issubset(
        set(summary["metric"])
    )


def test_multi_unconfoundedness_balance_tables_are_pairwise():
    data = _make_multi_causal_data(seed=7)
    estimate = _make_estimate(data)

    report = run_unconfoundedness_diagnostics(data, estimate)

    smd = report["balance"]["smd"]
    smd_unweighted = report["balance"]["smd_unweighted"]
    by_comparison = report["balance"]["by_comparison"]

    assert list(smd.index) == ["x1", "x2"]
    assert list(smd.columns) == ["d_0 vs d_1", "d_0 vs d_2"]
    assert list(smd_unweighted.columns) == ["d_0 vs d_1", "d_0 vs d_2"]

    assert list(by_comparison["comparison"]) == ["d_0 vs d_1", "d_0 vs d_2"]
    assert {"smd_max", "frac_violations", "pass", "overall_flag"}.issubset(
        set(by_comparison.columns)
    )


def test_multi_unconfoundedness_requires_strict_input_types():
    data = _make_multi_causal_data(seed=9)
    estimate = _make_estimate(data)

    with pytest.raises(TypeError, match="MultiCausalData"):
        run_unconfoundedness_diagnostics(data.get_df(), estimate)

    with pytest.raises(TypeError, match="MultiCausalEstimate"):
        run_unconfoundedness_diagnostics(data, {"diagnostic_data": estimate.diagnostic_data})


def test_multi_overlap_summary_is_long_with_comparisons():
    data = _make_multi_causal_data(seed=123)
    estimate = _make_estimate(data)

    report = run_overlap_diagnostics(data, estimate)
    summary = report["summary"]

    assert list(summary.columns) == ["comparison", "metric", "value", "flag"]
    assert {"d_0 vs d_1", "d_0 vs d_2"}.issubset(set(summary["comparison"]))
    assert {
        "edge_0.01_below",
        "edge_0.01_above",
        "KS",
        "AUC",
        "ESS_treated_ratio",
        "ESS_baseline_ratio",
        "clip_m_total",
        "overlap_pass",
    }.issubset(set(summary["metric"]))
    assert report["meta"]["propensity_source"] == "m_hat_raw"


def test_multi_overlap_requires_strict_input_types():
    data = _make_multi_causal_data(seed=99)
    estimate = _make_estimate(data)

    with pytest.raises(TypeError, match="MultiCausalData"):
        run_overlap_diagnostics(data.get_df(), estimate)

    with pytest.raises(TypeError, match="MultiCausalEstimate"):
        run_overlap_diagnostics(data, {"diagnostic_data": estimate.diagnostic_data})


def test_multi_overlap_diagnostics_support_atte_estimates():
    data = _make_multi_causal_data(seed=141)
    estimate = _make_estimate(data, score="ATTE")

    report = run_overlap_diagnostics(data, estimate)

    assert report["params"]["score"] == "ATTE"
    assert report["params"]["use_hajek"] is False
    assert list(report["summary"].columns) == ["comparison", "metric", "value", "flag"]
    assert {"d_0 vs d_1", "d_0 vs d_2"}.issubset(set(report["summary"]["comparison"]))


def test_multi_unconfoundedness_diagnostics_support_atte_estimates():
    data = _make_multi_causal_data(seed=141)
    estimate = _make_estimate(data, score="ATTE")

    report = run_unconfoundedness_diagnostics(data, estimate)

    assert report["params"]["score"] == "ATTE"
    assert report["params"]["normalize"] is False
    assert list(report["summary"].columns) == ["comparison", "metric", "value", "flag"]
    assert list(report["balance"]["by_comparison"]["comparison"]) == ["d_0 vs d_1", "d_0 vs d_2"]


def test_multi_refutation_namespace_exposes_overlap_runner():
    import causalis.scenarios.multi_unconfoundedness.refutation as ref

    assert hasattr(ref, "run_overlap_diagnostics")


def test_multi_overlap_plot_wrapper_api_works_with_data_and_estimate():
    data = _make_multi_causal_data(seed=321)
    estimate = _make_estimate(data)

    from causalis.scenarios.multi_unconfoundedness.refutation.overlap import overlap_plot

    fig = overlap_plot(data, estimate)
    assert fig is not None


def test_multi_overlap_uses_pairwise_conditional_propensity_for_ks():
    n0, n1, n2 = 30, 30, 10
    d = np.zeros((n0 + n1 + n2, 3), dtype=int)
    d[:n0, 0] = 1
    d[n0:n0 + n1, 1] = 1
    d[n0 + n1:, 2] = 1

    m_hat = np.zeros((n0 + n1 + n2, 3), dtype=float)
    m_hat[:n0, :] = np.array([0.70, 0.20, 0.10])   # baseline rows
    m_hat[n0:n0 + n1, :] = np.array([0.20, 0.20, 0.60])  # treated rows: same m_1 as baseline
    m_hat[n0 + n1:, :] = np.array([0.10, 0.10, 0.80])  # other arm

    data = _make_manual_multi_data(d=d, treatment_names=["d_0", "d_1", "d_2"])
    estimate = _make_manual_estimate(data=data, d=d, m_hat=m_hat)

    report = run_overlap_diagnostics(data, estimate)
    by_comp = report["overlap"]["by_comparison"]
    row = by_comp[by_comp["comparison"] == "d_0 vs d_1"].iloc[0]

    assert float(row["ks"]) > 0.9


def test_multi_overlap_hajek_tail_scale_matches_pair_sample_mean():
    n0, n1 = 80, 20
    d = np.zeros((n0 + n1, 2), dtype=int)
    d[:n0, 0] = 1
    d[n0:, 1] = 1

    m_hat = np.zeros((n0 + n1, 2), dtype=float)
    m_hat[:n0, :] = np.array([0.90, 0.10])
    m_hat[n0:, :] = np.array([0.80, 0.20])

    data = _make_manual_multi_data(d=d, treatment_names=["d_0", "d_1"])
    estimate = _make_manual_estimate(
        data=data,
        d=d,
        m_hat=m_hat,
        normalize_ipw=True,
    )

    report = run_overlap_diagnostics(data, estimate)
    row = report["overlap"]["by_comparison"].iloc[0]

    assert np.isclose(float(row["tails_treated_median"]), 5.0, atol=1e-12, rtol=0.0)


def test_multi_overlap_clip_total_is_union_share_not_sum():
    n0, n1, n2 = 20, 20, 10
    d = np.zeros((n0 + n1 + n2, 3), dtype=int)
    d[:n0, 0] = 1
    d[n0:n0 + n1, 1] = 1
    d[n0 + n1:, 2] = 1

    m_hat = np.zeros((n0 + n1 + n2, 3), dtype=float)
    m_hat[:n0 + n1, :] = np.array([0.10, 0.10, 0.80])  # both baseline and treated are at clipping edge
    m_hat[n0 + n1:, :] = np.array([0.20, 0.20, 0.60])

    data = _make_manual_multi_data(d=d, treatment_names=["d_0", "d_1", "d_2"])
    estimate = _make_manual_estimate(
        data=data,
        d=d,
        m_hat=m_hat,
        trimming_threshold=0.10,
    )

    report = run_overlap_diagnostics(data, estimate)
    by_comp = report["overlap"]["by_comparison"]
    row = by_comp[by_comp["comparison"] == "d_0 vs d_1"].iloc[0]

    assert np.isclose(float(row["clip_m_total"]), 1.0, atol=1e-12, rtol=0.0)


def test_multi_overlap_prefers_raw_propensity_when_available():
    n0, n1 = 30, 30
    d = np.zeros((n0 + n1, 2), dtype=int)
    d[:n0, 0] = 1
    d[n0:, 1] = 1

    m_hat_raw = np.zeros((n0 + n1, 2), dtype=float)
    m_hat_raw[:, :] = np.array([0.999, 0.001])
    m_hat_post = np.zeros((n0 + n1, 2), dtype=float)
    m_hat_post[:, :] = np.array([0.5, 0.5])

    data = _make_manual_multi_data(d=d, treatment_names=["d_0", "d_1"])
    estimate = _make_manual_estimate(
        data=data,
        d=d,
        m_hat=m_hat_post,
        m_hat_raw=m_hat_raw,
    )

    report = run_overlap_diagnostics(data, estimate)
    row = report["overlap"]["by_comparison"].iloc[0]

    assert report["meta"]["propensity_source"] == "m_hat_raw"
    assert np.isclose(float(row["edge_0.01_below"]), 1.0, atol=1e-12, rtol=0.0)
