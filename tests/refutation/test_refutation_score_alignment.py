import numpy as np
import pandas as pd

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)


def _build_estimate(data: CausalData, diag: UnconfoundednessDiagnosticData) -> CausalEstimate:
    d = np.asarray(diag.d, dtype=int).ravel()
    y = np.asarray(diag.y, dtype=float).ravel()
    y_t = y[d == 1]
    y_c = y[d == 0]
    return CausalEstimate(
        estimand="ATE",
        model="IRM",
        model_options={"normalize_ipw": bool(getattr(diag, "normalize_ipw", False))},
        value=0.0,
        ci_upper_absolute=0.1,
        ci_lower_absolute=-0.1,
        alpha=0.05,
        p_value=1.0,
        is_significant=False,
        n_treated=int(np.sum(d)),
        n_control=int(np.sum(1 - d)),
        treatment_mean=float(np.mean(y_t)) if y_t.size else 0.0,
        control_mean=float(np.mean(y_c)) if y_c.size else 0.0,
        outcome="y",
        treatment="d",
        confounders=list(data.confounders),
        diagnostic_data=diag,
    )


def test_score_diagnostics_prefers_estimator_psi_when_available():
    n = 20
    d = np.array([0, 1] * (n // 2), dtype=int)
    m = np.full(n, 0.5, dtype=float)
    x1 = np.linspace(-1.0, 1.0, n)
    y = 0.2 + 0.8 * d + 0.1 * x1
    g0 = np.full(n, 0.2, dtype=float)
    g1 = np.full(n, 1.0, dtype=float)

    psi = np.linspace(-2.0, 2.0, n, dtype=float)
    expected_se = float(np.std(psi, ddof=1) / np.sqrt(n))

    df = pd.DataFrame({"y": y, "d": d, "x1": x1})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])
    diag = UnconfoundednessDiagnosticData(
        m_hat=m,
        d=d,
        y=y,
        x=df[["x1"]].to_numpy(dtype=float),
        g0_hat=g0,
        g1_hat=g1,
        psi=psi,
        score="ATE",
    )
    estimate = _build_estimate(data, diag)

    report = run_score_diagnostics(data, estimate)

    assert report["meta"]["used_estimator_psi"] is True
    assert np.isclose(float(report["influence_diagnostics"]["se_plugin"]), expected_se)


def test_score_diagnostics_reports_oos_moment_tstats_from_folds():
    d = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    y = np.linspace(0.1, 0.8, d.size)
    x1 = np.linspace(-1.0, 1.0, d.size)
    m = np.full(d.size, 0.5, dtype=float)
    g0 = np.full(d.size, 0.1, dtype=float)
    g1 = np.full(d.size, 0.9, dtype=float)
    psi_b = np.array([1.0, 2.0, -1.0, -2.0, 3.0, 4.0, -3.0, -4.0], dtype=float)
    folds = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)

    df = pd.DataFrame({"y": y, "d": d, "x1": x1})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])
    diag = UnconfoundednessDiagnosticData(
        m_hat=m,
        d=d,
        y=y,
        x=df[["x1"]].to_numpy(dtype=float),
        g0_hat=g0,
        g1_hat=g1,
        psi_b=psi_b,
        folds=folds,
        score="ATE",
    )
    estimate = _build_estimate(data, diag)

    report = run_score_diagnostics(data, estimate, return_summary=True)
    oos = report["oos_moment_test"]

    assert oos["available"] is True
    assert np.isclose(float(oos["oos_tstat_fold"]), 0.0)
    assert np.isclose(float(oos["oos_tstat_strict"]), 0.0)
    assert np.isclose(float(oos["p_value_fold"]), 1.0)
    assert np.isclose(float(oos["p_value_strict"]), 1.0)
    assert isinstance(oos["fold_table"], pd.DataFrame)
    assert oos["fold_table"].shape[0] == 2

    summary = report["summary"]
    assert "oos_max_abs_t" in set(summary["metric"])
    summary_oos = summary.loc[summary["metric"] == "oos_max_abs_t"]
    assert not summary_oos.empty
    assert np.isclose(float(summary_oos["value"].iloc[0]), 0.0)


def test_unconfoundedness_uses_w_bar_for_weighted_ate_balance():
    d = np.array([1, 1, 0, 0], dtype=int)
    m = np.full(4, 0.5, dtype=float)
    x1 = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    y = 0.5 + d

    df = pd.DataFrame({"y": y, "d": d, "x1": x1})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])

    diag_unweighted = UnconfoundednessDiagnosticData(
        m_hat=m,
        d=d,
        y=y,
        x=df[["x1"]].to_numpy(dtype=float),
        g0_hat=np.zeros(4),
        g1_hat=np.ones(4),
        score="ATE",
    )
    est_unweighted = _build_estimate(data, diag_unweighted)
    out_unweighted = run_unconfoundedness_diagnostics(data, est_unweighted)

    diag_weighted = diag_unweighted.model_copy(update={"w_bar": np.array([10.0, 1.0, 1.0, 1.0])})
    est_weighted = _build_estimate(data, diag_weighted)
    out_weighted = run_unconfoundedness_diagnostics(data, est_weighted)

    smd_unweighted = float(out_unweighted["balance"]["smd"]["x1"])
    smd_weighted = float(out_weighted["balance"]["smd"]["x1"])

    assert np.isclose(smd_unweighted, 0.0)
    assert smd_weighted > 0.0


def test_overlap_prefers_raw_propensity_when_available():
    d = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    y = np.array([0.1, 0.9, 0.2, 1.0, 0.1, 1.1], dtype=float)
    x1 = np.arange(d.size, dtype=float)

    m_raw = np.array([0.001, 0.999, 0.10, 0.90, 0.20, 0.80], dtype=float)
    m_clipped = np.clip(m_raw, 0.1, 0.9)

    df = pd.DataFrame({"y": y, "d": d, "x1": x1})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])
    diag = UnconfoundednessDiagnosticData(
        m_hat=m_clipped,
        m_hat_raw=m_raw,
        d=d,
        y=y,
        trimming_threshold=0.1,
    )
    estimate = _build_estimate(data, diag)

    report = run_overlap_diagnostics(data, estimate)

    assert report["meta"]["propensity_source"] == "m_hat_raw"
    assert report["edge_mass"]["share_below_001"] > 0.0
