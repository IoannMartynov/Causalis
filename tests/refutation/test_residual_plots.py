import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.refutation.score.residual_plots import (
    plot_residual_diagnostics,
)


def _build_data_and_estimate(*, include_yd_in_diag: bool = True):
    n = 180
    rng = np.random.default_rng(321)
    x1 = rng.normal(size=n)
    m = np.clip(1.0 / (1.0 + np.exp(-0.7 * x1)), 1e-3, 1.0 - 1e-3)
    d = rng.binomial(1, m).astype(int)
    g0 = 0.1 + 0.2 * x1
    g1 = g0 + 0.8
    y = g0 + d * 0.8 + rng.normal(scale=0.2, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x1": x1})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])

    diag = UnconfoundednessDiagnosticData(
        m_hat=m,
        d=d,
        y=y,
        g0_hat=g0,
        g1_hat=g1,
        x=df[["x1"]].to_numpy(dtype=float),
        score="ATE",
    )
    if not include_yd_in_diag:
        diag = diag.model_copy(update={"d": None, "y": None})

    y_t = y[d == 1]
    y_c = y[d == 0]
    estimate = CausalEstimate(
        estimand="ATE",
        model="IRM",
        model_options={"normalize_ipw": False, "trimming_threshold": 1e-3},
        value=float(np.mean(y_t) - np.mean(y_c)),
        ci_upper_absolute=0.2,
        ci_lower_absolute=-0.2,
        alpha=0.05,
        p_value=1.0,
        is_significant=False,
        n_treated=int(np.sum(d)),
        n_control=int(np.sum(1 - d)),
        treatment_mean=float(np.mean(y_t)),
        control_mean=float(np.mean(y_c)),
        outcome="y",
        treatment="d",
        confounders=["x1"],
        diagnostic_data=diag,
    )
    return data, estimate


def test_plot_residual_diagnostics_basic():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    fig = plot_residual_diagnostics(estimate=estimate, data=data)

    assert fig is not None
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == 3
    titles = [ax.get_title() for ax in fig.axes]
    assert any("Treated: Residual vs Fitted" in title for title in titles)
    assert any("Control: Residual vs Fitted" in title for title in titles)
    assert any("Calibration Error by Propensity Bin" in title for title in titles)

    plt.close(fig)


def test_plot_residual_diagnostics_fallback_for_y_d():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=False)
    fig = plot_residual_diagnostics(estimate=estimate, data=data, n_bins=12)

    assert fig is not None
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == 3
    xlabels = [ax.get_xlabel() for ax in fig.axes]
    assert any(r"\hat g_1" in lbl for lbl in xlabels)
    assert any(r"\hat g_0" in lbl for lbl in xlabels)
    assert any(r"\hat m" in lbl for lbl in xlabels)

    plt.close(fig)


def test_plot_residual_diagnostics_limits_scatter_points():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    fig = plot_residual_diagnostics(
        estimate=estimate,
        data=data,
        max_scatter_points=10,
        random_state=7,
    )

    assert fig is not None
    treated_offsets = fig.axes[0].collections[0].get_offsets()
    control_offsets = fig.axes[1].collections[0].get_offsets()
    assert treated_offsets.shape[0] == 10
    assert control_offsets.shape[0] == 10
    assert len(fig.axes[0].lines) >= 2
    assert len(fig.axes[1].lines) >= 2

    plt.close(fig)


def test_plot_residual_diagnostics_uses_cached_inputs_without_data():
    _, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    diag = estimate.diagnostic_data
    residual_cache = {
        "y": np.asarray(diag.y, dtype=float).ravel(),
        "d": np.asarray(diag.d, dtype=float).ravel(),
        "g0": np.asarray(diag.g0_hat, dtype=float).ravel(),
        "g1": np.asarray(diag.g1_hat, dtype=float).ravel(),
        "m": np.asarray(diag.m_hat, dtype=float).ravel(),
    }
    diag_cached_only = diag.model_copy(
        update={
            "m_hat": None,
            "g0_hat": None,
            "g1_hat": None,
            "y": None,
            "d": None,
            "residual_plot_cache": residual_cache,
        }
    )
    estimate_cached_only = estimate.model_copy(update={"diagnostic_data": diag_cached_only})

    fig = plot_residual_diagnostics(estimate=estimate_cached_only, data=None)
    assert fig is not None
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_residual_diagnostics_populates_cache_on_first_call():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    diag_no_cache = estimate.diagnostic_data.model_copy(update={"residual_plot_cache": None})
    estimate_no_cache = estimate.model_copy(update={"diagnostic_data": diag_no_cache})

    fig = plot_residual_diagnostics(estimate=estimate_no_cache, data=data)
    assert fig is not None
    assert isinstance(estimate_no_cache.diagnostic_data.residual_plot_cache, dict)
    plt.close(fig)


def test_plot_residual_diagnostics_requires_g1_hat():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    diag_missing_g1 = estimate.diagnostic_data.model_copy(update={"g1_hat": None})
    estimate_missing_g1 = estimate.model_copy(update={"diagnostic_data": diag_missing_g1})

    with pytest.raises(ValueError, match="g1_hat"):
        plot_residual_diagnostics(estimate=estimate_missing_g1, data=data)
