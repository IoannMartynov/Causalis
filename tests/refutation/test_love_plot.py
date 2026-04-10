from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.refutation import love_plot
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import (
    love_plot as love_plot_unconf,
)
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)


def _make_data_and_estimate(
    *,
    n: int = 220,
    p: int = 4,
    seed: int = 321,
    include_x_in_diag: bool = True,
    m_hat: np.ndarray | None = None,
    d: np.ndarray | None = None,
    x: np.ndarray | None = None,
) -> tuple[CausalData, CausalEstimate]:
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x, dtype=float) if x is not None else rng.normal(size=(n, p))
    n_obs, n_features = x_arr.shape
    feature_names = [f"x{j + 1}" for j in range(n_features)]

    if m_hat is None:
        logits = 0.8 * x_arr[:, 0]
        if n_features > 1:
            logits = logits - 0.5 * x_arr[:, 1]
        if n_features > 2:
            logits = logits + 0.25 * x_arr[:, 2]
        m_hat_arr = 1.0 / (1.0 + np.exp(-logits))
        m_hat_arr = np.clip(m_hat_arr, 1e-3, 1.0 - 1e-3)
    else:
        m_hat_arr = np.asarray(m_hat, dtype=float).ravel()

    if d is None:
        d_arr = rng.binomial(1, m_hat_arr).astype(int)
    else:
        d_arr = np.asarray(d, dtype=int).ravel()

    g0 = 0.2 + 0.4 * x_arr[:, 0]
    if n_features > 1:
        g0 = g0 - 0.15 * x_arr[:, 1]
    y = g0 + 0.8 * d_arr + rng.normal(scale=0.35, size=n_obs)

    df = pd.DataFrame({"y": y, "d": d_arr})
    for j, name in enumerate(feature_names):
        df[name] = x_arr[:, j]

    data = CausalData(df=df, treatment="d", outcome="y", confounders=feature_names)
    diag = UnconfoundednessDiagnosticData(
        m_hat=m_hat_arr,
        d=d_arr,
        x=x_arr if include_x_in_diag else None,
        score="ATE",
    )

    y_t = y[d_arr == 1]
    y_c = y[d_arr == 0]
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
        n_treated=int(np.sum(d_arr)),
        n_control=int(np.sum(1 - d_arr)),
        treatment_mean=float(np.mean(y_t)),
        control_mean=float(np.mean(y_c)),
        outcome="y",
        treatment="d",
        confounders=feature_names,
        diagnostic_data=diag,
    )
    return data, estimate


def test_love_plot_basic():
    data, estimate = _make_data_and_estimate()

    fig = love_plot(data=data, estimate=estimate)

    assert fig is not None
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    assert "Love Plot" in ax.get_title()
    legend = ax.get_legend()
    legend_text = [text.get_text() for text in legend.get_texts()]
    assert "Before (unweighted)" in legend_text
    assert "After (weighted)" in legend_text
    assert any("Threshold" in text for text in legend_text)
    assert any(np.allclose(line.get_xdata(), [0.10, 0.10]) for line in ax.lines)

    plt.close(fig)


def test_love_plot_falls_back_to_causal_data_for_x():
    data, estimate = _make_data_and_estimate(include_x_in_diag=False)

    fig = love_plot(data=data, estimate=estimate)

    assert fig is not None
    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert len(labels) == len(data.confounders)

    plt.close(fig)


def test_love_plot_orders_by_worst_balance():
    d = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    x = np.array(
        [
            [4.0, 2.0, 1.0],
            [5.0, 2.0, 1.0],
            [5.0, 3.0, 1.0],
            [6.0, 3.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    m_hat = np.full(d.shape[0], 0.5, dtype=float)
    data, estimate = _make_data_and_estimate(x=x, d=d, m_hat=m_hat)

    report = run_unconfoundedness_diagnostics(data, estimate, return_summary=False)
    expected = (
        pd.concat(
            [
                report["balance"]["smd_unweighted"].rename("before"),
                report["balance"]["smd"].rename("after"),
            ],
            axis=1,
        )
        .max(axis=1, skipna=True)
        .sort_values(ascending=False, na_position="last", kind="mergesort")
        .index.tolist()
    )

    fig = love_plot(data=data, estimate=estimate)

    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == expected

    plt.close(fig)


def test_love_plot_large_confounder_set_autoscales_height():
    data, estimate = _make_data_and_estimate(p=65, n=320, seed=17)

    fig = love_plot(data=data, estimate=estimate)

    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert len(labels) == 65
    assert fig.get_size_inches()[1] > 15.0

    plt.close(fig)


def test_love_plot_requires_diagnostic_data():
    data, estimate = _make_data_and_estimate()
    estimate_missing_diag = estimate.model_copy(update={"diagnostic_data": None})

    with pytest.raises(ValueError, match="diagnostic_data"):
        love_plot(data=data, estimate=estimate_missing_diag)


def test_love_plot_validates_estimate_matches_data():
    data, estimate = _make_data_and_estimate()
    estimate_wrong_treatment = estimate.model_copy(update={"treatment": "other_d"})

    with pytest.raises(ValueError, match="estimate.treatment must match"):
        love_plot(data=data, estimate=estimate_wrong_treatment)


def test_unconfoundedness_namespace_exposes_love_plot():
    assert callable(love_plot_unconf)
