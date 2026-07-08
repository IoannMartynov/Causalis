import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.refutation.score.influence_plot import (
    plot_influence_instability,
)


def _build_data_and_estimate(*, include_yd_in_diag: bool = True):
    n = 120
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=n)
    m = np.clip(1.0 / (1.0 + np.exp(-0.8 * x1)), 1e-3, 1.0 - 1e-3)
    d = rng.binomial(1, m).astype(int)
    g0 = 0.2 + 0.1 * x1
    g1 = g0 + 0.7
    y = g0 + d * 0.7 + rng.normal(scale=0.15, size=n)

    df = pd.DataFrame({"user_id": [f"user_{i}" for i in range(n)], "y": y, "d": d, "x1": x1})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"], user_id="user_id")

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
        model_options={"normalize_ipw": False, "overlap_threshold": 1e-3},
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


def test_plot_influence_instability_returns_top_k_panels():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    fig = plot_influence_instability(estimate=estimate, data=data, top_k=7)

    assert fig is not None
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == 2

    ax_rank, ax_scatter = fig.axes
    assert "Top-7 |psi| Contributions" == ax_rank.get_title()
    assert "Top Influential Points by Propensity" == ax_scatter.get_title()
    assert len(ax_rank.patches) == 7
    assert all("user_id=user_" in tick.get_text() for tick in ax_rank.get_yticklabels())

    plt.close(fig)


def test_plot_influence_instability_falls_back_to_data_for_y_d():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=False)
    fig = plot_influence_instability(estimate=estimate, data=data, top_k=5)

    assert fig is not None
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 5

    plt.close(fig)



def test_plot_influence_instability_populates_cache_on_first_call():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    diag_no_cache = estimate.diagnostic_data.model_copy(update={"score_plot_cache": None})
    estimate_no_cache = estimate.model_copy(update={"diagnostic_data": diag_no_cache})

    fig = plot_influence_instability(estimate=estimate_no_cache, data=data, top_k=4)
    assert fig is not None
    cache = estimate_no_cache.diagnostic_data.score_plot_cache
    assert isinstance(cache, dict)
    assert {"score", "overlap_threshold", "d", "m_clipped", "psi", "row_index"}.issubset(set(cache))
    plt.close(fig)
