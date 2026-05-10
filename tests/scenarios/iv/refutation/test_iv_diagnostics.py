import numpy as np
import pandas as pd

from causalis.data_contracts.causal_diagnostic_data import IVDiagnosticData
from causalis.data_contracts.iv_causal_estimate import IVCausalEstimate
from causalis.scenarios.iv.refutation import (
    first_stage,
    instrument_overlap,
    instrument_overlap_plot,
    reduced_form,
)


def _make_result(
    *,
    z,
    m_hat,
    d=None,
    y=None,
    x=None,
    phi_y=None,
    phi_d=None,
) -> IVCausalEstimate:
    z = np.asarray(z, dtype=int).ravel()
    n = z.size
    m_hat = np.asarray(m_hat, dtype=float).ravel()
    d = np.asarray(z if d is None else d, dtype=int).ravel()
    y = np.asarray(d.astype(float) if y is None else y, dtype=float).ravel()
    if x is None:
        x = np.empty((n, 0), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

    phi_y = np.asarray(
        np.full(n, 1.0, dtype=float) if phi_y is None else phi_y,
        dtype=float,
    ).ravel()
    phi_d = np.asarray(
        np.full(n, 1.0, dtype=float) if phi_d is None else phi_d,
        dtype=float,
    ).ravel()
    theta = float(np.mean(phi_y) / np.mean(phi_d))
    psi_a = -phi_d
    psi_b = phi_y
    psi = psi_a * theta + psi_b

    diag = IVDiagnosticData(
        y=y,
        d=d,
        z=z,
        x=x,
        x_names=[f"x{i}" for i in range(x.shape[1])],
        g0_hat=np.zeros(n, dtype=float),
        g1_hat=np.ones(n, dtype=float),
        m_hat=m_hat,
        m_hat_raw=m_hat,
        r0_hat=np.zeros(n, dtype=float),
        r1_hat=np.ones(n, dtype=float),
        folds=np.zeros(n, dtype=int),
        psi=psi,
        psi_a=psi_a,
        psi_b=psi_b,
        phi_y=phi_y,
        phi_d=phi_d,
        diagnostics={"n_obs": n},
    )

    return IVCausalEstimate(
        estimand="LATE",
        model="IIVM",
        value=theta,
        std_error=0.1,
        t_stat=theta / 0.1,
        p_value=0.01,
        ci_lower_absolute=theta - 0.2,
        ci_upper_absolute=theta + 0.2,
        alpha=0.05,
        is_significant=True,
        outcome="y",
        treatment="d",
        instrument="z",
        diagnostic_data=diag,
        model_options={"weak_iv_threshold": 1e-2},
    )


def _flag(table: pd.DataFrame, metric: str) -> str:
    return str(table.loc[table["metric"] == metric, "flag"].iloc[0])


def _value(table: pd.DataFrame, metric: str):
    return table.loc[table["metric"] == metric, "value"].iloc[0]


def _assert_schema(table: pd.DataFrame) -> None:
    assert list(table.columns) == ["metric", "value", "flag", "threshold", "message"]


def test_instrument_overlap_returns_schema_and_green_flags():
    z = np.array([0, 1] * 20)
    result = _make_result(z=z, m_hat=np.full(z.size, 0.5))

    table = instrument_overlap(result)

    _assert_schema(table)
    assert set(table["metric"]) == {
        "instrument_auc",
        "instrument_propensity_ks",
        "instrument_ess_ratio",
    }
    assert _flag(table, "instrument_auc") == "GREEN"
    assert _flag(table, "instrument_propensity_ks") == "GREEN"
    assert _flag(table, "instrument_ess_ratio") == "GREEN"
    assert _value(table, "instrument_ess_ratio") == 1.0
    assert result.diagnostic_data.instrument_overlap is not None


def test_instrument_overlap_flags_yellow_auc():
    z = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    m_hat = np.r_[
        np.arange(10, dtype=float) / 10.0,
        np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.96, 0.97]),
    ]
    result = _make_result(z=z, m_hat=m_hat)

    table = instrument_overlap(result)

    assert _flag(table, "instrument_auc") == "YELLOW"


def test_instrument_overlap_flags_anti_predictive_auc_as_red():
    z = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    m_hat = np.r_[np.full(10, 0.9), np.full(10, 0.1)]
    result = _make_result(z=z, m_hat=m_hat)

    table = instrument_overlap(result)

    assert _value(table, "instrument_auc") == 0.0
    assert _flag(table, "instrument_auc") == "RED"


def test_first_stage_flags_strong_and_weak_designs():
    z_strong = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    d_strong = np.r_[
        np.ones(20, dtype=int),
        np.zeros(80, dtype=int),
        np.ones(80, dtype=int),
        np.zeros(20, dtype=int),
    ]
    strong = _make_result(
        z=z_strong,
        d=d_strong,
        m_hat=np.full(z_strong.size, 0.5),
        phi_d=np.full(z_strong.size, 0.6),
    )

    strong_table = first_stage(strong)

    _assert_schema(strong_table)
    assert _flag(strong_table, "weak_iv_flag") == "GREEN"
    assert _value(strong_table, "first_stage_f") >= 10.0

    z_weak = np.array([0, 1] * 100)
    d_weak = np.array([0, 0, 1, 1] * 50)
    weak = _make_result(
        z=z_weak,
        d=d_weak,
        m_hat=np.full(z_weak.size, 0.5),
    )

    weak_table = first_stage(weak)

    assert _flag(weak_table, "weak_iv_flag") == "RED"
    assert _value(weak_table, "first_stage_f") < 4.0


def test_reduced_form_non_significance_is_not_a_failure():
    z = np.array([0, 1] * 100)
    y = np.array([0.0, 0.0, 1.0, 1.0] * 50)
    result = _make_result(
        z=z,
        y=y,
        m_hat=np.full(z.size, 0.5),
        phi_y=np.zeros(z.size, dtype=float),
        phi_d=np.ones(z.size, dtype=float),
    )

    table = reduced_form(result)

    _assert_schema(table)
    assert _flag(table, "reduced_form_pvalue") == "GREEN"
    assert _value(table, "reduced_form_pvalue") > 0.5
    assert _flag(table, "late_ratio_check") == "GREEN"
    assert _value(table, "late_ratio_check") == 0.0


def test_instrument_overlap_plot_returns_matplotlib_figure():
    import matplotlib.pyplot as plt

    z = np.array([0, 1] * 20)
    result = _make_result(z=z, m_hat=np.full(z.size, 0.5))

    fig = instrument_overlap_plot(result)

    assert fig.axes
    plt.close(fig)
