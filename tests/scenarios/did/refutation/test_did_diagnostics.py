from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import pandas as pd

from causalis.data_contracts import PanelDataDID
from causalis.scenarios.did import (
    did_base_design_table,
    did_covariate_balance_table,
    did_support_table,
    plot_did_support,
    plot_raw_did_event_study,
    raw_did_event_study_table,
    run_did_diagnostics,
)


def _panel(*, cluster_col: str | None = "cluster") -> PanelDataDID:
    periods = pd.period_range("2020-01", periods=6, freq="M")
    units = {
        "A1": ("2020-04", 2.0, 10.0, 0.0, "north"),
        "A2": ("2020-04", 2.0, 11.0, 0.2, "north"),
        "B1": ("2020-05", 3.0, 20.0, 1.0, "south"),
        "B2": ("2020-05", 3.0, 21.0, 1.2, "south"),
        "C1": (None, 0.0, 30.0, 0.5, "control"),
        "C2": (None, 0.0, 31.0, 0.7, "control"),
    }
    rows = []
    for unit, (cohort, tau, base, x, cluster) in units.items():
        cohort_period = None if cohort is None else pd.Period(cohort, freq="M")
        for idx, period in enumerate(periods):
            treated = cohort_period is not None and period >= cohort_period
            row = {
                "unit": unit,
                "time": period,
                "y": base + idx + (tau if treated else 0.0),
                "d": int(treated),
                "x": x + 0.01 * idx,
            }
            if cluster_col == "cluster":
                row["cluster"] = cluster
            rows.append(row)

    kwargs = {
        "df": pd.DataFrame(rows),
        "y": "y",
        "unit_col": "unit",
        "time_col": "time",
        "treated_time": "d",
        "covariates": ["x"],
    }
    if cluster_col is not None:
        kwargs["cluster_col"] = cluster_col
    return PanelDataDID(**kwargs)


def test_did_diagnostics_tables_from_panel_data():
    panel = _panel()

    report = run_did_diagnostics(
        panel,
        min_treated_per_cell=1,
        min_control_per_cell=1,
        max_abs_covariate_smd=10.0,
    )
    support = did_support_table(panel)
    raw_event = raw_did_event_study_table(panel)
    balance = did_covariate_balance_table(panel)
    design = did_base_design_table(panel)

    assert list(report.columns) == ["test", "flag", "value", "threshold", "message"]
    assert set(report["test"]).issuperset(
        {
            "requested_cs_post_support",
            "raw_pretrend_placebo",
            "max_base_covariate_smd",
            "base_control_design_rank",
        }
    )
    assert report.loc[
        report["test"] == "requested_cs_post_support",
        "flag",
    ].iloc[0] == "GREEN"
    assert {"treated_completion_rate", "control_to_treated_ratio"}.issubset(
        support.columns
    )
    assert raw_event["event_time"].min() < 0
    assert {"covariate", "smd", "abs_smd"}.issubset(balance.columns)
    assert {"control_design_rank", "condition_number"}.issubset(design.columns)


def test_cluster_by_time_is_flagged_before_callaway_santanna_fit():
    panel = _panel(cluster_col="time")

    report = run_did_diagnostics(
        panel,
        min_treated_per_cell=1,
        min_control_per_cell=1,
    )

    row = report.loc[report["test"] == "cluster_readiness"].iloc[0]
    assert row["flag"] == "RED"
    assert "not stable within unit" in row["message"]


def test_did_diagnostic_plots_from_panel_data():
    panel = _panel()

    support_fig = plot_did_support(panel)
    event_fig = plot_raw_did_event_study(panel)

    assert isinstance(support_fig, mpl_figure.Figure)
    assert isinstance(event_fig, mpl_figure.Figure)
    assert not plt.fignum_exists(support_fig.number)
    assert not plt.fignum_exists(event_fig.number)
    assert support_fig.axes[0].get_title() == "DID Support by Cohort and Time"
    assert event_fig.axes[0].get_title() == "Raw DID Event Study"


def test_did_refutation_namespace_exports():
    import causalis.scenarios.did as did
    import causalis.scenarios.did.refutation as ref

    assert hasattr(ref, "run_did_diagnostics")
    assert hasattr(ref, "plot_did_support")
    assert hasattr(did, "run_did_diagnostics")
    assert hasattr(did, "plot_raw_did_event_study")
