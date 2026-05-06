from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import pandas as pd

from causalis.data_contracts import PanelDataDID
from causalis.scenarios.did import (
    CallawaySantAnnaDID,
    did_cluster_influence_table,
    did_influence_table,
    did_post_inference_cell_table,
    plot_did_influence_concentration,
    plot_did_post_inference_event_study,
    run_did_inference_diagnostics,
    run_did_post_inference_diagnostics,
)


def _panel() -> PanelDataDID:
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
            rows.append(
                {
                    "unit": unit,
                    "time": period,
                    "y": base + idx + (tau if treated else 0.0),
                    "d": int(treated),
                    "x": x + 0.01 * idx,
                    "cluster": cluster,
                }
            )
    return PanelDataDID(
        df=pd.DataFrame(rows),
        y="y",
        unit_col="unit",
        time_col="time",
        treated_time="d",
        covariates=["x"],
        cluster_col="cluster",
    )


def _estimate(*, diagnostic_data: bool = True):
    panel = _panel()
    model = CallawaySantAnnaDID(
        control_group="never_treated",
        include_pre_periods=True,
        base_period="varying",
        min_treated_per_cell=1,
        min_control_per_cell=1,
        min_control_ess=1.0,
        max_propensity_clip_share=1.0,
        max_condition_number=1e12,
        diagnostic_data=diagnostic_data,
    ).fit(panel)
    return panel, model.estimate(diagnostic_data=diagnostic_data)


def _relaxed_report(panel, estimate):
    return run_did_post_inference_diagnostics(
        panel,
        estimate,
        min_control_ess=1.0,
        max_propensity_clip_share=1.0,
        max_abs_weighted_smd=10.0,
        max_top_unit_influence_share=1.0,
        max_top_cluster_influence_share=1.0,
        min_influence_ess=1.0,
        max_abs_pretrend_t_stat=10.0,
        max_simple_cell_weight_share=1.0,
    )


def test_post_inference_report_accepts_panel_and_estimate():
    panel, estimate = _estimate()

    report = _relaxed_report(panel, estimate)
    cells = did_post_inference_cell_table(estimate)
    influence = did_influence_table(estimate)
    cluster_influence = did_cluster_influence_table(panel, estimate)

    assert list(report.columns) == ["test", "flag", "value", "threshold", "message"]
    assert report.loc[0, "test"] == "overall_inference_reliability"
    assert report.loc[0, "flag"] == "GREEN"
    assert "I can rely on the results" in report.loc[0, "message"]
    assert {"t_stat", "abs_t_stat"}.issubset(cells.columns)
    assert {"rank", "abs_influence_share"}.issubset(influence.columns)
    assert {"cluster", "abs_influence_share"}.issubset(cluster_influence.columns)


def test_post_inference_report_accepts_estimate_alone_with_cautions():
    _, estimate = _estimate()

    report = run_did_inference_diagnostics(
        estimate,
        min_control_ess=1.0,
        max_propensity_clip_share=1.0,
        max_abs_weighted_smd=10.0,
        max_top_unit_influence_share=1.0,
        min_influence_ess=1.0,
        max_abs_pretrend_t_stat=10.0,
        max_simple_cell_weight_share=1.0,
    )

    assert report.loc[0, "test"] == "overall_inference_reliability"
    assert report.loc[0, "flag"] == "YELLOW"
    assert "data_estimate_alignment" in set(report["test"])
    assert "cluster_influence_concentration" in set(report["test"])


def test_post_inference_report_flags_missing_diagnostic_payload():
    panel, estimate = _estimate(diagnostic_data=False)

    report = _relaxed_report(panel, estimate)

    payload_row = report.loc[report["test"] == "diagnostic_payload_available"].iloc[0]
    influence_row = report.loc[report["test"] == "unit_influence_concentration"].iloc[0]
    assert report.loc[0, "flag"] == "YELLOW"
    assert payload_row["flag"] == "YELLOW"
    assert influence_row["flag"] == "YELLOW"


def test_post_inference_plots_from_estimate():
    panel, estimate = _estimate()

    event_fig = plot_did_post_inference_event_study(estimate)
    influence_fig = plot_did_influence_concentration(panel, estimate, top_n=4)

    assert isinstance(event_fig, mpl_figure.Figure)
    assert isinstance(influence_fig, mpl_figure.Figure)
    assert not plt.fignum_exists(event_fig.number)
    assert not plt.fignum_exists(influence_fig.number)
    assert event_fig.axes[0].get_title() == "Fitted DID Event Study"
    assert influence_fig.axes[0].get_title() == "DID Influence Concentration"


def test_post_inference_namespace_exports():
    import causalis.scenarios.did as did
    import causalis.scenarios.did.refutation as ref

    assert hasattr(ref, "run_did_post_inference_diagnostics")
    assert hasattr(ref, "plot_did_influence_concentration")
    assert hasattr(did, "run_did_inference_diagnostics")
    assert hasattr(did, "plot_did_post_inference_event_study")
