import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import pandas as pd

from causalis.data_contracts import PanelDataSCM
from causalis.scenarios.synthetic_control import (
    ASCM,
    gap_over_time_plot,
    observed_vs_synthetic_plot,
)


def _make_panel_with_effect(effect: float = 2.5) -> pd.DataFrame:
    rows = []
    for idx, t in enumerate(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"],
        start=1,
    ):
        y_c1 = 10.0 + 0.5 * idx
        y_c2 = 12.0 + 0.2 * idx
        y_c3 = 9.0 + 0.3 * idx
        y_treat = 0.5 * y_c1 + 0.3 * y_c2 + 0.2 * y_c3
        if idx >= 4:
            y_treat += effect

        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": y_treat, "treated_time": 1 if idx >= 4 else 0},
                {"unit_id": "C1", "time_id": t, "y": y_c1, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": y_c2, "treated_time": 0},
                {"unit_id": "C3", "time_id": t, "y": y_c3, "treated_time": 0},
            ]
        )
    return pd.DataFrame(rows)


def _fit_estimate():
    df = _make_panel_with_effect(effect=3.0)
    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        y="y",
        treated_time="treated_time",
        df=df,
    )
    return ASCM(lambda_aug=0.5).fit(data).estimate()


def test_observed_vs_synthetic_plot_from_panel_estimate():
    estimate = _fit_estimate()

    fig = observed_vs_synthetic_plot(estimate)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.lines]

    assert isinstance(fig, mpl_figure.Figure)
    assert not plt.fignum_exists(fig.number)
    assert ax.get_title() == "Observed vs Synthetic"
    assert "Observed (treated)" in labels
    assert "Synthetic (augmented)" in labels
    assert "Intervention" in labels


def test_gap_over_time_plot_from_panel_estimate():
    estimate = _fit_estimate()

    fig = gap_over_time_plot(estimate)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.lines]

    assert isinstance(fig, mpl_figure.Figure)
    assert not plt.fignum_exists(fig.number)
    assert ax.get_title() == "Gap Over Time"
    assert "Gap (augmented)" in labels
    assert "Intervention" in labels


def test_synthetic_control_namespace_exposes_new_diagnostic_plots():
    import causalis.scenarios.synthetic_control as scm

    assert hasattr(scm, "observed_vs_synthetic_plot")
    assert hasattr(scm, "gap_over_time_plot")
