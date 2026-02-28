import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt

from causalis.dgp import generate_scm_data
from causalis.scenarios.synthetic_control import outcome_panel_plot


def test_outcome_panel_plot_basic_timeseries_view():
    panel = generate_scm_data(
        n_donors=6,
        n_pre_periods=12,
        n_post_periods=6,
        random_state=12,
    )

    fig = outcome_panel_plot(panel)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.lines]

    assert isinstance(fig, mpl_figure.Figure)
    assert not plt.fignum_exists(fig.number)
    assert ax.get_title() == "Outcome Time Series by Unit"
    assert ax.get_xlabel() == panel.time_col
    assert ax.get_ylabel() == panel.y
    assert any(str(label).startswith("Treated: ") for label in labels)
    assert "Donor mean" in labels
    assert "Intervention" in labels


def test_outcome_panel_plot_limits_number_of_donor_lines():
    panel = generate_scm_data(
        n_donors=8,
        n_pre_periods=10,
        n_post_periods=4,
        random_state=3,
    )

    fig = outcome_panel_plot(
        panel,
        donor_max_lines=2,
        show_donor_mean=False,
    )
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.lines]

    assert f"Donors (n=2/{len(panel.donor_pool())})" in labels
    assert "Donor mean" not in labels
    assert "Intervention" in labels


def test_synthetic_control_namespace_exposes_refutation_plot():
    import causalis.scenarios.synthetic_control as scm

    assert hasattr(scm, "outcome_panel_plot")
