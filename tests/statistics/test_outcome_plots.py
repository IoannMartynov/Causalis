import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mpl_figure
import pandas as pd

from causalis.dgp.causaldata import CausalData
from causalis.dgp.multicausaldata import MultiCausalData
from causalis.shared import outcome_plot_boxplot, outcome_plot_dist


def test_outcome_plot_dist_binary_causaldata_logic():
    df = pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1],
            "outcome": [0, 1, 1, 1],
            "x": [10, 11, 12, 13],
        }
    )
    data = CausalData.from_df(
        df,
        treatment="treatment",
        outcome="outcome",
        confounders=["x"],
    )

    fig = outcome_plot_dist(data)
    ax = fig.axes[0]

    assert isinstance(fig, mpl_figure.Figure)
    assert len(ax.patches) == 2
    assert ax.get_title() == "Outcome rate by treatment"
    assert ax.get_xlabel() == "treatment"
    assert ax.get_ylabel().startswith("Pr(outcome=")


def test_outcome_plot_dist_supports_multicausaldata_default_treatment():
    df = pd.DataFrame(
        {
            "y": [0, 1, 1, 0, 1, 0],
            "t0": [1, 0, 0, 1, 0, 0],
            "t1": [0, 1, 0, 0, 1, 0],
            "t2": [0, 0, 1, 0, 0, 1],
            "x": [10, 11, 12, 13, 14, 15],
        }
    )
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["t0", "t1", "t2"],
        confounders=["x"],
        control_treatment="t0",
    )

    fig = outcome_plot_dist(data)
    ax = fig.axes[0]

    assert isinstance(fig, mpl_figure.Figure)
    assert len(ax.patches) == 3
    assert ax.get_title() == "Outcome rate by treatment"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["t0", "t1", "t2"]
    assert ax.get_xlabel() == "treatment"
    assert ax.get_ylabel().startswith("Pr(y=")


def test_outcome_plot_boxplot_uses_current_matplotlib_label_keyword():
    df = pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1],
            "outcome": [1.0, 2.0, 3.0, 4.0],
            "x": [10, 11, 12, 13],
        }
    )
    data = CausalData.from_df(
        df,
        treatment="treatment",
        outcome="outcome",
        confounders=["x"],
    )

    fig = outcome_plot_boxplot(data)
    ax = fig.axes[0]

    assert isinstance(fig, mpl_figure.Figure)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0", "1"]
