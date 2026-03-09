from __future__ import annotations

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM


def _to_plot_time(value: object) -> object:
    if isinstance(value, pd.Period):
        return value.to_timestamp()
    return value


def _to_plot_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    return index


def outcome_panel_plot(
    paneldata: PanelDataSCM,
    *,
    show_donor_units: bool = True,
    donor_max_lines: Optional[int] = 20,
    show_donor_mean: bool = True,
    donor_alpha: float = 0.35,
    donor_linewidth: float = 1.2,
    shade_post_period: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    """
    Plot SCM panel outcomes over time.

    The figure shows treated-unit outcomes over time, optional donor-unit paths,
    optional donor mean, and the intervention boundary.

    Parameters
    ----------
    paneldata : PanelDataSCM
        Validated long-format panel contract.
    show_donor_units : bool, default True
        If True, draw donor trajectories.
    donor_max_lines : int or None, default 20
        Maximum number of donor-unit lines to draw. ``None`` draws all donors.
    show_donor_mean : bool, default True
        If True, draw the donor-pool mean outcome path.
    donor_alpha : float, default 0.35
        Opacity for donor-unit lines.
    donor_linewidth : float, default 1.2
        Line width for donor-unit lines.
    shade_post_period : bool, default True
        If True, lightly shade the post-treatment region.
    figsize : tuple, default (10.0, 5.5)
        Figure size in inches.
    dpi : int, default 220
        Dots per inch.
    font_scale : float, default 1.10
        Font scaling factor.
    save : str, optional
        Optional path to save the figure.
    save_dpi : int, optional
        DPI for saved raster outputs.
    transparent : bool, default False
        Whether to save with a transparent background.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if not isinstance(paneldata, PanelDataSCM):
        raise TypeError("paneldata must be a PanelDataSCM instance.")
    if donor_max_lines is not None and int(donor_max_lines) < 0:
        raise ValueError("donor_max_lines must be None or a non-negative integer.")

    unit_col = paneldata.unit_col
    time_col = paneldata.time_col
    outcome_col = paneldata.y

    df = paneldata.df_analysis().copy()
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    if df.empty:
        raise ValueError("No analysis rows available for plotting.")

    # Duplicate (unit, time) rows are averaged for a stable visualization.
    collapsed = (
        df.groupby([time_col, unit_col], as_index=False, sort=True)[outcome_col]
        .mean()
    )
    pivot = collapsed.pivot(index=time_col, columns=unit_col, values=outcome_col).sort_index()
    if paneldata.treated_unit not in pivot.columns:
        raise ValueError("treated_unit is not available in analysis data.")

    donors_all = [unit for unit in paneldata.donor_pool() if unit in pivot.columns]
    donors_shown = list(donors_all)
    if donor_max_lines is not None:
        donors_shown = donors_shown[: int(donor_max_lines)]

    treated_series = pivot[paneldata.treated_unit]
    donor_mean = pivot[donors_all].mean(axis=1, skipna=True) if donors_all else None
    x_index_plot = _to_plot_index(pivot.index)
    treatment_start_plot = _to_plot_time(paneldata.treatment_start)

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])

        if shade_post_period and len(pivot.index) > 0:
            x_max = x_index_plot.max()
            if treatment_start_plot <= x_max:
                ax.axvspan(
                    treatment_start_plot,
                    x_max,
                    color="0.75",
                    alpha=0.15,
                    linewidth=0.0,
                    zorder=0,
                )

        if show_donor_units and donors_shown:
            donor_label = f"Donors (n={len(donors_shown)}/{len(donors_all)})"
            for idx, donor in enumerate(donors_shown):
                donor_series = pivot[donor]
                ax.plot(
                    x_index_plot,
                    donor_series.values,
                    color="0.60",
                    linewidth=donor_linewidth,
                    alpha=donor_alpha,
                    label=donor_label if idx == 0 else None,
                    zorder=1,
                )

        if show_donor_mean and donor_mean is not None:
            ax.plot(
                x_index_plot,
                donor_mean.values,
                color=cycle[1 % len(cycle)],
                linewidth=2.2,
                label="Donor mean",
                zorder=2,
            )

        ax.plot(
            x_index_plot,
            treated_series.values,
            color=cycle[0],
            linewidth=2.6,
            label=f"Treated: {paneldata.treated_unit}",
            zorder=3,
        )
        ax.axvline(
            treatment_start_plot,
            linestyle="--",
            linewidth=1.7,
            color="0.25",
            label="Intervention",
            zorder=4,
        )

        ax.set_title("Outcome Time Series by Unit")
        ax.set_xlabel(str(time_col))
        ax.set_ylabel(str(outcome_col))
        ax.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            _dpi = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
            fig.savefig(
                save,
                dpi=_dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white",
            )

    plt.close(fig)
    return fig


__all__ = ["outcome_panel_plot"]
