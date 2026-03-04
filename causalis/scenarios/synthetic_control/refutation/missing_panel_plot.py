from __future__ import annotations

from math import ceil
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM


def _format_time_tick(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def missing_panel_plot(
    paneldata: PanelDataSCM,
    *,
    show_intervention: bool = True,
    max_xticks: int = 12,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    """
    Plot panel missingness as a unit-by-time heatmap.

    A cell value of 0 means observed and 1 means missing. Missingness is
    inferred from outcome NaNs.

    Parameters
    ----------
    paneldata : PanelDataSCM
        Validated long-format panel contract.
    show_intervention : bool, default True
        If True, draw the intervention boundary as a vertical dashed line.
    max_xticks : int, default 12
        Maximum number of x-axis tick labels shown.
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
    if int(max_xticks) <= 0:
        raise ValueError("max_xticks must be a positive integer.")

    unit_col = paneldata.unit_col
    time_col = paneldata.time_col
    outcome_col = paneldata.y

    df = paneldata.df_analysis().copy()
    df = df.dropna(subset=[time_col])
    if df.empty:
        raise ValueError("No analysis rows available for plotting.")

    missing = df[outcome_col].isna()
    df["_missing"] = missing.astype(float)

    collapsed = df.groupby([unit_col, time_col], as_index=False, sort=True)["_missing"].mean()
    pivot = collapsed.pivot(index=unit_col, columns=time_col, values="_missing")

    times = sorted(pd.Index(df[time_col].unique()).tolist())
    if not times:
        raise ValueError("No time points available for plotting.")

    preferred_units = [paneldata.treated_unit]
    preferred_units.extend([unit for unit in paneldata.donor_pool() if unit != paneldata.treated_unit])
    units_present = pd.Index(df[unit_col].unique()).tolist()
    units = [unit for unit in preferred_units if unit in units_present]
    units.extend([unit for unit in units_present if unit not in units])
    if not units:
        raise ValueError("No units available for plotting.")

    missing_grid = pivot.reindex(index=units, columns=times).fillna(1.0)
    matrix = np.clip(missing_grid.to_numpy(dtype=float), 0.0, 1.0)

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

        cmap = mpl.colors.ListedColormap(["#2A9D8F", "#E76F51"])
        norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        if show_intervention:
            boundary = float(pd.Index(times).searchsorted(paneldata.treatment_start, side="left")) - 0.5
            if -0.5 <= boundary <= (len(times) - 0.5):
                ax.axvline(
                    boundary,
                    linestyle="--",
                    linewidth=1.5,
                    color="0.25",
                    label="Intervention",
                )

        ax.set_title("Panel Missingness by Unit and Time")
        ax.set_xlabel(str(time_col))
        ax.set_ylabel(str(unit_col))

        ax.set_yticks(np.arange(len(units)))
        ax.set_yticklabels([str(unit) for unit in units])

        tick_step = max(1, ceil(len(times) / int(max_xticks)))
        xtick_positions = np.arange(0, len(times), tick_step, dtype=int)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([_format_time_tick(times[i]) for i in xtick_positions])

        ax.set_xticks(np.arange(-0.5, len(times), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(units), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.046, ticks=[0, 1])
        colorbar.ax.set_yticklabels(["Observed", "Missing"])

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc="upper right")
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


__all__ = ["missing_panel_plot"]
