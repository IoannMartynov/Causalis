from __future__ import annotations

from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from causalis.data_contracts import PanelDataDID
from causalis.data_contracts.panel_data_did import ComparisonGroup

from .diagnostics import BasePeriod, did_support_table, raw_did_event_study_table


def _ensure_panel_data(data: PanelDataDID) -> None:
    if not isinstance(data, PanelDataDID):
        raise TypeError("data must be a PanelDataDID instance.")


def _label(value: object) -> str:
    if isinstance(value, pd.Period):
        return str(value)
    return "" if pd.isna(value) else str(value)


def _aggregate_raw_event_study(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for event_time, subset in table.groupby("event_time", sort=True, observed=True):
        weights = pd.to_numeric(subset["n_treated"], errors="coerce").to_numpy(dtype=float)
        estimates = pd.to_numeric(subset["raw_did"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(subset["se"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(weights) & np.isfinite(estimates) & (weights > 0.0)
        if not bool(mask.any()):
            continue
        weights = weights[mask]
        estimates = estimates[mask]
        se = se[mask]
        weights = weights / float(np.sum(weights))
        aggregate_se = (
            float(np.sqrt(np.sum((weights * se) ** 2)))
            if np.isfinite(se).all()
            else float("nan")
        )
        rows.append(
            {
                "event_time": int(event_time),
                "raw_did": float(np.sum(weights * estimates)),
                "se": aggregate_se,
                "n_cells": int(mask.sum()),
                "n_treated": int(subset.loc[mask, "n_treated"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=["event_time", "raw_did", "se", "n_cells", "n_treated"])


def plot_did_support(
    data: PanelDataDID,
    *,
    control_group: ComparisonGroup = "not_yet_or_never",
    anticipation: int = 0,
    base_period: BasePeriod = "universal",
    include_pre_periods: bool = False,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.05,
) -> plt.Figure:
    """
    Plot Callaway & Sant'Anna support cells, sized by complete comparison-unit count.

    Visualizes which cohort-time cells are estimable given the selected control
    group and base period policy. Blue circles indicate supported cells, while
    red indicates unsupported cells. Circle size corresponds to the number of
    available control units.

    Parameters
    ----------
    data : PanelDataDID
        The validated panel data object.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        The definition of the comparison group.
    anticipation : int, default 0
        Anticipation periods to exclude.
    base_period : {"universal", "varying"}, default "universal"
        Base period policy.
    include_pre_periods : bool, default False
        Whether to include pre-treatment cells.
    figsize : tuple of float, default (10.0, 5.5)
        The size of the figure.
    dpi : int, default 220
        Resolution of the figure.
    font_scale : float, default 1.05
        Scaling factor for font sizes.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting support plot.

    Examples
    --------
    >>> from causalis.scenarios.did import generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import plot_did_support
    >>> data = generate_did_gamma_26(n_units=200, n_periods=5, seed=42)
    >>> fig = plot_did_support(data)
    >>> # fig.show()
    """

    _ensure_panel_data(data)
    support = did_support_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
    )
    plot_data = support.dropna(subset=["cohort", "time"]).copy()
    if plot_data.empty:
        raise ValueError("No ATT(g,t) support cells are available to plot.")

    times = sorted(plot_data["time"].drop_duplicates().tolist())
    cohorts = sorted(plot_data["cohort"].drop_duplicates().tolist())
    x_lookup = {time: idx for idx, time in enumerate(times)}
    y_lookup = {cohort: idx for idx, cohort in enumerate(cohorts)}
    x = plot_data["time"].map(x_lookup).to_numpy(dtype=float)
    y = plot_data["cohort"].map(y_lookup).to_numpy(dtype=float)
    controls = pd.to_numeric(plot_data["n_control_complete"], errors="coerce").fillna(0.0)
    max_controls = float(controls.max()) if len(controls) else 0.0
    if max_controls > 0.0:
        sizes = 80.0 + 260.0 * controls.to_numpy(dtype=float) / max_controls
    else:
        sizes = np.full(len(plot_data), 120.0)
    supported = plot_data["is_supported"].fillna(False).astype(bool).to_numpy()
    colors = np.where(supported, "#1f77b4", "#d62728")
    pre = ~plot_data["is_post_treatment"].fillna(False).astype(bool).to_numpy()

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(
            x[~pre],
            y[~pre],
            s=sizes[~pre],
            c=colors[~pre],
            alpha=0.84,
            edgecolor="0.25",
            linewidth=0.7,
            label="Post cells",
        )
        if bool(pre.any()):
            ax.scatter(
                x[pre],
                y[pre],
                s=sizes[pre],
                c=colors[pre],
                alpha=0.84,
                edgecolor="0.25",
                linewidth=0.7,
                marker="s",
                label="Pre cells",
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#1f77b4",
                markeredgecolor="0.25",
                label="Supported",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#d62728",
                markeredgecolor="0.25",
                label="Unsupported",
            ),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="best")
        ax.set_title("DID Support by Cohort and Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Treatment cohort")
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([_label(t) for t in times], rotation=45, ha="right")
        ax.set_yticks(range(len(cohorts)))
        ax.set_yticklabels([_label(c) for c in cohorts])
        ax.grid(True, linewidth=0.45, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

    plt.close(fig)
    return fig


def plot_raw_did_event_study(
    data: PanelDataDID,
    *,
    control_group: ComparisonGroup = "not_yet_or_never",
    anticipation: int = 0,
    base_period: BasePeriod = "varying",
    include_pre_periods: bool = True,
    figsize: Tuple[float, float] = (9.0, 5.2),
    dpi: int = 220,
    font_scale: float = 1.05,
) -> plt.Figure:
    """
    Plot an unadjusted, pre-fit DID event-study diagnostic.

    Aggregates raw cohort-time differences into a single event-study plot. This
    is useful for a quick visual check of parallel trends and the magnitude of
    the unadjusted treatment effect.

    Parameters
    ----------
    data : PanelDataDID
        The validated panel data object.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        The definition of the comparison group.
    anticipation : int, default 0
        Anticipation periods to exclude.
    base_period : {"varying", "universal"}, default "varying"
        Base period policy.
    include_pre_periods : bool, default True
        Whether to include pre-treatment periods.
    figsize : tuple of float, default (9.0, 5.2)
        The size of the figure.
    dpi : int, default 220
        Resolution of the figure.
    font_scale : float, default 1.05
        Scaling factor for font sizes.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting event-study plot.

    Examples
    --------
    >>> from causalis.scenarios.did import generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import plot_raw_did_event_study
    >>> data = generate_did_gamma_26(n_units=200, n_periods=8, seed=42)
    >>> fig = plot_raw_did_event_study(data)
    >>> # fig.show()
    """

    _ensure_panel_data(data)
    table = raw_did_event_study_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
    )
    if table.empty:
        raise ValueError("No raw DID event-study cells are available to plot.")
    event = _aggregate_raw_event_study(table)
    if event.empty:
        raise ValueError("No finite raw DID event-study values are available to plot.")

    x = event["event_time"].to_numpy(dtype=float)
    y = event["raw_did"].to_numpy(dtype=float)
    se = event["se"].to_numpy(dtype=float)
    yerr = np.where(np.isfinite(se), 1.96 * se, 0.0)

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            linewidth=2.0,
            markersize=6.0,
            capsize=3.0,
            color="#1f77b4",
            ecolor="0.45",
        )
        ax.axhline(0.0, color="0.35", linewidth=1.2, linestyle=":")
        ax.axvline(-0.5, color="0.25", linewidth=1.2, linestyle="--")
        ax.set_title("Raw DID Event Study")
        ax.set_xlabel("Event time")
        ax.set_ylabel("Unadjusted DID")
        ax.set_xticks(sorted(event["event_time"].tolist()))
        ax.grid(True, linewidth=0.45, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

    plt.close(fig)
    return fig


__all__ = [
    "plot_did_support",
    "plot_raw_did_event_study",
]
