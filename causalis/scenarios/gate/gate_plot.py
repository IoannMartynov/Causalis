from __future__ import annotations

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.gate_estimate import GateEstimate


def _extract_plot_payload(
    estimate: CausalEstimate | GateEstimate,
    label: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    if isinstance(estimate, GateEstimate):
        labels = [str(group_name) for group_name in estimate.group_names]
        values = np.asarray(estimate.values, dtype=float)
        ci_lower = np.asarray(estimate.ci_lower, dtype=float)
        ci_upper = np.asarray(estimate.ci_upper, dtype=float)
        title = f"{estimate.estimand} Estimates and Confidence Intervals"
        return values, ci_lower, ci_upper, labels, title

    if isinstance(estimate, CausalEstimate):
        labels = [label or str(estimate.estimand)]
        values = np.asarray([estimate.value], dtype=float)
        ci_lower = np.asarray([estimate.ci_lower_absolute], dtype=float)
        ci_upper = np.asarray([estimate.ci_upper_absolute], dtype=float)
        title = f"{estimate.estimand} Estimate and Confidence Interval"
        return values, ci_lower, ci_upper, labels, title

    raise TypeError(
        "estimate must be CausalEstimate or GateEstimate, "
        f"got {type(estimate).__name__}."
    )


def gate_plot(
    estimate: CausalEstimate | GateEstimate,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.0,
    color: str = "C0",
    zero_line_color: str = "0.5",
    label: Optional[str] = None,
    xlabel: str = "Effect",
    title: Optional[str] = None,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    """
    Plot effect estimates with confidence intervals for a single estimate or
    group-level estimates such as GATE/GATET.

    Parameters
    ----------
    estimate : CausalEstimate or GateEstimate
        Estimate object to visualize. ``GateEstimate`` produces one point per
        group regardless of whether the estimand is ``GATE`` or ``GATET``;
        ``CausalEstimate`` produces a single-row forest plot.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    figsize : tuple of float, optional
        Figure size. If omitted, a height is chosen automatically from the
        number of plotted rows.
    dpi : int, default 220
        Figure DPI.
    font_scale : float, default 1.0
        Multiplicative scale for text sizes.
    color : str, default "C0"
        Marker and interval color.
    zero_line_color : str, default "0.5"
        Reference line color for zero effect.
    label : str, optional
        Row label used when plotting a single ``CausalEstimate``.
    xlabel : str, default "Effect"
        X-axis label.
    title : str, optional
        Plot title. Defaults depend on the estimate type.
    save : str, optional
        Optional output path for saving the figure.
    save_dpi : int, optional
        DPI for saved raster output.
    transparent : bool, default False
        Whether to save with a transparent background.
    """
    values, ci_lower, ci_upper, labels, default_title = _extract_plot_payload(
        estimate=estimate,
        label=label,
    )
    n_rows = len(labels)
    resolved_figsize = figsize or (8.5, max(2.8, 0.65 * n_rows + 1.4))
    xerr = np.vstack([values - ci_lower, ci_upper - values])
    y = np.arange(n_rows, dtype=float)[::-1]

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }

    with mpl.rc_context(rc):
        ax_provided = ax is not None
        if not ax_provided:
            fig, ax = plt.subplots(figsize=resolved_figsize, dpi=dpi)
        else:
            fig = ax.figure
            try:
                fig.set_dpi(dpi)
            except Exception:
                pass

        ax.errorbar(
            x=values,
            y=y,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6,
            elinewidth=2,
            capsize=4,
        )

        ax.axvline(0.0, color=zero_line_color, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title or default_title)
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        xmin = float(np.nanmin(ci_lower))
        xmax = float(np.nanmax(ci_upper))
        span = xmax - xmin
        pad = 0.08 * span if np.isfinite(span) and span > 0 else 1.0
        ax.set_xlim(xmin - pad, xmax + pad)
        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            resolved_save_dpi = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
            fig.savefig(
                save,
                dpi=resolved_save_dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white",
            )

        if not ax_provided:
            plt.close(fig)

    return fig


plot_gate_estimate = gate_plot


__all__ = ["gate_plot", "plot_gate_estimate"]
