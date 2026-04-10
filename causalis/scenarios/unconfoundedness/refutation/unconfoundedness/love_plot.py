"""Love plot for covariate balance before and after weighting."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData

from .unconfoundedness_validation import run_unconfoundedness_diagnostics


def _default_figsize(p: int) -> Tuple[float, float]:
    """Choose a figure size that keeps all confounders visible."""
    width = 10.0
    height = max(4.8, 1.8 + 0.28 * max(int(p), 1))
    return width, height


def _prepare_balance_table(
    before: pd.Series,
    after: pd.Series,
) -> pd.DataFrame:
    """Align before/after SMD series and order worst imbalances first."""
    table = pd.DataFrame({"before": before.astype(float), "after": after.astype(float)})
    table["sort_key"] = table[["before", "after"]].max(axis=1, skipna=True)
    table = table.sort_values(
        "sort_key",
        ascending=False,
        na_position="last",
        kind="mergesort",
    )
    return table


def _coerce_plot_values(
    before: np.ndarray,
    after: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Convert non-finite SMDs into plottable coordinates and x-limits."""
    finite_values = np.concatenate(
        [
            before[np.isfinite(before)],
            after[np.isfinite(after)],
        ]
    )
    base_max = float(np.max(finite_values)) if finite_values.size else 0.0
    base_max = max(base_max, float(threshold), 0.10)

    has_inf = bool(np.any(np.isinf(before)) or np.any(np.isinf(after)))
    inf_cap = base_max * 1.08 if has_inf else base_max
    inf_cap = max(inf_cap, 0.12)

    before_plot = before.copy()
    after_plot = after.copy()
    before_plot[np.isposinf(before_plot)] = inf_cap
    after_plot[np.isposinf(after_plot)] = inf_cap

    x_limit = inf_cap * (1.05 if has_inf else 1.10)
    x_limit = max(x_limit, float(threshold) * 1.25, 0.12)
    return before_plot, after_plot, x_limit, has_inf


def love_plot(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    threshold: float = 0.10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    """
    Plot covariate balance before and after weighting implied by an estimate.

    Parameters
    ----------
    data : CausalData
        Dataset used to fit the estimator.
    estimate : CausalEstimate
        Effect estimate with diagnostic data needed for balance diagnostics.
    threshold : float, default 0.10
        Reference threshold for absolute standardized mean differences.
    figsize : tuple, optional
        Figure size. Defaults to an auto-scaled height based on confounder count.
    dpi : int, default 220
        Dots per inch.
    font_scale : float, default 1.10
        Font scaling factor.
    save : str, optional
        Path to save the figure.
    save_dpi : int, optional
        DPI for saving.
    transparent : bool, default False
        Whether to save with transparency.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    report = run_unconfoundedness_diagnostics(
        data=data,
        estimate=estimate,
        threshold=float(threshold),
        return_summary=False,
    )
    before = report["balance"]["smd_unweighted"]
    after = report["balance"]["smd"]
    balance_table = _prepare_balance_table(before=before, after=after)

    names = [str(name) for name in balance_table.index.tolist()]
    before_values = balance_table["before"].to_numpy(dtype=float)
    after_values = balance_table["after"].to_numpy(dtype=float)
    before_plot, after_plot, x_limit, has_inf = _coerce_plot_values(
        before_values,
        after_values,
        threshold=float(threshold),
    )
    y = np.arange(len(names), dtype=float)

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 9.5 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 9.5 * font_scale,
    }

    with mpl.rc_context(rc):
        fig, ax = plt.subplots(
            figsize=figsize or _default_figsize(len(names)),
            dpi=dpi,
        )

        for yi, b_val, a_val in zip(y, before_plot, after_plot):
            if np.isfinite(b_val) and np.isfinite(a_val):
                ax.plot([b_val, a_val], [yi, yi], color="0.80", linewidth=1.0, zorder=1)

        before_mask = np.isfinite(before_plot)
        after_mask = np.isfinite(after_plot)

        ax.scatter(
            before_plot[before_mask],
            y[before_mask],
            s=42.0,
            marker="o",
            color="C1",
            edgecolors="white",
            linewidths=0.6,
            label="Before (unweighted)",
            zorder=3,
        )
        ax.scatter(
            after_plot[after_mask],
            y[after_mask],
            s=48.0,
            marker="D",
            color="C0",
            edgecolors="white",
            linewidths=0.6,
            label="After (weighted)",
            zorder=4,
        )

        ax.axvline(
            float(threshold),
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            label=f"Threshold |SMD| = {float(threshold):.2f}",
            zorder=2,
        )

        ax.set_xlim(0.0, x_limit)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("|SMD|")
        ax.set_ylabel("Confounders")
        ax.set_title(f"Covariate Balance Love Plot ({report['params']['score']})")
        ax.grid(True, axis="x", linewidth=0.5, alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")

        if has_inf:
            ax.text(
                0.995,
                0.01,
                "Infinite SMDs clipped at right edge",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="0.40",
                fontsize=9 * font_scale,
            )

        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            save_dpi_eff = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
            fig.savefig(
                save,
                dpi=save_dpi_eff,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white",
            )

    plt.close(fig)
    return fig


__all__ = ["love_plot"]
