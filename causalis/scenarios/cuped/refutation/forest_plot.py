from __future__ import annotations

from statistics import NormalDist
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.causal_diagnostic_data import CUPEDDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate


def _normal_critical(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def cuped_forest_plot(
    estimate_with_cuped: CausalEstimate,
    estimate_without_cuped: Optional[CausalEstimate] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.5, 3.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    label_with_cuped: str = "With CUPED",
    label_without_cuped: str = "Without CUPED",
    color_with_cuped: str = "C0",
    color_without_cuped: str = "C1",
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    """
    Forest plot of absolute estimates and CIs for CUPED vs non-CUPED.

    Parameters
    ----------
    estimate_with_cuped : CausalEstimate
        Effect estimated with CUPED adjustment.
    estimate_without_cuped : CausalEstimate, optional
        Effect estimated without CUPED adjustment. If omitted, the function
        uses ``estimate_with_cuped.diagnostic_data.ate_naive`` and
        ``estimate_with_cuped.diagnostic_data.se_naive`` to build a normal-approx CI.
    """
    if estimate_without_cuped is None:
        diag = estimate_with_cuped.diagnostic_data
        if not isinstance(diag, CUPEDDiagnosticData):
            raise ValueError(
                "estimate_with_cuped.diagnostic_data must be CUPEDDiagnosticData "
                "when estimate_without_cuped is not provided."
            )
        z = _normal_critical(estimate_with_cuped.alpha)
        no_cuped_estimate = float(diag.ate_naive)
        no_cuped_ci_low = float(no_cuped_estimate - z * float(diag.se_naive))
        no_cuped_ci_high = float(no_cuped_estimate + z * float(diag.se_naive))
    else:
        no_cuped_estimate = float(estimate_without_cuped.value)
        no_cuped_ci_low = float(estimate_without_cuped.ci_lower_absolute)
        no_cuped_ci_high = float(estimate_without_cuped.ci_upper_absolute)

    cuped_estimate = float(estimate_with_cuped.value)
    cuped_ci_low = float(estimate_with_cuped.ci_lower_absolute)
    cuped_ci_high = float(estimate_with_cuped.ci_upper_absolute)

    labels = [label_without_cuped, label_with_cuped]
    estimates = np.asarray([no_cuped_estimate, cuped_estimate], dtype=float)
    ci_low = np.asarray([no_cuped_ci_low, cuped_ci_low], dtype=float)
    ci_high = np.asarray([no_cuped_ci_high, cuped_ci_high], dtype=float)
    xerr = np.vstack([estimates - ci_low, ci_high - estimates])
    y = np.asarray([0, 1], dtype=float)
    colors = [color_without_cuped, color_with_cuped]

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
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
            try:
                fig.set_dpi(dpi)
            except Exception:
                pass

        for idx in range(2):
            ax.errorbar(
                x=estimates[idx],
                y=y[idx],
                xerr=xerr[:, idx].reshape(2, 1),
                fmt="o",
                color=colors[idx],
                ecolor=colors[idx],
                markersize=6,
                elinewidth=2,
                capsize=4,
            )

        ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Estimate (absolute scale)")
        ax.set_title("Estimate and Absolute CI: CUPED vs Non-CUPED")
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        xmin = float(np.min(ci_low))
        xmax = float(np.max(ci_high))
        span = xmax - xmin
        pad = 0.08 * span if span > 0 else 1.0
        ax.set_xlim(xmin - pad, xmax + pad)
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

        if not ax_provided:
            plt.close(fig)

    return fig
