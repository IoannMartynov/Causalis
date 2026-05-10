r"""Reliability diagram for propensity calibration diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData

from .overlap_validation import _calibration_report, _validate_estimate_matches_data


def _logit(x: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logit transform to probabilities."""
    clipped = np.clip(np.asarray(x, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic transform."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    pos = z_arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    exp_z = np.exp(z_arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _resolve_calibration_payload(
    estimate: CausalEstimate,
    data: Optional[CausalData],
    *,
    n_bins: int,
) -> Dict[str, Any]:
    """Extract and validate the calibration inputs used by reliability plots."""
    if data is not None:
        _validate_estimate_matches_data(data=data, estimate=estimate)

    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    m_hat = getattr(diagnostic_data, "m_hat", None)
    m_hat_raw = getattr(diagnostic_data, "m_hat_raw", None)
    m_source = m_hat_raw if m_hat_raw is not None else m_hat
    if m_source is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat`.")

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        if data is None:
            raise ValueError(
                "diagnostic_data must include `d`, or pass `data` for fallback."
            )
        d_raw = data.get_df()[str(data.treatment_name)].to_numpy(dtype=float)

    p = np.asarray(m_source, dtype=float).ravel()
    d = (np.asarray(d_raw, dtype=float).ravel() > 0.5).astype(int)
    if p.size != d.size:
        raise ValueError("diagnostic_data.m_hat and treatment vector must have matching length.")

    finite = np.isfinite(p) & np.isfinite(d)
    p = p[finite]
    d = d[finite]
    if p.size == 0:
        raise ValueError("No finite propensity/treatment pairs available for reliability plotting.")

    calibration = _calibration_report(p, d, n_bins=int(n_bins))
    table_raw = calibration.get("reliability_table", None)
    if table_raw is None:
        raise ValueError(
            "Calibration payload must include `reliability_table` with "
            "`count`, `mean_p`, and `frac_pos`."
        )
    table = table_raw.copy() if isinstance(table_raw, pd.DataFrame) else pd.DataFrame(table_raw)
    required = {"count", "mean_p", "frac_pos"}
    missing = required.difference(set(table.columns))
    if missing:
        raise ValueError(
            "reliability_table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    recal = calibration.get("recalibration", {}) or {}
    slope = float(recal.get("slope", np.nan))
    intercept = float(recal.get("intercept", np.nan))

    return {
        "table": table,
        "ece": float(calibration.get("ece", np.nan)),
        "slope": slope,
        "intercept": intercept,
        "n_bins": int(calibration.get("n_bins", int(n_bins))),
    }


def plot_propensity_reliability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    n_bins: int = 10,
    show_recalibration: bool = True,
    annotate_metrics: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7.2, 6.2),
    dpi: int = 220,
    font_scale: float = 1.10,
    point_color: Optional[Any] = None,
    diagonal_color: Any = "0.35",
    recalibration_color: Any = "C1",
    min_marker_size: float = 35.0,
    marker_size_scale: float = 250.0,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    r"""
    Plot a propensity calibration reliability diagram.

    The reliability plot groups observations into propensity bins and compares
    predicted treatment probability with observed treatment frequency. In a
    well-calibrated propensity model, points should stay near the diagonal:

    .. math::

        \mathbb{E}[D \mid m(X) \in b] \approx \mathbb{E}[m(X) \mid m(X) \in b].

    Parameters
    ----------
    estimate : CausalEstimate
        Estimate with diagnostic data (`m_hat`; optionally `m_hat_raw`, `d`).
    data : CausalData, optional
        Optional fallback source for treatment `d` when not stored in diagnostic data.
    n_bins : int, default 10
        Number of calibration bins used to build the reliability table.
    show_recalibration : bool, default True
        Overlay logistic recalibration curve
        `sigmoid(alpha + beta * logit(p))` when parameters are available.
    annotate_metrics : bool, default True
        Annotate ECE and logistic recalibration parameters on the figure.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    figsize : tuple, default (7.2, 6.2)
        Figure size.
    dpi : int, default 220
        Dots per inch.
    font_scale : float, default 1.10
        Font scaling factor.
    point_color : color, optional
        Marker color for binned reliability points.
    diagonal_color : color, default "0.35"
        Color for the perfect calibration diagonal.
    recalibration_color : color, default "C1"
        Color for the logistic recalibration curve.
    min_marker_size : float, default 35.0
        Base marker area for non-empty bins.
    marker_size_scale : float, default 250.0
        Additional marker area scaled by bin count share.
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

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    >>> from causalis.dgp import obs_linear_26_dataset
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> data = obs_linear_26_dataset(
    ...     n=1000,
    ...     seed=3141,
    ...     include_oracle=False,
    ...     return_causal_data=True,
    ... )
    >>> irm = IRM(
    ...     data=data,
    ...     ml_g=RandomForestRegressor(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     ml_m=RandomForestClassifier(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     n_folds=3,
    ...     random_state=3141,
    ... )
    >>> estimate = irm.fit().estimate(score="ATE")
    >>> fig = plot_propensity_reliability(estimate, data=data)  # doctest: +SKIP
    """

    payload = _resolve_calibration_payload(estimate=estimate, data=data, n_bins=int(n_bins))
    table = payload["table"]
    ece = float(payload["ece"])
    slope = float(payload["slope"])
    intercept = float(payload["intercept"])
    n_bins = int(payload["n_bins"])

    mean_p = pd.to_numeric(table["mean_p"], errors="coerce").to_numpy(dtype=float)
    frac_pos = pd.to_numeric(table["frac_pos"], errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(table["count"], errors="coerce").to_numpy(dtype=float)
    counts = np.where(np.isfinite(counts), counts, 0.0)

    mask = np.isfinite(mean_p) & np.isfinite(frac_pos) & (counts > 0)
    if not np.any(mask):
        raise ValueError(
            "No non-empty finite bins available in reliability_table to plot."
        )

    x = np.clip(mean_p[mask], 0.0, 1.0)
    y = np.clip(frac_pos[mask], 0.0, 1.0)
    c = counts[mask]
    order = np.argsort(x)
    x, y, c = x[order], y[order], c[order]

    c_max = float(np.max(c)) if np.max(c) > 0 else 1.0
    sizes = min_marker_size + marker_size_scale * (c / c_max)

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "legend.fontsize": 10 * font_scale,
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

        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.4,
            color=diagonal_color,
            label="Perfect calibration",
            zorder=1,
        )

        used_point_color = point_color or mpl.rcParams["axes.prop_cycle"].by_key().get(
            "color",
            ["C0"],
        )[0]
        ax.scatter(
            x,
            y,
            s=sizes,
            color=used_point_color,
            alpha=0.80,
            linewidth=0.6,
            edgecolor="white",
            label="Bin means (size = count)",
            zorder=3,
        )

        if show_recalibration and np.isfinite(intercept) and np.isfinite(slope):
            grid = np.linspace(1e-4, 1.0 - 1e-4, 500)
            curve = _sigmoid(intercept + slope * _logit(grid))
            ax.plot(
                grid,
                curve,
                linewidth=2.1,
                color=recalibration_color,
                label=r"Logit recalibration: $\sigma(\alpha + \beta \,\mathrm{logit}(p))$",
                zorder=2,
            )

        if annotate_metrics:
            ece_text = "nan" if not np.isfinite(ece) else f"{ece:.3f}"
            slope_text = "nan" if not np.isfinite(slope) else f"{slope:.3f}"
            intercept_text = "nan" if not np.isfinite(intercept) else f"{intercept:.3f}"
            note = (
                f"ECE = {ece_text}\n"
                f"Slope (beta) = {slope_text}\n"
                f"Intercept (alpha) = {intercept_text}\n"
                f"Bins = {n_bins}"
            )
            ax.text(
                0.02,
                0.98,
                note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.80",
                    "alpha": 0.90,
                    "boxstyle": "round,pad=0.35",
                },
            )

        ax.set_xlabel(r"$\overline{m}_b = \mathbb{E}[m_i \mid i \in b]$")
        ax.set_ylabel(r"$\overline{D}_b = \mathbb{E}[D_i \mid i \in b]$")
        ax.set_title("Propensity Calibration (Reliability Diagram)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            _dpi = save_dpi or (
                300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi
            )
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


__all__ = ["plot_propensity_reliability"]
