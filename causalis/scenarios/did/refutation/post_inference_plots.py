from __future__ import annotations

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts import PanelDataDID
from causalis.data_contracts.panel_did_estimate import CallawaySantAnnaDIDEstimate

from .post_inference import did_influence_table


def _resolve_estimate(
    data_or_estimate: PanelDataDID | CallawaySantAnnaDIDEstimate,
    estimate: Optional[CallawaySantAnnaDIDEstimate],
) -> tuple[Optional[PanelDataDID], CallawaySantAnnaDIDEstimate]:
    if estimate is None:
        if not isinstance(data_or_estimate, CallawaySantAnnaDIDEstimate):
            raise TypeError("estimate must be a CallawaySantAnnaDIDEstimate instance.")
        return None, data_or_estimate
    if not isinstance(data_or_estimate, PanelDataDID):
        raise TypeError("data must be a PanelDataDID instance when estimate is passed.")
    if not isinstance(estimate, CallawaySantAnnaDIDEstimate):
        raise TypeError("estimate must be a CallawaySantAnnaDIDEstimate instance.")
    return data_or_estimate, estimate


def plot_did_post_inference_event_study(
    data_or_estimate: PanelDataDID | CallawaySantAnnaDIDEstimate,
    estimate: Optional[CallawaySantAnnaDIDEstimate] = None,
    *,
    show_simultaneous: bool = True,
    figsize: Tuple[float, float] = (9.0, 5.2),
    dpi: int = 220,
    font_scale: float = 1.05,
) -> plt.Figure:
    r"""Plot the fitted CS event-study estimates with confidence intervals.

    This function visualizes the dynamic effects of treatment over time relative to
    the start of treatment (event time). It displays the event-study aggregation of
    the group-time average treatment effects :math:`ATT(g,t)`.

    Parameters
    ----------
    data_or_estimate : PanelDataDID or CallawaySantAnnaDIDEstimate
        Either the validated panel data or the fitted estimate object. If data is
        passed, the `estimate` parameter must also be provided.
    estimate : CallawaySantAnnaDIDEstimate, optional
        The fitted model results. Required if `data_or_estimate` is a `PanelDataDID`.
    show_simultaneous : bool, default True
        Whether to show the simultaneous confidence bands if available in the estimate.
    figsize : tuple of float, default (9.0, 5.2)
        The size of the figure in inches (width, height).
    dpi : int, default 220
        The resolution of the figure.
    font_scale : float, default 1.05
        Scale factor for font sizes in the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting event-study plot.

    Examples
    --------
    >>> from causalis.scenarios.did.dgp import generate_did_gamma_26
    >>> from causalis.scenarios.did.model import CallawaySantAnnaDID
    >>> from causalis.scenarios.did.refutation import plot_did_post_inference_event_study
    >>> # Fit the model
    >>> data = generate_did_gamma_26(n_treated_units=30, n_control_units=60, seed=1)
    >>> model = CallawaySantAnnaDID().fit(data)
    >>> results = model.estimate(bootstrap_replications=100)
    >>> # Generate the plot
    >>> fig = plot_did_post_inference_event_study(results)
    >>> # fig.show()

    Notes
    -----
    The event-study aggregation at event time :math:`e` is defined as:

    .. math::

        ATT_{event}(e) = \sum_{g} w(g, e) ATT(g, g+e)

    where :math:`w(g, e)` are weights based on the sample size of each group at that event time.
    Pointwise confidence intervals are shown by default. If the model was estimated using
    the multiplier bootstrap, simultaneous confidence bands can also be displayed to
    account for multiple testing across event times.
    """

    _, estimate = _resolve_estimate(data_or_estimate, estimate)
    event = estimate.event_study()
    if event.empty:
        raise ValueError("estimate.event_study() is empty.")

    x = pd.to_numeric(event["event_time"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(event["estimate"], errors="coerce").to_numpy(dtype=float)
    ci_lower = pd.to_numeric(event["ci_lower"], errors="coerce").to_numpy(dtype=float)
    ci_upper = pd.to_numeric(event["ci_upper"], errors="coerce").to_numpy(dtype=float)
    yerr = np.vstack([y - ci_lower, ci_upper - y])

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
            label="Pointwise CI",
        )
        if (
            show_simultaneous
            and {"sim_ci_lower", "sim_ci_upper"}.issubset(event.columns)
        ):
            sim_lower = pd.to_numeric(event["sim_ci_lower"], errors="coerce").to_numpy(dtype=float)
            sim_upper = pd.to_numeric(event["sim_ci_upper"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(sim_lower).any() and np.isfinite(sim_upper).any():
                order = np.argsort(x)
                ax.fill_between(
                    x[order],
                    sim_lower[order],
                    sim_upper[order],
                    color="#1f77b4",
                    alpha=0.16,
                    label="Simultaneous CI",
                )
        ax.axhline(0.0, color="0.35", linewidth=1.2, linestyle=":")
        ax.axvline(-0.5, color="0.25", linewidth=1.2, linestyle="--")
        ax.set_title("Fitted DID Event Study")
        ax.set_xlabel("Event time")
        ax.set_ylabel("ATT")
        ax.set_xticks(sorted(event["event_time"].tolist()))
        ax.grid(True, linewidth=0.45, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

    plt.close(fig)
    return fig


def plot_did_influence_concentration(
    data_or_estimate: PanelDataDID | CallawaySantAnnaDIDEstimate,
    estimate: Optional[CallawaySantAnnaDIDEstimate] = None,
    *,
    top_n: int = 15,
    figsize: Tuple[float, float] = (9.5, 5.2),
    dpi: int = 220,
    font_scale: float = 1.05,
) -> plt.Figure:
    r"""Plot top unit-level influence shares for the simple overall ATT.

    This plot helps identify outlier units that disproportionately affect the
    overall average treatment effect estimate. Large influence shares may indicate
    lack of overlap or extreme outcomes.

    Parameters
    ----------
    data_or_estimate : PanelDataDID or CallawaySantAnnaDIDEstimate
        Either the validated panel data or the fitted estimate object.
    estimate : CallawaySantAnnaDIDEstimate, optional
        The fitted model results. Required if `data_or_estimate` is a `PanelDataDID`.
    top_n : int, default 15
        The number of top influential units to display.
    figsize : tuple of float, default (9.5, 5.2)
        The size of the figure in inches (width, height).
    dpi : int, default 220
        The resolution of the figure.
    font_scale : float, default 1.05
        Scale factor for font sizes in the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting influence concentration bar plot.

    Examples
    --------
    >>> from causalis.scenarios.did.dgp import generate_did_gamma_26
    >>> from causalis.scenarios.did.model import CallawaySantAnnaDID
    >>> from causalis.scenarios.did.refutation import plot_did_influence_concentration
    >>> # Fit the model
    >>> data = generate_did_gamma_26(n_treated_units=30, n_control_units=60, seed=1)
    >>> results = CallawaySantAnnaDID().fit(data).estimate()
    >>> # Generate the plot
    >>> fig = plot_did_influence_concentration(results, top_n=10)
    >>> # fig.show()

    Notes
    -----
    The influence share for unit :math:`i` is calculated based on its contribution
    to the simple aggregated ATT influence function :math:`\psi`.
    The absolute influence share is defined as:

    .. math::

        Share_i = \frac{|\psi_i|}{\sum_{j=1}^n |\psi_j|}

    where :math:`\psi_i` is the value of the influence function for unit :math:`i`
    on the overall ATT estimate :math:`\hat{\theta}`.
    """

    data, estimate = _resolve_estimate(data_or_estimate, estimate)
    influence = (
        did_influence_table(data, estimate, top_n=top_n)
        if data is not None
        else did_influence_table(estimate, top_n=top_n)
    )
    if influence.empty:
        raise ValueError("No influence rows are available to plot.")

    labels = influence[estimate.unit_col].astype(str).tolist()
    shares = pd.to_numeric(
        influence["abs_influence_share"],
        errors="coerce",
    ).to_numpy(dtype=float)

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        x = np.arange(len(influence), dtype=float)
        ax.bar(x, shares, color="#1f77b4", alpha=0.86)
        ax.set_title("DID Influence Concentration")
        ax.set_xlabel("Unit")
        ax.set_ylabel("Absolute influence share")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", linewidth=0.45, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

    plt.close(fig)
    return fig


__all__ = [
    "plot_did_post_inference_event_study",
    "plot_did_influence_concentration",
]
