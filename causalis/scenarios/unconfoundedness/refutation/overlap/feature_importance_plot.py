"""Native feature-importance plots for IRM nuisance learners."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.causal_estimate import CausalEstimate


def _resolve_feature_importance(estimate: CausalEstimate) -> dict[str, Any]:
    """Return feature-importance diagnostics from an estimate."""
    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. Fit IRM with "
            "store_diagnostics=True and call estimate() first."
        )

    feature_importance = getattr(diagnostic_data, "feature_importance", None)
    if not isinstance(feature_importance, dict):
        raise ValueError(
            "Feature importance was not collected. Fit IRM with "
            "store_diagnostics=True and call estimate()."
        )

    nuisances = feature_importance.get("nuisances", {})
    if not isinstance(nuisances, dict) or not any(
        isinstance(payload, dict) and payload.get("available", False)
        for payload in nuisances.values()
    ):
        raise ValueError(
            "No native feature importance was available from the fitted learners. "
            "Use learners exposing feature_importances_, coef_, or CatBoost "
            "get_feature_importance()."
        )

    return feature_importance


def _default_figsize(top_k: int) -> Tuple[float, float]:
    """Choose a size for three compact horizontal-bar panels."""
    return 15.0, max(4.8, 1.8 + 0.32 * max(int(top_k), 1))


def _plot_one_panel(
    ax: plt.Axes,
    *,
    payload: Optional[dict[str, Any]],
    feature_names: list[str],
    top_k: int,
    title: str,
) -> None:
    """Render one nuisance learner's native importance summary."""
    if not isinstance(payload, dict) or not payload.get("available", False):
        ax.text(
            0.5,
            0.5,
            "No native\nimportance",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    mean = np.asarray(payload.get("mean"), dtype=float).ravel()
    std_raw = payload.get("std")
    std = (
        np.zeros_like(mean)
        if std_raw is None
        else np.asarray(std_raw, dtype=float).ravel()
    )
    if mean.size == 0 or mean.size != len(feature_names):
        raise ValueError(
            "feature_importance payload has incompatible feature dimensions."
        )
    if std.size != mean.size:
        std = np.zeros_like(mean)

    finite = np.isfinite(mean)
    if not np.any(finite):
        raise ValueError(
            "feature_importance mean values must include at least one finite value."
        )

    values = np.where(finite, np.maximum(mean, 0.0), 0.0)
    errors = np.where(np.isfinite(std), np.maximum(std, 0.0), 0.0)
    k = min(max(int(top_k), 1), values.size)
    order = np.argsort(-values, kind="mergesort")[:k]
    order = order[np.argsort(values[order], kind="mergesort")]

    y = np.arange(order.size, dtype=float)
    xerr = errors[order] if np.any(errors[order] > 0.0) else None
    ax.barh(
        y,
        values[order],
        xerr=xerr,
        color="C0",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("Normalized importance")
    ax.set_title(f"{title} (folds={int(payload.get('n_folds', 0))})")
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_feature_importance(
    estimate: CausalEstimate,
    *,
    top_k: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    r"""
    Plot native feature importances collected from IRM nuisance learners.

    Parameters
    ----------
    estimate : CausalEstimate
        Effect estimate with ``diagnostic_data.feature_importance`` collected by
        fitting IRM with ``store_diagnostics=True``.
    top_k : int, default 20
        Number of top features to show per nuisance learner.
    figsize : tuple, optional
        Figure size. Defaults to an auto-scaled height based on ``top_k``.
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
    if int(top_k) <= 0:
        raise ValueError("top_k must be a positive integer.")

    feature_importance = _resolve_feature_importance(estimate)
    feature_names = [str(name) for name in feature_importance.get("feature_names", [])]
    if not feature_names:
        n_features = int(feature_importance.get("n_features", 0))
        feature_names = [f"x{j + 1}" for j in range(n_features)]
    if not feature_names:
        raise ValueError("feature_importance payload must include feature names.")

    nuisances = feature_importance.get("nuisances", {})
    panels = [
        ("m", "Propensity m(X)"),
        ("g0", "Outcome g0(X)"),
        ("g1", "Outcome g1(X)"),
    ]

    rc = {
        "font.size": 10.5 * font_scale,
        "axes.titlesize": 12.5 * font_scale,
        "axes.labelsize": 10.5 * font_scale,
        "xtick.labelsize": 9.5 * font_scale,
        "ytick.labelsize": 9.0 * font_scale,
    }

    with mpl.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize or _default_figsize(int(top_k)),
            dpi=dpi,
            sharex=False,
        )
        for ax, (key, title) in zip(np.asarray(axes).ravel(), panels):
            payload = nuisances.get(key) if isinstance(nuisances, dict) else None
            _plot_one_panel(
                ax,
                payload=payload,
                feature_names=feature_names,
                top_k=int(top_k),
                title=title,
            )

        fig.suptitle("Native Feature Importance Diagnostics", y=0.995)
        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            save_dpi_eff = save_dpi or (
                300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi
            )
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


__all__ = ["plot_feature_importance"]
