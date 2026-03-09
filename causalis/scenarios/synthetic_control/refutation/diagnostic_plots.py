from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts.panel_estimate import PanelEstimate


def _to_plot_time(value: object) -> object:
    if isinstance(value, pd.Period):
        return value.to_timestamp()
    return value


def _to_plot_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    return index


def _ensure_panel_estimate(estimate: PanelEstimate) -> None:
    if not isinstance(estimate, PanelEstimate):
        raise TypeError("estimate must be a PanelEstimate instance.")


def _extract_placebo_att_distribution(
    estimate: PanelEstimate,
    *,
    source: Literal["augmented", "sc"],
) -> np.ndarray:
    diagnostics = dict(estimate.diagnostics or {})
    key = (
        "att_placebo_att_distribution"
        if source == "augmented"
        else "att_sc_placebo_att_distribution"
    )
    raw = diagnostics.get(key)
    if raw is None and source == "augmented":
        # Newer ASCM diagnostics expose cross-fit fold ATT estimates instead of placebo draws.
        raw = diagnostics.get("average_att_fold_estimates")
        if isinstance(raw, dict):
            raw = list(raw.values())
    if raw is None:
        return np.asarray([], dtype=float)

    values = pd.to_numeric(pd.Series(list(raw)), errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _optional_series_attr(estimate: PanelEstimate, attr_name: str) -> pd.Series | None:
    value = getattr(estimate, attr_name, None)
    return value if isinstance(value, pd.Series) else None


def _resolve_att_value(estimate: PanelEstimate, *, source: Literal["augmented", "sc"]) -> float:
    diagnostics = dict(estimate.diagnostics or {})
    candidates: list[Any] = []
    if source == "augmented":
        candidates.extend(
            [
                getattr(estimate, "att", None),
                diagnostics.get("att"),
                diagnostics.get("average_att_estimate"),
            ]
        )
    else:
        candidates.extend(
            [
                getattr(estimate, "att_sc", None),
                diagnostics.get("att_sc"),
            ]
        )

    if source == "augmented":
        effect_vals = pd.to_numeric(estimate.effect_by_time, errors="coerce").to_numpy(dtype=float)
        finite_effects = effect_vals[np.isfinite(effect_vals)]
        if finite_effects.size > 0:
            candidates.append(float(np.mean(finite_effects)))

    for value in candidates:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            return numeric_value

    raise ValueError("Unable to resolve treated ATT from estimate/diagnostics.")


def observed_vs_synthetic_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.10,
) -> plt.Figure:
    """Plot observed treated path against augmented/SC synthetic paths."""
    _ensure_panel_estimate(estimate)

    observed = estimate.observed_outcome
    synthetic_aug = estimate.synthetic_outcome
    synthetic_sc = _optional_series_attr(estimate, "synthetic_outcome_sc")

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
        observed_x = _to_plot_index(observed.index)
        synthetic_aug_x = _to_plot_index(synthetic_aug.index)
        synthetic_sc_x = _to_plot_index(synthetic_sc.index) if synthetic_sc is not None else None

        ax.plot(
            observed_x,
            observed.values,
            color=cycle[0],
            linewidth=2.6,
            label="Observed (treated)",
            zorder=3,
        )
        ax.plot(
            synthetic_aug_x,
            synthetic_aug.values,
            color=cycle[1 % len(cycle)],
            linewidth=2.2,
            label="Synthetic (augmented)",
            zorder=2,
        )
        if show_sc and synthetic_sc is not None and synthetic_sc_x is not None:
            ax.plot(
                synthetic_sc_x,
                synthetic_sc.values,
                color=cycle[2 % len(cycle)],
                linewidth=1.8,
                linestyle="--",
                label="Synthetic (SC)",
                zorder=1,
            )

        ax.axvline(
            _to_plot_time(estimate.treatment_start),
            linestyle="--",
            linewidth=1.7,
            color="0.25",
            label="Intervention",
            zorder=4,
        )
        ax.set_title("Observed vs Synthetic")
        ax.set_xlabel("Time")
        ax.set_ylabel("Outcome")
        ax.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

    plt.close(fig)
    return fig


def gap_over_time_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.10,
) -> plt.Figure:
    """Plot observed-minus-synthetic gap over time with intervention boundary."""
    _ensure_panel_estimate(estimate)

    observed = estimate.observed_outcome
    gap_aug = observed - estimate.synthetic_outcome
    synthetic_sc = _optional_series_attr(estimate, "synthetic_outcome_sc")
    gap_sc = (observed - synthetic_sc) if synthetic_sc is not None else None

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
        gap_aug_x = _to_plot_index(gap_aug.index)
        gap_sc_x = _to_plot_index(gap_sc.index) if gap_sc is not None else None

        ax.plot(
            gap_aug_x,
            gap_aug.values,
            color=cycle[0],
            linewidth=2.3,
            label="Gap (augmented)",
        )
        if show_sc and gap_sc is not None and gap_sc_x is not None:
            ax.plot(
                gap_sc_x,
                gap_sc.values,
                color=cycle[1 % len(cycle)],
                linewidth=1.9,
                linestyle="--",
                label="Gap (SC)",
            )
        ax.axhline(0.0, color="0.35", linewidth=1.2, linestyle=":")
        ax.axvline(
            _to_plot_time(estimate.treatment_start),
            linestyle="--",
            linewidth=1.7,
            color="0.25",
            label="Intervention",
        )

        ax.set_title("Gap Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Gap")
        ax.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

    plt.close(fig)
    return fig


def placebo_att_histogram_plot(
    estimate: PanelEstimate,
    *,
    source: Literal["augmented", "sc"] = "augmented",
    bins: Optional[int] = None,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.10,
) -> plt.Figure:
    """Plot placebo ATT histogram with treated ATT line."""
    _ensure_panel_estimate(estimate)
    if source not in {"augmented", "sc"}:
        raise ValueError("source must be 'augmented' or 'sc'.")

    placebo_atts = _extract_placebo_att_distribution(estimate, source=source)
    treated_att = _resolve_att_value(estimate, source=source)

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

        if placebo_atts.size > 0:
            bins_eff = (
                int(np.clip(np.ceil(np.sqrt(placebo_atts.size)) + 2, 5, 30))
                if bins is None
                else int(bins)
            )
            if bins_eff <= 0:
                raise ValueError("bins must be positive when provided.")
            ax.hist(
                placebo_atts,
                bins=bins_eff,
                color="#5B8FF9",
                edgecolor="white",
                alpha=0.85,
                label="Placebo ATTs",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No placebo ATT draws available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        label_suffix = "augmented" if source == "augmented" else "SC"
        ax.axvline(
            treated_att,
            color="#D7263D",
            linewidth=2.0,
            linestyle="--",
            label=f"Treated ATT ({label_suffix}) = {treated_att:.4g}",
        )

        ax.set_title("Placebo ATT Histogram")
        ax.set_xlabel("ATT")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

    plt.close(fig)
    return fig


__all__ = [
    "observed_vs_synthetic_plot",
    "gap_over_time_plot",
    "placebo_att_histogram_plot",
]
