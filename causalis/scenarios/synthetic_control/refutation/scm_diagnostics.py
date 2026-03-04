from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.data_contracts.panel_estimate import PanelEstimate


def _to_plot_time(value: object) -> object:
    if isinstance(value, pd.Period):
        return value.to_timestamp()
    return value


def _to_plot_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    return index


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _series_rmse(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _extract_placebo_atts(diagnostics: Dict[str, Any]) -> np.ndarray:
    raw = diagnostics.get("att_placebo_att_distribution")
    if raw is None:
        return np.asarray([], dtype=float)

    values = pd.to_numeric(pd.Series(list(raw)), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values


def _missing_cell_fraction(paneldata: PanelDataSCM) -> float:
    df = paneldata.df_analysis().copy()
    if df.empty:
        return 0.0

    missing = df[paneldata.y].isna()
    return float(missing.mean())


def _save_observed_vs_synthetic(
    *,
    observed: pd.Series,
    synthetic_aug: pd.Series,
    synthetic_sc: pd.Series,
    treatment_start: Any,
    save_path: Path,
    dpi: int,
) -> None:
    rc = {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=dpi)
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
        observed_x = _to_plot_index(observed.index)
        synthetic_aug_x = _to_plot_index(synthetic_aug.index)
        synthetic_sc_x = _to_plot_index(synthetic_sc.index)

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
            _to_plot_time(treatment_start),
            linestyle="--",
            linewidth=1.7,
            color="0.25",
            label="Intervention",
            zorder=4,
        )

        ax.set_title("Observed vs Synthetic Outcome Path")
        ax.set_xlabel("Time")
        ax.set_ylabel("Outcome")
        ax.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def _save_gap_over_time(
    *,
    gap_aug: pd.Series,
    gap_sc: pd.Series,
    treatment_start: Any,
    save_path: Path,
    dpi: int,
) -> None:
    rc = {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=dpi)
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
        gap_aug_x = _to_plot_index(gap_aug.index)
        gap_sc_x = _to_plot_index(gap_sc.index)

        ax.plot(
            gap_aug_x,
            gap_aug.values,
            color=cycle[0],
            linewidth=2.3,
            label="Gap (augmented)",
        )
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
            _to_plot_time(treatment_start),
            linestyle="--",
            linewidth=1.7,
            color="0.25",
            label="Intervention",
        )

        ax.set_title("Gap Over Time (Observed - Synthetic)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Gap")
        ax.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def _save_placebo_histogram(
    *,
    placebo_atts: np.ndarray,
    treated_att: float,
    save_path: Path,
    dpi: int,
) -> None:
    rc = {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=dpi)

        if placebo_atts.size > 0:
            n_bins = int(np.clip(np.ceil(np.sqrt(placebo_atts.size)) + 2, 5, 30))
            ax.hist(
                placebo_atts,
                bins=n_bins,
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

        ax.axvline(
            treated_att,
            color="#D7263D",
            linewidth=2.0,
            linestyle="--",
            label=f"Treated ATT = {treated_att:.4g}",
        )

        ax.set_title("Placebo ATT Distribution")
        ax.set_xlabel("ATT")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def run_scm_diagnostics(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    output_dir: str | Path | None = None,
    filename_prefix: str = "scm_diagnostics",
    pre_tail_k: int = 3,
    dpi: int = 220,
) -> Dict[str, Any]:
    """Run compact SCM diagnostics and save the three v1 diagnostic plots."""
    if not isinstance(estimate, PanelEstimate):
        raise TypeError("estimate must be a PanelEstimate instance.")
    if not isinstance(paneldata, PanelDataSCM):
        raise TypeError("paneldata must be a PanelDataSCM instance.")
    if int(pre_tail_k) <= 0:
        raise ValueError("pre_tail_k must be a positive integer.")
    if estimate.treated_unit != paneldata.treated_unit:
        raise ValueError(
            "estimate.treated_unit must match paneldata.treated_unit "
            f"({estimate.treated_unit!r} != {paneldata.treated_unit!r})."
        )
    if estimate.treatment_start != paneldata.treatment_start:
        raise ValueError(
            "estimate.treatment_start must match paneldata.treatment_start "
            f"({estimate.treatment_start!r} != {paneldata.treatment_start!r})."
        )

    diagnostics = dict(estimate.diagnostics or {})
    out_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    observed = estimate.observed_outcome.copy()
    synthetic_aug = estimate.synthetic_outcome.copy()
    synthetic_sc = estimate.synthetic_outcome_sc.copy()
    gap_aug = observed - synthetic_aug
    gap_sc = observed - synthetic_sc

    pre_times = list(estimate.pre_times)
    k_eff = min(int(pre_tail_k), len(pre_times))
    if k_eff > 0:
        tail_pre_times = pre_times[-k_eff:]
        mean_gap_last_k_pre_aug = _as_finite_float(gap_aug.loc[tail_pre_times].mean())
        mean_gap_last_k_pre_sc = _as_finite_float(gap_sc.loc[tail_pre_times].mean())
    else:
        mean_gap_last_k_pre_aug = None
        mean_gap_last_k_pre_sc = None

    pre_gap_aug = gap_aug.loc[pre_times]
    pre_gap_sc = gap_sc.loc[pre_times]

    pre_rmse_sc = _as_finite_float(diagnostics.get("pre_rmse_sc"))
    if pre_rmse_sc is None:
        pre_rmse_sc = _series_rmse(pre_gap_sc)

    pre_rmse_aug = _as_finite_float(diagnostics.get("pre_rmse_augmented"))
    if pre_rmse_aug is None:
        pre_rmse_aug = _series_rmse(pre_gap_aug)

    w_sc = np.asarray(list(estimate.donor_weights_sc.values()), dtype=float)
    w_aug = np.asarray(list(estimate.donor_weights_augmented.values()), dtype=float)
    max_weight_sc = float(np.max(w_sc)) if w_sc.size > 0 else None
    max_abs_weight_aug = float(np.max(np.abs(w_aug))) if w_aug.size > 0 else None
    l1_norm_weight_aug = float(np.sum(np.abs(w_aug))) if w_aug.size > 0 else None

    placebo_atts = _extract_placebo_atts(diagnostics)
    n_placebos = diagnostics.get("att_placebo_n")
    try:
        n_placebos_eff = int(n_placebos) if n_placebos is not None else int(placebo_atts.size)
    except (TypeError, ValueError):
        n_placebos_eff = int(placebo_atts.size)
    if n_placebos_eff < 0:
        n_placebos_eff = int(placebo_atts.size)

    min_possible_p = _as_finite_float(diagnostics.get("att_placebo_min_possible_p"))
    if min_possible_p is None and n_placebos_eff > 0:
        min_possible_p = float(1.0 / float(n_placebos_eff + 1))

    p_value_att = _as_finite_float(diagnostics.get("att_placebo_p_value"))
    if p_value_att is None:
        p_value_att = _as_finite_float(estimate.p_value)

    ci_low_abs = _as_finite_float(diagnostics.get("att_placebo_ci_lower_absolute"))
    if ci_low_abs is None:
        ci_low_abs = _as_finite_float(estimate.ci_lower_absolute)
    ci_high_abs = _as_finite_float(diagnostics.get("att_placebo_ci_upper_absolute"))
    if ci_high_abs is None:
        ci_high_abs = _as_finite_float(estimate.ci_upper_absolute)

    placebo_ci_is_unbounded_raw = diagnostics.get("att_placebo_ci_is_unbounded")
    placebo_ci_is_unbounded = (
        bool(placebo_ci_is_unbounded_raw) if placebo_ci_is_unbounded_raw is not None else False
    )

    is_robust_model = str(estimate.model) == "RobustSyntheticControl"
    missing_cell_fraction = _as_finite_float(diagnostics.get("missing_cell_fraction"))
    if missing_cell_fraction is None and is_robust_model:
        missing_cell_fraction = _missing_cell_fraction(paneldata)

    completion_converged_raw = diagnostics.get("completion_converged")
    completion_converged = (
        bool(completion_converged_raw) if completion_converged_raw is not None else None
    )
    completion_effective_rank = diagnostics.get("completion_effective_rank")
    if completion_effective_rank is not None:
        try:
            completion_effective_rank = int(completion_effective_rank)
        except (TypeError, ValueError):
            completion_effective_rank = None

    observed_vs_synth_path = out_dir / f"{filename_prefix}_observed_vs_synthetic.png"
    gap_path = out_dir / f"{filename_prefix}_gap_over_time.png"
    placebo_hist_path = out_dir / f"{filename_prefix}_placebo_att_histogram.png"

    _save_observed_vs_synthetic(
        observed=observed,
        synthetic_aug=synthetic_aug,
        synthetic_sc=synthetic_sc,
        treatment_start=estimate.treatment_start,
        save_path=observed_vs_synth_path,
        dpi=int(dpi),
    )
    _save_gap_over_time(
        gap_aug=gap_aug,
        gap_sc=gap_sc,
        treatment_start=estimate.treatment_start,
        save_path=gap_path,
        dpi=int(dpi),
    )
    _save_placebo_histogram(
        placebo_atts=placebo_atts,
        treated_att=float(estimate.att),
        save_path=placebo_hist_path,
        dpi=int(dpi),
    )

    metrics: Dict[str, Any] = {
        "n_donors": int(len(estimate.donor_weights_sc)),
        "n_pre": int(len(estimate.pre_times)),
        "n_post": int(len(estimate.post_times)),
        "pre_rmse_sc": pre_rmse_sc,
        "pre_rmse_aug": pre_rmse_aug,
        "att_sc": _as_finite_float(estimate.att_sc),
        "att_aug": _as_finite_float(estimate.att),
        "max_weight_sc": max_weight_sc,
        "max_abs_weight_aug": max_abs_weight_aug,
        "l1_norm_weight_aug": l1_norm_weight_aug,
        "cond_augmented_gram": _as_finite_float(diagnostics.get("cond_augmented_gram")),
        "n_placebos": n_placebos_eff,
        "min_possible_p": min_possible_p,
        "p_value_att": p_value_att,
        "ci_low_abs": ci_low_abs,
        "ci_high_abs": ci_high_abs,
        "placebo_ci_is_unbounded": placebo_ci_is_unbounded,
        "missing_cell_fraction": missing_cell_fraction,
        "completion_converged": completion_converged,
        "completion_effective_rank": completion_effective_rank,
        "mean_gap_last_k_pre_sc": mean_gap_last_k_pre_sc,
        "mean_gap_last_k_pre_aug": mean_gap_last_k_pre_aug,
        "pre_tail_k_used": int(k_eff),
    }

    return {"metrics": metrics}


__all__ = ["run_scm_diagnostics"]
