from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM


def _validate_paneldata(paneldata: PanelDataSCM) -> None:
    if not isinstance(paneldata, PanelDataSCM):
        raise TypeError("paneldata must be a PanelDataSCM instance.")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _series_slope(series: pd.Series) -> float | None:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return None
    x = np.arange(arr.size, dtype=float)[mask]
    y = arr[mask]
    x_centered = x - np.mean(x)
    denom = float(np.sum(x_centered**2))
    if denom <= 0.0:
        return None
    slope = float(np.sum(x_centered * (y - np.mean(y))) / denom)
    return slope


def _pre_pivot(paneldata: PanelDataSCM) -> pd.DataFrame:
    df = paneldata.df_analysis().copy()
    collapsed = (
        df.groupby([paneldata.time_col, paneldata.unit_col], as_index=False, sort=True)[paneldata.y]
        .mean()
    )
    pivot = collapsed.pivot(index=paneldata.time_col, columns=paneldata.unit_col, values=paneldata.y)
    return pivot.reindex(index=list(paneldata.pre_times()))


def _rank_values(values: pd.Series, *, ascending: bool) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if ascending:
        sort_key = np.where(np.isfinite(arr), arr, np.inf)
    else:
        sort_key = np.where(np.isfinite(arr), -arr, np.inf)
    order = np.argsort(sort_key, kind="mergesort")
    ranks = np.empty(arr.size, dtype=int)
    ranks[order] = np.arange(1, arr.size + 1, dtype=int)
    return ranks


def donors_diagnostics(panel_data: PanelDataSCM) -> pd.DataFrame:
    """Build one donor-level diagnostics table for SCM feasibility checks."""
    _validate_paneldata(panel_data)

    pre_pivot = _pre_pivot(panel_data)
    donors = list(panel_data.donor_pool())
    treated = panel_data.treated_unit
    if treated not in pre_pivot.columns:
        raise ValueError("treated_unit is not present in pre-period panel data.")

    treated_series = pre_pivot[treated]
    treated_mean = _safe_float(np.nanmean(pd.to_numeric(treated_series, errors="coerce")))
    treated_std = _safe_float(np.nanstd(pd.to_numeric(treated_series, errors="coerce"), ddof=0))
    treated_slope = _series_slope(treated_series)

    base_df = panel_data.df_analysis()
    rows: list[dict[str, Any]] = []
    for donor in donors:
        donor_series = (
            pre_pivot[donor]
            if donor in pre_pivot.columns
            else pd.Series(np.nan, index=pre_pivot.index, dtype=float)
        )

        donor_vals = pd.to_numeric(donor_series, errors="coerce").to_numpy(dtype=float)
        treated_vals = pd.to_numeric(treated_series, errors="coerce").to_numpy(dtype=float)
        overlap = np.isfinite(donor_vals) & np.isfinite(treated_vals)

        corr = None
        if int(overlap.sum()) >= 2:
            t_ov = treated_vals[overlap]
            d_ov = donor_vals[overlap]
            if np.std(t_ov) > 0.0 and np.std(d_ov) > 0.0:
                corr = _safe_float(np.corrcoef(t_ov, d_ov)[0, 1])

        rmse = None
        max_abs_gap = None
        if int(overlap.sum()) >= 1:
            gaps = donor_vals[overlap] - treated_vals[overlap]
            rmse = _safe_float(np.sqrt(np.mean(np.square(gaps))))
            max_abs_gap = _safe_float(np.max(np.abs(gaps)))

        donor_mean = _safe_float(np.nanmean(donor_vals))
        donor_slope = _series_slope(donor_series)
        rmse_std = None
        if rmse is not None and treated_std is not None and treated_std > 0.0:
            rmse_std = _safe_float(rmse / treated_std)

        is_never_treated = bool(
            (base_df.loc[base_df[panel_data.unit_col] == donor, panel_data.treated_time] == 0).all()
        )

        rows.append(
            {
                "donor": donor,
                "pre_mean": donor_mean,
                "pre_std": _safe_float(np.nanstd(donor_vals, ddof=0)),
                "pre_slope": donor_slope,
                "corr_with_treated_pre": corr,
                "rmse_to_treated_pre": rmse,
                "rmse_to_treated_pre_standardized": rmse_std,
                "mean_diff_pre": (
                    None
                    if donor_mean is None or treated_mean is None
                    else _safe_float(donor_mean - treated_mean)
                ),
                "slope_diff_pre": (
                    None
                    if donor_slope is None or treated_slope is None
                    else _safe_float(donor_slope - treated_slope)
                ),
                "max_abs_gap_pre": max_abs_gap,
                "is_never_treated": is_never_treated,
                "n_missing_pre": int(pd.isna(donor_series).sum()),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "donor",
                "pre_mean",
                "pre_std",
                "pre_slope",
                "corr_with_treated_pre",
                "rmse_to_treated_pre",
                "rmse_to_treated_pre_standardized",
                "mean_diff_pre",
                "slope_diff_pre",
                "max_abs_gap_pre",
                "is_never_treated",
                "n_missing_pre",
                "corr_rank",
                "std_rmse_rank",
                "slope_rank",
                "composite_similarity_score",
                "rank_by_similarity",
                "notes",
            ]
        )

    corr_rank = _rank_values(table["corr_with_treated_pre"], ascending=False)
    rmse_rank = _rank_values(table["rmse_to_treated_pre_standardized"], ascending=True)
    slope_rank = _rank_values(table["slope_diff_pre"].abs(), ascending=True)

    n = float(len(table))
    composite = (
        (n + 1.0 - corr_rank.astype(float))
        + (n + 1.0 - rmse_rank.astype(float))
        + (n + 1.0 - slope_rank.astype(float))
    ) / (3.0 * n)

    table["corr_rank"] = corr_rank.astype(int)
    table["std_rmse_rank"] = rmse_rank.astype(int)
    table["slope_rank"] = slope_rank.astype(int)
    table["composite_similarity_score"] = composite
    table["rank_by_similarity"] = _rank_values(table["composite_similarity_score"], ascending=False).astype(int)

    notes: list[str] = []
    for _, row in table.iterrows():
        flags: list[str] = []
        missing_pre = int(row["n_missing_pre"])
        corr = _safe_float(row["corr_with_treated_pre"])
        std_rmse = _safe_float(row["rmse_to_treated_pre_standardized"])
        if missing_pre > 0:
            flags.append("missing_pre")
        if corr is not None and corr < 0.0:
            flags.append("negative_corr")
        if std_rmse is not None and std_rmse > 1.0:
            flags.append("high_std_rmse")
        notes.append("; ".join(flags) if flags else "ok")
    table["notes"] = notes

    return table.sort_values("rank_by_similarity", kind="mergesort").reset_index(drop=True)


def run_scm_feasibility(paneldata: PanelDataSCM) -> Dict[str, pd.DataFrame]:
    """Return core SCM feasibility tables from panel data only (EDA phase)."""
    _validate_paneldata(paneldata)
    return {
        "donors_diagnostics": donors_diagnostics(paneldata),
    }


__all__ = [
    "donors_diagnostics",
    "run_scm_feasibility",
]
