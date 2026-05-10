r"""Residual diagnostic plots for multi-treatment nuisance models g_k and m_k."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData

from .score_validation import _validate_estimate_matches_data


def _binned_mean_line(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return np.array([], dtype=float), np.array([], dtype=float)

    n = x_arr.size
    n_bins_eff = max(2, min(int(n_bins), n))
    quantiles = np.linspace(0.0, 1.0, n_bins_eff + 1)
    edges = np.quantile(x_arr, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float)

    mids = []
    means = []
    for j in range(edges.size - 1):
        lo = float(edges[j])
        hi = float(edges[j + 1])
        if j < edges.size - 2:
            mask = (x_arr >= lo) & (x_arr < hi)
        else:
            mask = (x_arr >= lo) & (x_arr <= hi)
        if np.any(mask):
            mids.append(float(np.mean(x_arr[mask])))
            means.append(float(np.mean(y_arr[mask])))

    return np.asarray(mids, dtype=float), np.asarray(means, dtype=float)


def _binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    n = x_arr.size
    n_bins_eff = max(2, min(int(n_bins), n))
    quantiles = np.linspace(0.0, 1.0, n_bins_eff + 1)
    edges = np.quantile(x_arr, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    mids = []
    means = []
    counts = []
    for j in range(edges.size - 1):
        lo = float(edges[j])
        hi = float(edges[j + 1])
        if j < edges.size - 2:
            mask = (x_arr >= lo) & (x_arr < hi)
        else:
            mask = (x_arr >= lo) & (x_arr <= hi)
        if np.any(mask):
            mids.append(float(np.mean(x_arr[mask])))
            means.append(float(np.mean(y_arr[mask])))
            counts.append(float(np.sum(mask)))

    return np.asarray(mids, dtype=float), np.asarray(means, dtype=float), np.asarray(counts, dtype=float)


def _coerce_to_one_hot(d_raw: np.ndarray, n_treatments: int) -> np.ndarray:
    d_arr = np.asarray(d_raw, dtype=float)
    if d_arr.ndim == 1:
        labels_float = d_arr.ravel()
        if np.any(~np.isfinite(labels_float)):
            raise ValueError("1D treatment labels must be finite integer-coded values.")
        labels_rounded = np.round(labels_float)
        if np.any(~np.isclose(labels_float, labels_rounded, atol=1e-8, rtol=0.0)):
            raise ValueError(
                "1D treatment labels must be integer-coded values in [0, K-1]; "
                "received non-integer inputs."
            )
        labels = np.asarray(labels_rounded, dtype=int)
        if np.any(labels < 0) or np.any(labels >= n_treatments):
            raise ValueError(
                f"1D treatment labels must be in [0, {n_treatments - 1}], got out-of-range values."
            )
        return np.eye(n_treatments, dtype=float)[labels]

    if d_arr.ndim != 2 or d_arr.shape[1] != n_treatments:
        raise ValueError(
            f"d must be 2D one-hot with shape (n, {n_treatments}) "
            f"or 1D labels; got shape {d_arr.shape}."
        )

    d_bin = (d_arr > 0.5).astype(float)
    row_sums = d_bin.sum(axis=1)
    if np.any(~np.isclose(row_sums, 1.0, atol=1e-8, rtol=0.0)):
        # Fallback: if input is probabilistic-like, convert via argmax.
        idx = np.asarray(np.argmax(d_arr, axis=1), dtype=int)
        d_bin = np.eye(n_treatments, dtype=float)[idx]
    return d_bin


def _resolve_treatment_names(
    *,
    estimate: MultiCausalEstimate,
    data: Optional[MultiCausalData],
    diagnostic_data: Any,
    n_treatments: int,
    cache_names: Any = None,
) -> list[str]:
    if cache_names is not None:
        names = [str(name) for name in list(cache_names)]
        if len(names) == n_treatments:
            return names

    if data is not None:
        names = [str(name) for name in list(data.treatment_names)]
        if len(names) == n_treatments:
            return names

    est_names = [str(name) for name in list(estimate.treatment)]
    if len(est_names) == n_treatments:
        return est_names

    diag_names = getattr(diagnostic_data, "treatment_names", None)
    if diag_names is None:
        diag_names = getattr(diagnostic_data, "d_names", None)
    if diag_names is not None:
        names = [str(name) for name in list(diag_names)]
        if len(names) == n_treatments:
            return names

    return [f"d_{idx}" for idx in range(n_treatments)]


def _clip_propensity_for_plot(m_raw: np.ndarray, *, clip_propensity: float) -> np.ndarray:
    m_arr = np.asarray(m_raw, dtype=float)
    if m_arr.ndim != 2:
        raise ValueError(f"Propensity matrix must be 2D (n, K). Got shape {m_arr.shape}.")
    n_treatments = m_arr.shape[1]
    if n_treatments < 2:
        raise ValueError("Need at least 2 treatment columns for multiclass residual diagnostics.")

    eps = float(max(clip_propensity, 1e-12))
    if not np.isfinite(eps) or not (0.0 <= eps < 0.5):
        raise ValueError(
            f"clip_propensity must be finite and in [0, 0.5); got {eps}."
        )
    return np.clip(m_arr, eps, 1.0 - eps)


def _resolve_residual_inputs(
    estimate: MultiCausalEstimate,
    data: Optional[MultiCausalData],
    *,
    clip_propensity: float,
) -> dict[str, np.ndarray | list[str]]:
    if data is not None:
        _validate_estimate_matches_data(data=data, estimate=estimate)

    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Call estimate(diagnostic_data=True) first."
        )

    # Fast path: reuse cached residual-plot inputs stored in diagnostic_data.
    cache = getattr(diagnostic_data, "residual_plot_cache", None)
    if isinstance(cache, dict):
        required = {"y", "d", "g", "m"}
        if required.issubset(set(cache.keys())):
            y_cached = np.asarray(cache.get("y"), dtype=float).ravel()
            d_cached = np.asarray(cache.get("d"), dtype=float)
            g_cached = np.asarray(cache.get("g"), dtype=float)
            m_cached = np.asarray(cache.get("m"), dtype=float)

            if (
                y_cached.size > 0
                and g_cached.ndim == 2
                and m_cached.ndim == 2
                and g_cached.shape == m_cached.shape
                and g_cached.shape[0] == y_cached.size
            ):
                n_treatments = int(g_cached.shape[1])
                d_cached_oh = _coerce_to_one_hot(d_cached, n_treatments)
                if (
                    d_cached_oh.shape == g_cached.shape
                    and np.all(np.isfinite(y_cached))
                    and np.all(np.isfinite(d_cached_oh))
                    and np.all(np.isfinite(g_cached))
                    and np.all(np.isfinite(m_cached))
                ):
                    treatment_names = _resolve_treatment_names(
                        estimate=estimate,
                        data=data,
                        diagnostic_data=diagnostic_data,
                        n_treatments=n_treatments,
                        cache_names=cache.get("treatment_names"),
                    )
                    return {
                        "y": y_cached,
                        "d": d_cached_oh,
                        "g": g_cached,
                        "m": _clip_propensity_for_plot(
                            m_cached,
                            clip_propensity=float(clip_propensity),
                        ),
                        "treatment_names": treatment_names,
                    }

    m_post = getattr(diagnostic_data, "m_hat", None)
    m_raw = getattr(diagnostic_data, "m_hat_raw", None)
    if m_raw is None:
        m_raw = m_post
    g_raw = getattr(diagnostic_data, "g_hat", None)
    if m_raw is None or g_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `g_hat`.")

    y_raw = getattr(diagnostic_data, "y", None)
    if y_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `y`, or pass `data` for fallback.")
        y_raw = data.get_df()[str(data.outcome)].to_numpy(dtype=float)

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `d`, or pass `data` for fallback.")
        d_raw = data.get_df()[list(data.treatment_names)].to_numpy(dtype=float)

    y = np.asarray(y_raw, dtype=float).ravel()
    g = np.asarray(g_raw, dtype=float)
    m = np.asarray(m_raw, dtype=float)
    if g.ndim != 2 or m.ndim != 2:
        raise ValueError("`g_hat` and `m_hat` must be 2D arrays with shape (n, K).")
    if g.shape != m.shape:
        raise ValueError(f"`g_hat` and `m_hat` must have same shape; got {g.shape} and {m.shape}.")

    n, n_treatments = g.shape
    if y.size != n:
        raise ValueError("All diagnostic arrays must have matching sample size n.")

    d = _coerce_to_one_hot(d_raw, n_treatments)
    if d.shape[0] != n:
        raise ValueError("All diagnostic arrays must have matching sample size n.")

    finite = (
        np.isfinite(y)
        & np.isfinite(d).all(axis=1)
        & np.isfinite(g).all(axis=1)
        & np.isfinite(m).all(axis=1)
    )
    y = y[finite]
    d = d[finite]
    g = g[finite]
    m = m[finite]
    if y.size == 0:
        raise ValueError("No finite observations available for residual plotting.")

    m_unc = np.asarray(m, dtype=float).copy()
    m = _clip_propensity_for_plot(m_unc, clip_propensity=float(clip_propensity))

    treatment_names = _resolve_treatment_names(
        estimate=estimate,
        data=data,
        diagnostic_data=diagnostic_data,
        n_treatments=n_treatments,
    )

    resolved = {
        "y": y,
        "d": d,
        "g": g,
        "m": m,
        "treatment_names": treatment_names,
    }
    try:
        diagnostic_data.residual_plot_cache = {
            "y": np.asarray(y, dtype=float).ravel(),
            "d": np.asarray(d, dtype=float),
            "g": np.asarray(g, dtype=float),
            "m": m_unc,
            "treatment_names": list(treatment_names),
        }
    except Exception:
        pass
    return resolved


def plot_residual_diagnostics(
    estimate: MultiCausalEstimate,
    data: Optional[MultiCausalData] = None,
    *,
    clip_propensity: float = 1e-6,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    r"""
    Plot residual diagnostics for multi-treatment nuisance models.

    Panels
    ------
    1..K. Arm-specific residual-vs-fitted:
          ``u_k = y - g_k`` vs ``g_k`` within arm ``D_k=1``.
    K+1.  Binned calibration error for each arm:
          ``E[D_k - m_k | m_k in bin]`` vs binned ``m_k``.
    """

    resolved = _resolve_residual_inputs(
        estimate=estimate,
        data=data,
        clip_propensity=float(clip_propensity),
    )
    y = np.asarray(resolved["y"], dtype=float).ravel()
    d = np.asarray(resolved["d"], dtype=float)
    g = np.asarray(resolved["g"], dtype=float)
    m = np.asarray(resolved["m"], dtype=float)
    treatment_names = [str(name) for name in list(resolved["treatment_names"])]

    n_treatments = int(g.shape[1])
    missing_arms = [treatment_names[idx] for idx in range(n_treatments) if not np.any(d[:, idx] > 0.5)]
    if missing_arms:
        raise ValueError(
            "All treatment arms must have at least one observation for residual plots. "
            f"Missing: {missing_arms}."
        )

    residuals = y[:, None] - g

    n_panels = n_treatments + 1
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    if figsize == (14.0, 4.8) and n_panels > 3:
        figsize_eff = (4.8 * n_cols, 4.4 * n_rows)
    else:
        figsize_eff = figsize

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 12.5 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 9.5 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }

    with mpl.rc_context(rc):
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=figsize_eff, dpi=dpi)
        axes = np.atleast_1d(axes_grid).ravel()
        arm_axes = axes[:n_treatments]
        ax_m = axes[n_treatments]

        # 1..K) Arm-specific residual diagnostics
        for idx, ax in enumerate(arm_axes):
            color = f"C{idx % 10}"
            arm_mask = d[:, idx] > 0.5
            x_arm = g[arm_mask, idx]
            y_arm = residuals[arm_mask, idx]

            ax.scatter(x_arm, y_arm, s=marker_size, alpha=alpha, color=color, linewidths=0.0)
            xb, yb = _binned_mean_line(x_arm, y_arm, n_bins=int(n_bins))
            if xb.size > 0:
                ax.plot(xb, yb, color="black", linewidth=1.6, label="Binned mean")
                ax.legend(frameon=False)
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
            ax.set_xlabel(rf"$\hat g_{{{idx}}}(X)$")
            ax.set_ylabel(rf"$u_{{{idx}}} = Y - \hat g_{{{idx}}}(X)$")
            ax.set_title(f"{treatment_names[idx]}: Residual vs Fitted")
            ax.grid(True, linewidth=0.5, alpha=0.45)

        # K+1) Calibration error by arm
        summaries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        cmax = 1.0
        for idx in range(n_treatments):
            xb, yb, cb = _binned_summary(
                m[:, idx],
                d[:, idx] - m[:, idx],
                n_bins=int(n_bins),
            )
            summaries.append((xb, yb, cb))
            if cb.size > 0:
                cmax = max(cmax, float(np.max(cb)))

        for idx, (xb, yb, cb) in enumerate(summaries):
            if xb.size == 0:
                continue
            color = f"C{idx % 10}"
            sizes = 30.0 + 180.0 * (cb / cmax)
            ax_m.scatter(
                xb,
                yb,
                s=sizes,
                alpha=0.80,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                label=treatment_names[idx],
                zorder=3,
            )
            ax_m.plot(xb, yb, color=color, linewidth=1.4, zorder=2)

        ax_m.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_m.set_xlim(0.0, 1.0)
        ax_m.set_xlabel(r"$\hat m_k(X)$")
        ax_m.set_ylabel(r"$\mathbb{E}[D_k-\hat m_k(X)\mid \hat m_k(X)\in \mathrm{bin}]$")
        ax_m.set_title("Calibration Error by Propensity Bin")
        ax_m.grid(True, linewidth=0.5, alpha=0.45)
        ax_m.legend(frameon=False, loc="best", ncol=1 if n_treatments <= 4 else 2)

        for ax in axes[:n_panels]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes[n_panels:]:
            fig.delaxes(ax)

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

    plt.close(fig)
    return fig


__all__ = ["plot_residual_diagnostics"]
