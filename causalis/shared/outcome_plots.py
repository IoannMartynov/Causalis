from __future__ import annotations

from typing import Optional, Tuple, Union, TYPE_CHECKING

import inspect
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from causalis.dgp.causaldata import CausalData
    from causalis.dgp.multicausaldata import MultiCausalData


def _silverman_bandwidth(x: np.ndarray) -> float:
    """
    Silverman's rule of thumb for KDE bandwidth.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return 0.04
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    s = sd if iqr <= 0 else min(sd, iqr / 1.34)
    h = 0.9 * s * n ** (-1 / 5)
    return float(max(h, 1e-6))


def _kde_unbounded(x: np.ndarray, xs: np.ndarray, h: float) -> np.ndarray:
    """
    Gaussian KDE on R, implemented with NumPy (no SciPy).
    Handles degenerate cases by drawing a small bump at the mean.
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return np.zeros_like(xs)
    if x.size < 2 or np.std(x) < 1e-12:
        mu = float(np.mean(x)) if x.size else 0.0
        h0 = max(h, 1e-3)
        z = (xs - mu) / h0
        return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * h0)
    diff = (xs[None, :] - x[:, None]) / h
    kern = np.exp(-0.5 * diff ** 2) / (np.sqrt(2 * np.pi) * h)
    return kern.mean(axis=0)


def _first_patch_color(patches, fallback):
    """
    Get the face color of the first patch in a collection.
    """
    for p in patches:
        fc = p.get_facecolor()
        if fc is not None:
            return fc
    return fallback


def _is_binary_numeric(s: pd.Series) -> bool:
    """
    Check if a numeric series has at most two unique finite values.
    """
    s = s.dropna()
    if s.empty:
        return False
    if pd.api.types.is_bool_dtype(s):
        return True
    if not pd.api.types.is_numeric_dtype(s):
        return False
    uniq = pd.unique(s)
    if len(uniq) > 2:
        return False
    uniq = np.asarray(uniq, float)
    uniq = uniq[np.isfinite(uniq)]
    return 0 < uniq.size <= 2


def _resolve_palette(treatments, palette, default_cycle):
    """
    Resolve a treatment->color mapping from a list or dict palette.
    """
    if palette is None:
        return {tr: default_cycle[i % len(default_cycle)] for i, tr in enumerate(treatments)}
    if isinstance(palette, dict):
        return {
            tr: palette.get(tr, default_cycle[i % len(default_cycle)])
            for i, tr in enumerate(treatments)
        }
    if isinstance(palette, (list, tuple)):
        if len(palette) == 0:
            return {tr: default_cycle[i % len(default_cycle)] for i, tr in enumerate(treatments)}
        return {tr: palette[i % len(palette)] for i, tr in enumerate(treatments)}
    raise ValueError("palette must be a dict, list/tuple, or None.")


def _boxplot_with_tick_labels(ax, data, labels, **kwargs):
    """
    Call Axes.boxplot across Matplotlib versions that accept either
    ``tick_labels`` or the older ``labels`` keyword.
    """
    label_kw = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
    return ax.boxplot(data, **{label_kw: labels}, **kwargs)


def _looks_like_multicausal_data(data: object) -> bool:
    """
    Runtime check for MultiCausalData-like contracts without importing at runtime.
    """
    return hasattr(data, "treatment_names") and hasattr(data, "control_treatment")


def _resolve_plot_frame(
        data,
        treatment: Optional[str],
        outcome: Optional[str],
) -> Tuple[pd.DataFrame, str, str, Optional[np.ndarray]]:
    """
    Resolve plotting frame and column names for CausalData / MultiCausalData.
    """
    df = getattr(data, "df")

    y_attr = getattr(data, "outcome")
    y_col_default = y_attr.name if isinstance(y_attr, pd.Series) else y_attr
    y_col = outcome or y_col_default

    treatment_order: Optional[np.ndarray] = None

    if _looks_like_multicausal_data(data) and treatment is None:
        t_cols = list(getattr(data, "treatment_names"))
        if y_col not in df.columns:
            raise ValueError("Specified treatment/outcome columns not found in DataFrame.")
        if not all(col in df.columns for col in t_cols):
            raise ValueError("Specified treatment/outcome columns not found in DataFrame.")

        assigned_idx = df[t_cols].to_numpy(dtype=int, copy=False).argmax(axis=1)
        assigned_treatment = pd.Categorical.from_codes(
            assigned_idx,
            categories=t_cols,
            ordered=True,
        )

        t_col = "treatment"
        while t_col in df.columns or t_col == y_col:
            t_col = "_" + t_col

        df = pd.DataFrame({t_col: assigned_treatment, y_col: df[y_col]})
        treatment_order = np.asarray(t_cols, dtype=object)
    else:
        if _looks_like_multicausal_data(data):
            t_col_default = None
        else:
            t_attr = getattr(data, "treatment")
            t_col_default = t_attr.name if isinstance(t_attr, pd.Series) else t_attr

        t_col = treatment or t_col_default
        if t_col is None:
            raise ValueError("Treatment column is not available in the provided data contract.")

    if t_col not in df.columns or y_col not in df.columns:
        raise ValueError("Specified treatment/outcome columns not found in DataFrame.")

    return df, t_col, y_col, treatment_order


def outcome_plot_dist(
        data: Union[CausalData, MultiCausalData],
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        bins: Union[str, int] = "fd",
        density: bool = True,
        alpha: float = 0.45,
        sharex: bool = True,
        kde: bool = True,
        clip: Optional[Tuple[float, float]] = (0.01, 0.99),
        figsize: Tuple[float, float] = (9, 5.5),
        dpi: int = 220,
        font_scale: float = 1.15,
        palette: Optional[Union[list, dict]] = None,
        save: Optional[str] = None,
        save_dpi: Optional[int] = None,
        transparent: bool = False,
) -> plt.Figure:
    """
    Plot the distribution of the outcome for each treatment on a single, pretty plot.

    Features
    --------
    - High-DPI canvas + scalable fonts
    - Default Matplotlib colors; KDE & mean lines match their histogram colors
    - Numeric outcomes: shared x-range (optional), optional KDE, quantile clipping
    - Categorical outcomes: normalized grouped bars by treatment
    - Binary outcomes: proportion bars with percent labels (no KDE)
    - Optional hi-res export (PNG/SVG/PDF)

    Parameters
    ----------
    data : CausalData or MultiCausalData
        The causal dataset containing the dataframe and metadata.
    treatment : str, optional
        Treatment column name. For MultiCausalData, if not provided, one-hot
        treatment columns are converted to assigned treatment labels.
    outcome : str, optional
        Outcome column name. Defaults to the one in `data_contracts`.
    bins : str or int, default "fd"
        Number of bins for histograms (e.g., "fd", "auto", or an integer).
    density : bool, default True
        Whether to normalize histograms to form a density.
    alpha : float, default 0.45
        Transparency for overlaid histograms and bars.
    sharex : bool, default True
        If True, use the same x-limits across treatments for numeric outcomes.
    kde : bool, default True
        Whether to overlay a smooth density (KDE) for numeric outcomes.
    clip : tuple, optional, default (0.01, 0.99)
        Quantiles to trim tails for nicer view of numeric outcomes.
    figsize : tuple, default (9, 5.5)
        Figure size in inches (width, height).
    dpi : int, default 220
        Dots per inch for the figure.
    font_scale : float, default 1.15
        Scaling factor for all font sizes in the plot.
    palette : list or dict, optional
        Color palette for treatments (list in treatment order or dict {treatment: color}).
    save : str, optional
        Path to save the figure (e.g., "outcome.png").
    save_dpi : int, optional
        DPI for the saved figure. Defaults to 300 for raster formats.
    transparent : bool, default False
        Whether to save the figure with a transparent background.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    df, t_col, y_col, treatment_order = _resolve_plot_frame(data, treatment, outcome)
    treatments = treatment_order if treatment_order is not None else pd.unique(df[t_col])
    valid = df[[t_col, y_col]].dropna()
    if valid.empty:
        raise ValueError("No non-missing values for the selected treatment/outcome.")

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

        is_numeric = pd.api.types.is_numeric_dtype(valid[y_col])
        is_binary_numeric = _is_binary_numeric(valid[y_col])
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        colors = _resolve_palette(treatments, palette, cycle)

        if is_binary_numeric:
            # BINARY: 1 bar per treatment (rate of "positive" value)
            vals = pd.unique(valid[y_col])
            if pd.api.types.is_bool_dtype(valid[y_col]):
                pos_value = True
            else:
                vals_num = np.asarray(vals, float)
                vals_num = vals_num[np.isfinite(vals_num)]
                pos_value = float(np.max(vals_num)) if vals_num.size else 1.0

            heights = []
            labels = []
            ns = []
            for tr in treatments:
                sub = valid.loc[valid[t_col] == tr, y_col]
                n = sub.shape[0]
                ns.append(n)
                if n == 0:
                    heights.append(0.0)
                else:
                    if pd.api.types.is_bool_dtype(sub):
                        heights.append(float(sub.mean()))
                    else:
                        heights.append(float((sub == pos_value).mean()))
                labels.append(str(tr))

            x = np.arange(len(treatments))
            bars = ax.bar(
                x,
                heights,
                width=0.6,
                alpha=alpha,
                color=[colors[tr] for tr in treatments],
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, tr, n in zip(bars, treatments, ns):
                bar.set_label(f"{tr} (n={n})")
            for bar, h in zip(bars, heights):
                if h <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.015,
                    f"{h:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9 * font_scale,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([str(lab) for lab in labels])
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
            ax.set_ylabel(f"Pr({y_col}={pos_value})")
            ax.set_xlabel(str(t_col))
            ax.set_title("Outcome rate by treatment")
            ax.grid(True, axis="y", linewidth=0.5, alpha=0.45)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.legend(title=str(t_col), frameon=False)
            fig.tight_layout()

        elif not is_numeric:
            # CATEGORICAL: normalized grouped bars
            vals = pd.unique(valid[y_col])
            vals_sorted = sorted(vals, key=lambda v: (str(type(v)), str(v)))
            width = 0.8 / max(1, len(treatments))
            x = np.arange(len(vals_sorted))

            for i, tr in enumerate(treatments):
                sub = valid.loc[valid[t_col] == tr, y_col]
                counts = sub.value_counts(normalize=True)
                heights = [float(counts.get(v, 0.0)) for v in vals_sorted]
                ax.bar(
                    x + i * width,
                    heights,
                    width=width,
                    alpha=alpha,
                    label=f"{tr} (n={sub.shape[0]})",
                    color=colors[tr],
                    edgecolor="white",
                    linewidth=0.6,
                )

            ax.set_xticks(x + (len(treatments) - 1) * width / 2)
            ax.set_xticklabels([str(v) for v in vals_sorted])
            ax.set_ylabel("Proportion")
            ax.set_xlabel(str(y_col))
            ax.set_title("Outcome distribution by treatment")
            ax.grid(True, axis="y", linewidth=0.5, alpha=0.45)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.legend(title=str(t_col), frameon=False)
            fig.tight_layout()

        else:
            # NUMERIC: overlay histograms (+ optional KDE) per treatment
            y_all = valid[y_col].to_numpy()

            if sharex:
                if clip:
                    lo, hi = np.quantile(y_all, [clip[0], clip[1]])
                else:
                    lo, hi = np.nanmin(y_all), np.nanmax(y_all)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                hist_range = (float(lo), float(hi))
                ax.set_xlim(*hist_range)
            else:
                hist_range = None

            used_colors = {}

            for i, tr in enumerate(treatments):
                y_vals = valid.loc[valid[t_col] == tr, y_col].to_numpy()
                y_vals = y_vals[np.isfinite(y_vals)]
                if y_vals.size == 0:
                    continue

                color_this = colors[tr]
                h = ax.hist(
                    y_vals,
                    bins=bins,
                    density=density,
                    alpha=alpha,
                    label=f"{tr} (n={y_vals.size})",
                    range=hist_range,
                    edgecolor="white",
                    linewidth=0.6,
                    color=color_this,
                )
                used_colors[tr] = _first_patch_color(h[2], color_this)

            if kde and len(used_colors) > 0:
                if hist_range is None:
                    if clip:
                        lo, hi = np.quantile(y_all, [clip[0], clip[1]])
                    else:
                        lo, hi = np.nanmin(y_all), np.nanmax(y_all)
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo, hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                else:
                    lo, hi = hist_range

                xs = np.linspace(float(lo), float(hi), 800)

                for tr in treatments:
                    if tr not in used_colors:
                        continue
                    y_vals = valid.loc[valid[t_col] == tr, y_col].to_numpy()
                    y_vals = y_vals[np.isfinite(y_vals)]
                    if y_vals.size == 0:
                        continue
                    hbw = _silverman_bandwidth(y_vals)
                    dens = _kde_unbounded(y_vals, xs, hbw)
                    if not density:
                        bw = (hi - lo) / (bins if isinstance(bins, int) else 30)
                        dens = dens * y_vals.size * bw
                    ax.plot(xs, dens, linewidth=2.2, color=used_colors[tr],
                            label=f"{tr} (KDE)")

            for tr in treatments:
                y_vals = valid.loc[valid[t_col] == tr, y_col].to_numpy()
                y_vals = y_vals[np.isfinite(y_vals)]
                if y_vals.size == 0:
                    continue
                mu = float(np.mean(y_vals))
                if sharex and hist_range is not None:
                    mu = float(np.clip(mu, hist_range[0], hist_range[1]))
                ax.axvline(mu, linestyle=":", linewidth=1.8,
                           color=used_colors.get(tr, "k"), alpha=0.95)

            ax.set_xlabel(str(y_col))
            ax.set_ylabel("Density" if density else "Count")
            ax.set_title("Outcome distribution by treatment")
            ax.grid(True, linewidth=0.5, alpha=0.45)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.legend(title=str(t_col), frameon=False)
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


def outcome_plot_boxplot(
        data: Union[CausalData, MultiCausalData],
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        figsize: Tuple[float, float] = (9, 5.5),
        dpi: int = 220,
        font_scale: float = 1.15,
        showfliers: bool = True,
        patch_artist: bool = True,
        palette: Optional[Union[list, dict]] = None,
        save: Optional[str] = None,
        save_dpi: Optional[int] = None,
        transparent: bool = False,
) -> plt.Figure:
    """
    Prettified boxplot of the outcome by treatment.

    Features
    --------
    - High-DPI figure, scalable fonts
    - Soft modern color styling (default Matplotlib palette)
    - Optional outliers, gentle transparency
    - Optional save to PNG/SVG/PDF

    Parameters
    ----------
    data : CausalData or MultiCausalData
        The causal dataset containing the dataframe and metadata.
    treatment : str, optional
        Treatment column name. For MultiCausalData, if not provided, one-hot
        treatment columns are converted to assigned treatment labels.
    outcome : str, optional
        Outcome column name. Defaults to the one in `data_contracts`.
    figsize : tuple, default (9, 5.5)
        Figure size in inches (width, height).
    dpi : int, default 220
        Dots per inch for the figure.
    font_scale : float, default 1.15
        Scaling factor for all font sizes in the plot.
    showfliers : bool, default True
        Whether to show outliers (fliers).
    patch_artist : bool, default True
        Whether to fill boxes with color.
    palette : list or dict, optional
        Color palette for treatments (list in treatment order or dict {treatment: color}).
    save : str, optional
        Path to save the figure (e.g., "boxplot.png").
    save_dpi : int, optional
        DPI for the saved figure. Defaults to 300 for raster formats.
    transparent : bool, default False
        Whether to save the figure with a transparent background.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    df, t_col, y_col, treatment_order = _resolve_plot_frame(data, treatment, outcome)

    df_valid = df[[t_col, y_col]].dropna()
    if df_valid.empty:
        raise ValueError("No valid rows with both treatment and outcome present.")

    treatments = treatment_order if treatment_order is not None else pd.unique(df_valid[t_col])
    plot_data = [df_valid.loc[df_valid[t_col] == tr, y_col].values for tr in treatments]

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
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        color_map = _resolve_palette(treatments, palette, cycle)
        colors = [color_map[tr] for tr in treatments]

        bp = _boxplot_with_tick_labels(
            ax,
            plot_data,
            patch_artist=patch_artist,
            labels=[str(tr) for tr in treatments],
            showfliers=showfliers,
            boxprops=dict(linewidth=1.1, alpha=0.8),
            whiskerprops=dict(linewidth=1.0, alpha=0.8),
            capprops=dict(linewidth=1.0, alpha=0.8),
            medianprops=dict(linewidth=2.0, color="black"),
            flierprops=dict(
                marker="o",
                markersize=4,
                markerfacecolor="grey",
                alpha=0.6,
                markeredgewidth=0.3,
            ),
        )

        if patch_artist:
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)

        ax.set_xlabel(str(t_col))
        ax.set_ylabel(str(y_col))
        ax.set_title("Outcome by treatment (boxplot)")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
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




def outcome_plots(
        data: Union[CausalData, MultiCausalData],
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        bins: int = 30,
        density: bool = True,
        alpha: float = 0.5,
        figsize: Tuple[float, float] = (7, 4),
        sharex: bool = True,
        palette: Optional[Union[list, dict]] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot the distribution of the outcome for every treatment on one plot,
    and also produce a boxplot by treatment to visualize outliers.

    Parameters
    ----------
    data : CausalData or MultiCausalData
        The causal dataset containing the dataframe and metadata.
    treatment : str, optional
        Treatment column name. Defaults to the one in `data_contracts`.
    outcome : str, optional
        Outcome column name. Defaults to the one in `data_contracts`.
    bins : int, default 30
        Number of bins for histograms when the outcome is numeric.
    density : bool, default True
        Whether to normalize histograms to form a density.
    alpha : float, default 0.5
        Transparency for overlaid histograms.
    figsize : tuple, default (7, 4)
        Figure size for the plots (width, height).
    sharex : bool, default True
        If True and the outcome is numeric, use the same x-limits across treatments.
    palette : list or dict, optional
        Color palette for treatments (list in treatment order or dict {treatment: color}).

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
        (fig_distribution, fig_boxplot)
    """
    fig_hist = outcome_plot_dist(
        data=data,
        treatment=treatment,
        outcome=outcome,
        bins=bins,
        density=density,
        alpha=alpha,
        figsize=figsize,
        sharex=sharex,
        palette=palette,
    )
    fig_box = outcome_plot_boxplot(
        data=data,
        treatment=treatment,
        outcome=outcome,
        figsize=figsize,
        palette=palette,
    )

    return fig_hist, fig_box
