from __future__ import annotations

from statistics import NormalDist
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, Field

from causalis.data_contracts.causal_diagnostic_data import CUPEDDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData

FLAG_GREEN = "GREEN"
FLAG_YELLOW = "YELLOW"
FLAG_RED = "RED"
FLAG_LEVEL = {FLAG_GREEN: 0, FLAG_YELLOW: 1, FLAG_RED: 2}
FLAG_COLOR = {
    FLAG_GREEN: "#2e7d32",
    FLAG_YELLOW: "#f9a825",
    FLAG_RED: "#c62828",
}


class RegressionChecks(BaseModel):
    """Lightweight OLS/regression health checks for CUPED refutation."""

    ate_naive: float
    ate_adj: float
    ate_gap: float
    ate_gap_over_se_naive: Optional[float] = None

    k: int
    rank: int
    full_rank: bool
    condition_number: float
    p_main_covariates: int
    near_duplicate_pairs: list[tuple[str, str, float]] = Field(default_factory=list)
    vif: Optional[Dict[str, float]] = None

    resid_scale_mad: float
    n_std_resid_gt_3: int
    n_std_resid_gt_4: int
    max_abs_std_resid: float

    max_leverage: float
    leverage_cutoff: float
    n_high_leverage: int

    max_cooks: float
    cooks_cutoff: float
    n_high_cooks: int

    min_one_minus_h: float
    n_tiny_one_minus_h: int

    winsor_q: Optional[float] = None
    ate_adj_winsor: Optional[float] = None
    ate_adj_winsor_gap: Optional[float] = None


def _normalize_flag(flag: Any) -> str:
    value = str(flag).upper()
    return value if value in FLAG_LEVEL else FLAG_GREEN


def design_matrix_checks(design: pd.DataFrame) -> tuple[int, int, bool, float]:
    """Return rank/conditioning refutation checks for a numeric design matrix."""
    z = np.asarray(design, dtype=float)
    k = int(z.shape[1])
    rank = int(np.linalg.matrix_rank(z))
    full_rank = bool(rank == k)
    cond = float(np.linalg.cond(z))
    return k, rank, full_rank, cond


def near_duplicate_corr_pairs(
    x: pd.DataFrame,
    tol: float,
    max_pairs: int = 50,
) -> list[tuple[str, str, float]]:
    """Find pairs with absolute correlation very close to one."""
    if x.shape[1] < 2:
        return []
    cmat = np.corrcoef(np.asarray(x, dtype=float), rowvar=False)
    cols = list(x.columns)
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = float(cmat[i, j])
            if np.isfinite(corr) and abs(corr) > 1.0 - tol:
                pairs.append((str(cols[i]), str(cols[j]), corr))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs[:max_pairs]


def vif_from_corr(x: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Approximate VIF from inverse correlation matrix of standardized covariates.."""
    if x.shape[1] < 2:
        return None
    arr = np.asarray(x, dtype=float)
    std = arr.std(axis=0, ddof=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
        return None
    arr = (arr - arr.mean(axis=0)) / std
    corr = np.corrcoef(arr, rowvar=False)
    if np.any(~np.isfinite(corr)):
        return None
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        return None
    vifs = np.diag(inv_corr)
    return {str(name): float(vifs[i]) for i, name in enumerate(x.columns)}


def leverage_and_cooks(
    y: np.ndarray,
    z: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute leverage, Cook's distance, and internally studentized residuals."""
    n, k = z.shape
    resid = y - z @ params
    xtx_inv = np.linalg.pinv(z.T @ z)
    h = np.einsum("ij,jk,ik->i", z, xtx_inv, z)
    h = np.clip(h, 0.0, 1.0)

    df_resid = max(int(n - k), 1)
    rss = float(resid @ resid)
    mse = rss / float(df_resid)
    if (not np.isfinite(mse)) or mse <= 0.0:
        mse = float(np.finfo(float).eps)

    one_minus_h = np.maximum(1.0 - h, 1e-15)
    cooks = (resid ** 2 / (float(k) * mse)) * (h / (one_minus_h ** 2))
    std_resid = resid / (np.sqrt(mse) * np.sqrt(one_minus_h))
    return h, cooks, std_resid


def winsor_fit_tau(
    y: pd.Series,
    design: pd.DataFrame,
    cov_type: str,
    use_t_fit: bool,
    winsor_q: Optional[float],
) -> Optional[float]:
    """Refit OLS on winsorized outcome and return treatment coefficient."""
    if winsor_q is None:
        return None
    q = float(winsor_q)
    if not (0.0 < q < 0.5):
        return None
    yv = y.to_numpy(dtype=float)
    lo, hi = np.quantile(yv, [q, 1.0 - q])
    y_w = np.clip(yv, lo, hi)
    try:
        res = sm.OLS(y_w, design).fit(cov_type=cov_type, use_t=use_t_fit)
    except Exception:
        return None
    params = np.asarray(res.params, dtype=float)
    if params.size < 2:
        return None
    tau_w = float(params[1])
    return tau_w if np.isfinite(tau_w) else None


def run_regression_checks(
    y: pd.Series,
    design: pd.DataFrame,
    result: Any,
    result_naive: Any,
    cov_type: str,
    use_t_fit: bool,
    corr_near_one_tol: float,
    tiny_one_minus_h_tol: float,
    winsor_q: Optional[float],
) -> RegressionChecks:
    """Build a compact payload with design, residual, and influence refutation checks."""
    z = np.asarray(design, dtype=float)
    yv = y.to_numpy(dtype=float)
    n = int(len(yv))

    k, rank, full_rank, cond = design_matrix_checks(design)

    params = np.asarray(result.params, dtype=float)
    params_naive = np.asarray(result_naive.params, dtype=float)
    bse_naive = np.asarray(result_naive.bse, dtype=float)

    ate_adj = float(params[1])
    ate_naive = float(params_naive[1])
    ate_gap = float(ate_adj - ate_naive)
    se_naive = float(bse_naive[1])
    if np.isfinite(se_naive) and se_naive > 0.0:
        ate_gap_over_se_naive: Optional[float] = float(ate_gap / se_naive)
    else:
        ate_gap_over_se_naive = None

    main_cov_cols = [str(c) for c in design.columns if str(c).endswith("__centered")]
    x_main = design[main_cov_cols] if main_cov_cols else pd.DataFrame(index=design.index)
    p_main_covariates = int(x_main.shape[1])
    dup_pairs = (
        near_duplicate_corr_pairs(x_main, tol=float(corr_near_one_tol))
        if x_main.shape[1] > 0
        else []
    )
    vif = vif_from_corr(x_main)

    h, cooks, std_resid = leverage_and_cooks(y=yv, z=z, params=params)
    leverage_cutoff = float(2.0 * k / max(n, 1))
    cooks_cutoff = float(4.0 / max(n, 1))
    n_high_leverage = int(np.sum(h > leverage_cutoff))
    n_high_cooks = int(np.sum(cooks > cooks_cutoff))

    one_minus_h = 1.0 - h
    min_one_minus_h = float(np.min(one_minus_h))
    n_tiny_one_minus_h = int(np.sum(one_minus_h < float(tiny_one_minus_h_tol)))

    resid = yv - z @ params
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if mad > 0.0:
        resid_scale_mad = float(1.4826 * mad)
    else:
        ddof = 1 if n > 1 else 0
        resid_scale_mad = float(np.std(resid, ddof=ddof))

    n_std_resid_gt_3 = int(np.sum(np.abs(std_resid) > 3.0))
    n_std_resid_gt_4 = int(np.sum(np.abs(std_resid) > 4.0))
    max_abs_std_resid = float(np.max(np.abs(std_resid))) if n > 0 else float("nan")

    ate_adj_winsor = winsor_fit_tau(
        y=y,
        design=design,
        cov_type=cov_type,
        use_t_fit=use_t_fit,
        winsor_q=winsor_q,
    )
    if ate_adj_winsor is not None and np.isfinite(ate_adj_winsor):
        ate_adj_winsor_gap: Optional[float] = float(ate_adj_winsor - ate_adj)
    else:
        ate_adj_winsor_gap = None

    return RegressionChecks(
        ate_naive=ate_naive,
        ate_adj=ate_adj,
        ate_gap=ate_gap,
        ate_gap_over_se_naive=ate_gap_over_se_naive,
        k=k,
        rank=rank,
        full_rank=full_rank,
        condition_number=cond,
        p_main_covariates=p_main_covariates,
        near_duplicate_pairs=dup_pairs,
        vif=vif,
        resid_scale_mad=resid_scale_mad,
        n_std_resid_gt_3=n_std_resid_gt_3,
        n_std_resid_gt_4=n_std_resid_gt_4,
        max_abs_std_resid=max_abs_std_resid,
        max_leverage=float(np.max(h)),
        leverage_cutoff=leverage_cutoff,
        n_high_leverage=n_high_leverage,
        max_cooks=float(np.max(cooks)),
        cooks_cutoff=cooks_cutoff,
        n_high_cooks=n_high_cooks,
        min_one_minus_h=min_one_minus_h,
        n_tiny_one_minus_h=n_tiny_one_minus_h,
        winsor_q=float(winsor_q) if winsor_q is not None else None,
        ate_adj_winsor=float(ate_adj_winsor) if ate_adj_winsor is not None else None,
        ate_adj_winsor_gap=ate_adj_winsor_gap,
    )


def _row(
    test_id: str,
    test: str,
    flag: str,
    value: Any,
    threshold: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "test_id": str(test_id),
        "test": str(test),
        "flag": str(flag),
        "value": value,
        "threshold": str(threshold),
        "message": str(message),
    }


def assumption_design_rank(checks: RegressionChecks) -> Dict[str, Any]:
    """Check that the design matrix is full rank."""
    value = f"rank={checks.rank}, k={checks.k}"
    if checks.full_rank:
        return _row(
            test_id="design_rank",
            test="Design rank",
            flag=FLAG_GREEN,
            value=value,
            threshold="rank == k",
            message="Design matrix is full rank.",
        )
    return _row(
        test_id="design_rank",
        test="Design rank",
        flag=FLAG_RED,
        value=value,
        threshold="rank == k",
        message="Rank deficiency detected; duplicate or perfectly collinear features likely.",
    )


def assumption_condition_number(
    checks: RegressionChecks,
    warn_threshold: float = 1e8,
    red_multiplier: float = 100.0,
) -> Dict[str, Any]:
    """Check global collinearity via condition number."""
    cond = float(checks.condition_number)
    red_threshold = float(warn_threshold) * float(red_multiplier)
    if not np.isfinite(cond):
        return _row(
            test_id="condition_number",
            test="Condition number",
            flag=FLAG_RED,
            value=cond,
            threshold=f"<= {warn_threshold:.3e}",
            message="Condition number is non-finite.",
        )
    if cond <= warn_threshold:
        return _row(
            test_id="condition_number",
            test="Condition number",
            flag=FLAG_GREEN,
            value=cond,
            threshold=f"<= {warn_threshold:.3e}",
            message="Condition number is within expected range.",
        )
    if cond <= red_threshold:
        return _row(
            test_id="condition_number",
            test="Condition number",
            flag=FLAG_YELLOW,
            value=cond,
            threshold=f"yellow: > {warn_threshold:.3e}, red: > {red_threshold:.3e}",
            message="Design is ill-conditioned; inference may be unstable.",
        )
    return _row(
        test_id="condition_number",
        test="Condition number",
        flag=FLAG_RED,
        value=cond,
        threshold=f"yellow: > {warn_threshold:.3e}, red: > {red_threshold:.3e}",
        message="Condition number is extremely high; estimates are likely unstable.",
    )


def assumption_near_duplicates(
    checks: RegressionChecks,
    red_pairs_threshold: int = 3,
) -> Dict[str, Any]:
    """Check near-duplicate centered covariate pairs."""
    n_pairs = int(len(checks.near_duplicate_pairs))
    if n_pairs == 0:
        return _row(
            test_id="near_duplicates",
            test="Near-duplicate covariates",
            flag=FLAG_GREEN,
            value=n_pairs,
            threshold="0 pairs",
            message="No near-duplicate centered covariates found.",
        )
    if n_pairs < int(red_pairs_threshold):
        return _row(
            test_id="near_duplicates",
            test="Near-duplicate covariates",
            flag=FLAG_YELLOW,
            value=n_pairs,
            threshold=f"red if >= {red_pairs_threshold}",
            message="Near-duplicate covariate pairs detected.",
        )
    return _row(
        test_id="near_duplicates",
        test="Near-duplicate covariates",
        flag=FLAG_RED,
        value=n_pairs,
        threshold=f"red if >= {red_pairs_threshold}",
        message="Multiple near-duplicate covariate pairs detected.",
    )


def assumption_vif(
    checks: RegressionChecks,
    warn_threshold: float = 20.0,
    red_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """Check VIF from centered main-effect covariates."""
    if checks.vif is None or len(checks.vif) == 0:
        has_multiple_covariates = int(checks.p_main_covariates) >= 2
        has_near_duplicates = len(checks.near_duplicate_pairs) > 0
        if has_multiple_covariates or has_near_duplicates:
            return _row(
                test_id="vif",
                test="Variance inflation factor",
                flag=FLAG_YELLOW,
                value=np.nan,
                threshold=f"<= {warn_threshold:.3g}",
                message="VIF unavailable; correlation matrix may be singular/unstable.",
            )
        return _row(
            test_id="vif",
            test="Variance inflation factor",
            flag=FLAG_GREEN,
            value=np.nan,
            threshold=f"<= {warn_threshold:.3g}",
            message="VIF not applicable (fewer than two usable covariates).",
        )

    max_vif = float(max(checks.vif.values()))
    red_threshold = float(warn_threshold) * float(red_multiplier)
    if not np.isfinite(max_vif):
        return _row(
            test_id="vif",
            test="Variance inflation factor",
            flag=FLAG_RED,
            value=max_vif,
            threshold=f"<= {warn_threshold:.3g}",
            message="VIF is non-finite.",
        )
    if max_vif <= warn_threshold:
        return _row(
            test_id="vif",
            test="Variance inflation factor",
            flag=FLAG_GREEN,
            value=max_vif,
            threshold=f"<= {warn_threshold:.3g}",
            message="VIF is within expected range.",
        )
    if max_vif <= red_threshold:
        return _row(
            test_id="vif",
            test="Variance inflation factor",
            flag=FLAG_YELLOW,
            value=max_vif,
            threshold=f"yellow: > {warn_threshold:.3g}, red: > {red_threshold:.3g}",
            message="High multicollinearity signal from VIF.",
        )
    return _row(
        test_id="vif",
        test="Variance inflation factor",
        flag=FLAG_RED,
        value=max_vif,
        threshold=f"yellow: > {warn_threshold:.3g}, red: > {red_threshold:.3g}",
        message="Very large VIF indicates severe multicollinearity.",
    )


def assumption_ate_gap(
    checks: RegressionChecks,
    yellow_threshold: float = 2.0,
    red_threshold: float = 2.5,
) -> Dict[str, Any]:
    """Check adjusted-vs-naive ATE gap relative to naive SE."""
    score = checks.ate_gap_over_se_naive
    if score is None or not np.isfinite(score):
        return _row(
            test_id="ate_gap",
            test="Adjusted vs naive ATE",
            flag=FLAG_YELLOW,
            value=np.nan,
            threshold=f"|gap/SE_naive| <= {yellow_threshold:.2f}",
            message="ATE gap scaling unavailable (naive SE missing or non-finite).",
        )

    abs_score = abs(float(score))
    if abs_score <= yellow_threshold:
        return _row(
            test_id="ate_gap",
            test="Adjusted vs naive ATE",
            flag=FLAG_GREEN,
            value=abs_score,
            threshold=f"yellow: > {yellow_threshold:.2f}, red: > {red_threshold:.2f}",
            message="Adjusted and naive ATE are reasonably aligned.",
        )
    if abs_score <= red_threshold:
        return _row(
            test_id="ate_gap",
            test="Adjusted vs naive ATE",
            flag=FLAG_YELLOW,
            value=abs_score,
            threshold=f"yellow: > {yellow_threshold:.2f}, red: > {red_threshold:.2f}",
            message="Material adjusted-vs-naive ATE gap; inspect influence and coding.",
        )
    return _row(
        test_id="ate_gap",
        test="Adjusted vs naive ATE",
        flag=FLAG_RED,
        value=abs_score,
        threshold=f"yellow: > {yellow_threshold:.2f}, red: > {red_threshold:.2f}",
        message="Large adjusted-vs-naive ATE gap; model fit or data issues likely.",
    )


def assumption_residual_tails(
    checks: RegressionChecks,
    yellow_abs_std_resid: float = 7.0,
    red_abs_std_resid: float = 10.0,
) -> Dict[str, Any]:
    """Check residual extremes using max standardized residual only."""
    max_abs = float(checks.max_abs_std_resid)

    if not np.isfinite(max_abs) or max_abs > red_abs_std_resid:
        return _row(
            test_id="residual_tails",
            test="Residual extremes",
            flag=FLAG_RED,
            value=f"max|std resid|={max_abs:.3g}",
            threshold=f"yellow > {yellow_abs_std_resid:.3g}, red > {red_abs_std_resid:.3g}",
            message="Extremely large standardized residuals; outliers likely dominate.",
        )
    if max_abs > yellow_abs_std_resid:
        return _row(
            test_id="residual_tails",
            test="Residual extremes",
            flag=FLAG_YELLOW,
            value=f"max|std resid|={max_abs:.3g}",
            threshold=f"yellow > {yellow_abs_std_resid:.3g}, red > {red_abs_std_resid:.3g}",
            message="Large standardized residuals; inspect influential points.",
        )
    return _row(
        test_id="residual_tails",
        test="Residual extremes",
        flag=FLAG_GREEN,
        value=f"max|std resid|={max_abs:.3g}",
        threshold=f"yellow > {yellow_abs_std_resid:.3g}, red > {red_abs_std_resid:.3g}",
        message="Residual extremes look reasonable.",
    )


def assumption_leverage(
    checks: RegressionChecks,
    yellow_multiplier: float = 5.0,
    red_multiplier: float = 10.0,
    red_floor: float = 0.5,
) -> Dict[str, Any]:
    """Check leverage concentration."""
    max_h = float(checks.max_leverage)
    cutoff = float(checks.leverage_cutoff)
    n_high = int(checks.n_high_leverage)
    red_rule = max(float(red_floor), float(red_multiplier) * cutoff)
    yellow_rule = float(yellow_multiplier) * cutoff

    if not np.isfinite(max_h) or max_h > red_rule:
        return _row(
            test_id="leverage",
            test="Leverage",
            flag=FLAG_RED,
            value=f"max_h={max_h:.4g}, n_high={n_high}",
            threshold=(
                f"yellow if max_h > {yellow_multiplier:.3g}*{cutoff:.4g}, "
                f"red if max_h > max({red_floor:.3g}, {red_multiplier:.3g}*{cutoff:.4g})"
            ),
            message="Extreme leverage points detected.",
        )
    if max_h > yellow_rule:
        return _row(
            test_id="leverage",
            test="Leverage",
            flag=FLAG_YELLOW,
            value=f"max_h={max_h:.4g}, n_high={n_high}",
            threshold=(
                f"yellow if max_h > {yellow_multiplier:.3g}*{cutoff:.4g}, "
                f"red if max_h > max({red_floor:.3g}, {red_multiplier:.3g}*{cutoff:.4g})"
            ),
            message="High-leverage observations are present.",
        )
    return _row(
        test_id="leverage",
        test="Leverage",
        flag=FLAG_GREEN,
        value=f"max_h={max_h:.4g}, n_high={n_high}",
        threshold=(
            f"yellow if max_h > {yellow_multiplier:.3g}*{cutoff:.4g}, "
            f"red if max_h > max({red_floor:.3g}, {red_multiplier:.3g}*{cutoff:.4g})"
        ),
        message="No high-leverage concentration detected.",
    )


def assumption_cooks(
    checks: RegressionChecks,
    yellow_threshold: float = 0.1,
    red_threshold: float = 1.0,
) -> Dict[str, Any]:
    """Check Cook's distance influence refutation checks."""
    max_cooks = float(checks.max_cooks)
    n_high = int(checks.n_high_cooks)

    if not np.isfinite(max_cooks) or max_cooks > red_threshold:
        return _row(
            test_id="cooks",
            test="Cook's distance",
            flag=FLAG_RED,
            value=f"max={max_cooks:.4g}, n_high={n_high}",
            threshold=f"yellow if max Cook's > {yellow_threshold:.3g}, red if > {red_threshold:.3g}",
            message="Strongly influential observations detected.",
        )
    if max_cooks > yellow_threshold:
        return _row(
            test_id="cooks",
            test="Cook's distance",
            flag=FLAG_YELLOW,
            value=f"max={max_cooks:.4g}, n_high={n_high}",
            threshold=f"yellow if max Cook's > {yellow_threshold:.3g}, red if > {red_threshold:.3g}",
            message="Potentially influential observations detected.",
        )
    return _row(
        test_id="cooks",
        test="Cook's distance",
        flag=FLAG_GREEN,
        value=f"max={max_cooks:.4g}, n_high={n_high}",
        threshold=f"yellow if max Cook's > {yellow_threshold:.3g}, red if > {red_threshold:.3g}",
        message="No strong influence signal from Cook's distance.",
    )


def assumption_hc23_stability(
    checks: RegressionChecks,
    cov_type: str,
    tiny_one_minus_h_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Check HC2/HC3 stability when leverage terms approach one."""
    cov_upper = str(cov_type).strip().upper()
    if cov_upper not in {"HC2", "HC3"}:
        return _row(
            test_id="hc23_stability",
            test="HC2/HC3 stability",
            flag=FLAG_GREEN,
            value="not_applicable",
            threshold="applies only to HC2/HC3",
            message="Covariance type is not HC2/HC3.",
        )

    min_one_minus_h = float(checks.min_one_minus_h)
    n_tiny = int(checks.n_tiny_one_minus_h)
    if n_tiny > 0 or min_one_minus_h < tiny_one_minus_h_tol:
        return _row(
            test_id="hc23_stability",
            test="HC2/HC3 stability",
            flag=FLAG_RED,
            value=f"min(1-h)={min_one_minus_h:.3e}, n_tiny={n_tiny}",
            threshold=f"red if min(1-h) < {tiny_one_minus_h_tol:.1e}",
            message="HC2/HC3 denominator instability detected.",
        )
    if min_one_minus_h < 100.0 * tiny_one_minus_h_tol:
        return _row(
            test_id="hc23_stability",
            test="HC2/HC3 stability",
            flag=FLAG_YELLOW,
            value=f"min(1-h)={min_one_minus_h:.3e}, n_tiny={n_tiny}",
            threshold=f"yellow if min(1-h) < {100.0 * tiny_one_minus_h_tol:.1e}",
            message="HC2/HC3 denominator is small; monitor stability.",
        )
    return _row(
        test_id="hc23_stability",
        test="HC2/HC3 stability",
        flag=FLAG_GREEN,
        value=f"min(1-h)={min_one_minus_h:.3e}, n_tiny={n_tiny}",
        threshold=f"min(1-h) >= {100.0 * tiny_one_minus_h_tol:.1e}",
        message="HC2/HC3 stability check passed.",
    )


def assumption_winsor_sensitivity(
    checks: RegressionChecks,
    winsor_reference_se: Optional[float] = None,
    yellow_sigma: float = 1.0,
    red_sigma: float = 2.0,
    yellow_ratio: float = 0.10,
    red_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Check sensitivity of adjusted ATE to winsorized-outcome refit."""
    gap = checks.ate_adj_winsor_gap
    if gap is None or not np.isfinite(gap):
        return _row(
            test_id="winsor_sensitivity",
            test="Winsor sensitivity",
            flag=FLAG_GREEN,
            value=np.nan,
            threshold="winsor_q enabled",
            message="Winsor sensitivity was not computed.",
        )

    abs_gap = abs(float(gap))
    if winsor_reference_se is not None and np.isfinite(winsor_reference_se) and winsor_reference_se > 0.0:
        score = abs_gap / float(winsor_reference_se)
        if score > red_sigma:
            return _row(
                test_id="winsor_sensitivity",
                test="Winsor sensitivity",
                flag=FLAG_RED,
                value=score,
                threshold=f"yellow: > {yellow_sigma:.2f} SE, red: > {red_sigma:.2f} SE",
                message="Winsorized refit materially changes ATE.",
            )
        if score > yellow_sigma:
            return _row(
                test_id="winsor_sensitivity",
                test="Winsor sensitivity",
                flag=FLAG_YELLOW,
                value=score,
                threshold=f"yellow: > {yellow_sigma:.2f} SE, red: > {red_sigma:.2f} SE",
                message="Winsorized refit changes ATE moderately.",
            )
        return _row(
            test_id="winsor_sensitivity",
            test="Winsor sensitivity",
            flag=FLAG_GREEN,
            value=score,
            threshold=f"yellow: > {yellow_sigma:.2f} SE, red: > {red_sigma:.2f} SE",
            message="Winsorized refit is close to baseline ATE.",
        )

    denom = max(abs(float(checks.ate_adj)), 1e-12)
    ratio = abs_gap / denom
    if ratio > red_ratio:
        return _row(
            test_id="winsor_sensitivity",
            test="Winsor sensitivity",
            flag=FLAG_RED,
            value=ratio,
            threshold=f"yellow: > {yellow_ratio:.2f}, red: > {red_ratio:.2f}",
            message="Winsorized refit materially changes ATE.",
        )
    if ratio > yellow_ratio:
        return _row(
            test_id="winsor_sensitivity",
            test="Winsor sensitivity",
            flag=FLAG_YELLOW,
            value=ratio,
            threshold=f"yellow: > {yellow_ratio:.2f}, red: > {red_ratio:.2f}",
            message="Winsorized refit changes ATE moderately.",
        )
    return _row(
        test_id="winsor_sensitivity",
        test="Winsor sensitivity",
        flag=FLAG_GREEN,
        value=ratio,
        threshold=f"yellow: > {yellow_ratio:.2f}, red: > {red_ratio:.2f}",
        message="Winsorized refit is close to baseline ATE.",
    )


def regression_assumption_rows_from_checks(
    checks: RegressionChecks,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 1e8,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-8,
    winsor_reference_se: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """Run all CUPED regression assumption tests and return row payloads."""
    return [
        assumption_design_rank(checks=checks),
        assumption_condition_number(
            checks=checks,
            warn_threshold=condition_number_warn_threshold,
        ),
        assumption_near_duplicates(checks=checks),
        assumption_vif(checks=checks, warn_threshold=vif_warn_threshold),
        assumption_ate_gap(checks=checks),
        assumption_residual_tails(checks=checks),
        assumption_leverage(checks=checks),
        assumption_cooks(checks=checks),
        assumption_hc23_stability(
            checks=checks,
            cov_type=cov_type,
            tiny_one_minus_h_tol=tiny_one_minus_h_tol,
        ),
        assumption_winsor_sensitivity(
            checks=checks,
            winsor_reference_se=winsor_reference_se,
        ),
    ]


def regression_assumptions_table_from_checks(
    checks: RegressionChecks,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 1e8,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-8,
    winsor_reference_se: Optional[float] = None,
) -> pd.DataFrame:
    """Return a table of GREEN/YELLOW/RED assumption flags from checks payload."""
    rows = regression_assumption_rows_from_checks(
        checks=checks,
        cov_type=cov_type,
        condition_number_warn_threshold=condition_number_warn_threshold,
        vif_warn_threshold=vif_warn_threshold,
        tiny_one_minus_h_tol=tiny_one_minus_h_tol,
        winsor_reference_se=winsor_reference_se,
    )
    return pd.DataFrame(rows)


def _approx_se_from_estimate(estimate: CausalEstimate) -> Optional[float]:
    if (
        estimate.value is None
        or estimate.ci_upper_absolute is None
        or estimate.ci_lower_absolute is None
        or estimate.alpha is None
    ):
        return None
    alpha = float(estimate.alpha)
    if not (0.0 < alpha < 1.0):
        return None
    z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    if not np.isfinite(z) or z <= 0.0:
        return None
    half_width = max(
        abs(float(estimate.ci_upper_absolute) - float(estimate.value)),
        abs(float(estimate.value) - float(estimate.ci_lower_absolute)),
    )
    se = half_width / z
    return float(se) if np.isfinite(se) and se > 0.0 else None


def _validate_estimate_matches_data(data: CausalData, estimate: CausalEstimate) -> None:
    """Validate that a CUPED estimate aligns with provided causal data metadata."""
    if str(estimate.treatment) != str(data.treatment_name):
        raise ValueError(
            "estimate.treatment must match data.treatment_name "
            f"({estimate.treatment!r} != {data.treatment_name!r})."
        )

    if str(estimate.outcome) != str(data.outcome_name):
        raise ValueError(
            "estimate.outcome must match data.outcome_name "
            f"({estimate.outcome!r} != {data.outcome_name!r})."
        )

    missing_confounders = [name for name in estimate.confounders if name not in data.df.columns]
    if missing_confounders:
        raise ValueError(
            "estimate.confounders are missing in data.df: "
            + ", ".join(sorted(map(str, missing_confounders)))
        )


def regression_assumptions_table_from_diagnostic_data(
    diagnostic_data: CUPEDDiagnosticData,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 1e8,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-8,
    winsor_reference_se: Optional[float] = None,
) -> pd.DataFrame:
    """Build assumption table from ``CUPEDDiagnosticData`` payload."""
    checks = diagnostic_data.regression_checks
    if checks is None:
        raise ValueError(
            "diagnostic_data.regression_checks is missing. "
            "Fit with run_regression_checks=True first."
        )
    return regression_assumptions_table_from_checks(
        checks=checks,
        cov_type=cov_type,
        condition_number_warn_threshold=condition_number_warn_threshold,
        vif_warn_threshold=vif_warn_threshold,
        tiny_one_minus_h_tol=tiny_one_minus_h_tol,
        winsor_reference_se=winsor_reference_se,
    )


def regression_assumptions_table_from_estimate(
    data_or_estimate: CausalData | CausalEstimate,
    estimate: Optional[CausalEstimate] = None,
    style_regression_assumptions_table: Optional[Callable[[pd.DataFrame], Any]] = None,
    cov_type: Optional[str] = None,
    condition_number_warn_threshold: float = 1e8,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-8,
) -> Any:
    """
    Build assumptions table from a CUPED estimate.

    Supports both call styles:
    1) ``regression_assumptions_table_from_estimate(estimate, ...)``
    2) ``regression_assumptions_table_from_estimate(data, estimate, ...)``
    """
    data: Optional[CausalData]
    estimate_eff: CausalEstimate

    if estimate is None:
        if not isinstance(data_or_estimate, CausalEstimate):
            raise TypeError(
                "Expected CausalEstimate when `estimate` is omitted. "
                "Use either (estimate, ...) or (data, estimate, ...)."
            )
        data = None
        estimate_eff = data_or_estimate
    else:
        if not isinstance(data_or_estimate, CausalData):
            raise TypeError(
                "Expected CausalData as first argument when `estimate` is provided."
            )
        data = data_or_estimate
        estimate_eff = estimate
        _validate_estimate_matches_data(data=data, estimate=estimate_eff)

    diagnostic_data = estimate_eff.diagnostic_data
    if not isinstance(diagnostic_data, CUPEDDiagnosticData):
        raise ValueError(
            "estimate.diagnostic_data must be CUPEDDiagnosticData with regression checks."
        )

    cov_type_eff = cov_type
    if cov_type_eff is None:
        cov_type_eff = str(estimate_eff.model_options.get("cov_type", "HC2"))

    se_ref = _approx_se_from_estimate(estimate_eff)
    table = regression_assumptions_table_from_diagnostic_data(
        diagnostic_data=diagnostic_data,
        cov_type=str(cov_type_eff),
        condition_number_warn_threshold=condition_number_warn_threshold,
        vif_warn_threshold=vif_warn_threshold,
        tiny_one_minus_h_tol=tiny_one_minus_h_tol,
        winsor_reference_se=se_ref,
    )
    transform = style_regression_assumptions_table or (lambda value: value)
    return transform(table)


def regression_assumptions_table_from_data(
    data: CausalData,
    covariates: Sequence[str],
    model_kwargs: Optional[Dict[str, Any]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Fit CUPED on ``CausalData`` and return the assumptions flag table."""
    from causalis.scenarios.cuped.model import CUPEDModel
    from causalis.scenarios.cuped.refutation.config import CUPEDRefutationConfig

    model_kwargs = dict(model_kwargs or {})
    fit_kwargs = dict(fit_kwargs or {})

    model_kwargs.setdefault(
        "refutation_config",
        CUPEDRefutationConfig(run_regression_checks=True, check_action="ignore"),
    )

    model = CUPEDModel(**model_kwargs).fit(
        data=data,
        covariates=covariates,
        **fit_kwargs,
    )
    estimate = model.estimate(diagnostic_data=True)
    return regression_assumptions_table_from_estimate(
        data_or_estimate=data,
        estimate=estimate,
        cov_type=model.cov_type,
    )


def overall_assumption_flag(table: pd.DataFrame) -> str:
    """Return overall GREEN/YELLOW/RED status from an assumptions table."""
    if "flag" not in table.columns or table.empty:
        return FLAG_GREEN
    levels = [FLAG_LEVEL[_normalize_flag(flag)] for flag in table["flag"].tolist()]
    worst = max(levels) if levels else 0
    if worst == 2:
        return FLAG_RED
    if worst == 1:
        return FLAG_YELLOW
    return FLAG_GREEN


def style_regression_assumptions_table(table: pd.DataFrame):
    """Return pandas Styler with colored flag cells for notebook display."""
    if "flag" not in table.columns:
        return table.style

    def _style_flag(value: Any) -> str:
        color = FLAG_COLOR.get(_normalize_flag(value), "#6d6d6d")
        return f"background-color: {color}; color: white; font-weight: bold;"

    return table.style.map(_style_flag, subset=["flag"])
