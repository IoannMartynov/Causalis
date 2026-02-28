from __future__ import annotations

import warnings
from statistics import NormalDist
from typing import Any, Hashable, List, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.data_contracts.panel_estimate import PanelEstimate


class AugmentedSyntheticControl:
    """
    Augmented Synthetic Control Method (ASCM) for a single treated unit.

    This implementation uses a ridge-augmented donor-weight formulation:
    it first fits simplex-constrained SCM weights, then computes an augmented
    ridge solution (optionally constrained to sum to one).

    Interface:
    - fit(data: PanelDataSCM) -> self
    - estimate() -> PanelEstimate
    """

    def __init__(
        self,
        *,
        lambda_aug: float = 1.0,
        lambda_sc: float = 1e-6,
        max_iter: int = 2_000,
        tol: float = 1e-9,
        enforce_sum_to_one_augmented: bool = True,
        inference_policy: Literal["placebo"] = "placebo",
    ) -> None:
        self.lambda_aug = float(lambda_aug)
        self.lambda_sc = float(lambda_sc)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.enforce_sum_to_one_augmented = bool(enforce_sum_to_one_augmented)
        self.inference_policy = str(inference_policy)

        if not np.isfinite(self.lambda_aug) or self.lambda_aug < 0.0:
            raise ValueError("lambda_aug must be finite and >= 0.")
        if not np.isfinite(self.lambda_sc) or self.lambda_sc < 0.0:
            raise ValueError("lambda_sc must be finite and >= 0.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and > 0.")
        if self.inference_policy != "placebo":
            raise ValueError("inference_policy must be 'placebo'.")

        self._is_fitted: bool = False
        self._data: PanelDataSCM | None = None

    @staticmethod
    def _rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        """Project a vector onto the probability simplex {w >= 0, sum(w)=1}."""
        vec = np.asarray(v, dtype=float).reshape(-1)
        if vec.size < 1:
            raise ValueError("Simplex projection requires at least one coefficient.")
        if not np.isfinite(vec).all():
            raise ValueError("Simplex projection input must be finite.")

        sorted_vec = np.sort(vec)[::-1]
        cssv = np.cumsum(sorted_vec) - 1.0
        idx = np.arange(1, vec.size + 1, dtype=float)
        positive = sorted_vec - (cssv / idx) > 0.0
        if not bool(np.any(positive)):
            return np.full(vec.size, 1.0 / float(vec.size), dtype=float)

        rho = int(np.nonzero(positive)[0][-1])
        theta = float(cssv[rho] / float(rho + 1))
        projected = np.maximum(vec - theta, 0.0)
        proj_sum = float(np.sum(projected))
        if not np.isfinite(proj_sum) or proj_sum <= 0.0:
            return np.full(vec.size, 1.0 / float(vec.size), dtype=float)
        return projected / proj_sum

    def _fit_simplex_weights_projected_gradient(
        self,
        *,
        gram: np.ndarray,
        rhs: np.ndarray,
        w_init: np.ndarray,
    ) -> np.ndarray:
        """Convex fallback solver for simplex-constrained SC weights."""
        gram_arr = np.asarray(gram, dtype=float)
        rhs_arr = np.asarray(rhs, dtype=float)
        w = self._project_to_simplex(np.asarray(w_init, dtype=float))

        if gram_arr.shape[0] != gram_arr.shape[1]:
            raise ValueError("gram must be square in simplex fallback.")
        if rhs_arr.shape != (gram_arr.shape[0],):
            raise ValueError("rhs must have shape (n_donors,) in simplex fallback.")
        if not np.isfinite(gram_arr).all() or not np.isfinite(rhs_arr).all():
            raise RuntimeError("SC simplex fallback received non-finite optimization terms.")

        try:
            eigvals = np.linalg.eigvalsh(gram_arr)
            lmax = float(np.max(eigvals)) if eigvals.size > 0 else 0.0
        except np.linalg.LinAlgError:
            lmax = float(np.linalg.norm(gram_arr, ord=2))
        lipschitz = float(max(2.0 * lmax, 1e-12))
        step = 1.0 / lipschitz

        tol_eff = float(max(self.tol, 1e-10))
        max_iter_pg = int(max(2_000, 5 * self.max_iter))
        for _ in range(max_iter_pg):
            grad = 2.0 * (gram_arr @ w - rhs_arr)
            if not np.isfinite(grad).all():
                break
            w_next = self._project_to_simplex(w - step * grad)
            if float(np.max(np.abs(w_next - w))) <= tol_eff:
                return w_next
            w = w_next

        w = self._project_to_simplex(w)
        if not np.isfinite(w).all():
            raise RuntimeError("SC simplex fallback returned non-finite donor weights.")
        return w

    @staticmethod
    def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(a, b, rcond=None)[0]

    @staticmethod
    def _z_critical(alpha: float) -> float:
        if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be finite and in (0, 1).")
        return float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))

    @classmethod
    def _t_critical(cls, *, alpha: float, df: int) -> float:
        if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be finite and in (0, 1).")
        df_eff = max(int(df), 1)
        critical = float(student_t.ppf(1.0 - float(alpha) / 2.0, df=df_eff))
        if not np.isfinite(critical) or critical <= 0.0:
            critical = cls._z_critical(alpha=alpha)
        return critical

    @classmethod
    def _two_sided_t_p_value(cls, *, stat_abs: float, df: int) -> float:
        if not np.isfinite(stat_abs):
            return 1.0
        df_eff = max(int(df), 1)
        p_value = float(2.0 * (1.0 - student_t.cdf(float(abs(stat_abs)), df=df_eff)))
        if not np.isfinite(p_value):
            p_value = float(2.0 * (1.0 - NormalDist().cdf(float(abs(stat_abs)))))
        return float(np.clip(p_value, 0.0, 1.0))

    @staticmethod
    def _newey_west_long_run_variance(
        residuals: np.ndarray,
        *,
        max_lag: int | None = None,
    ) -> tuple[float, int]:
        resid = np.asarray(residuals, dtype=float)
        if resid.size < 1:
            raise ValueError("residuals must contain at least one value.")

        centered = resid - float(np.mean(resid))
        n = int(centered.size)

        if max_lag is None:
            lag = int(np.floor(4.0 * (float(n) / 100.0) ** (2.0 / 9.0)))
        else:
            lag = int(max_lag)
        lag = int(np.clip(lag, 0, max(n - 1, 0)))

        gamma0 = float(np.dot(centered, centered) / float(n))
        lrv = float(gamma0)
        for k in range(1, lag + 1):
            weight = 1.0 - (float(k) / float(lag + 1))
            gamma_k = float(np.dot(centered[k:], centered[:-k]) / float(n))
            lrv += 2.0 * weight * gamma_k

        return float(max(lrv, 0.0)), lag

    @classmethod
    def _att_inference_from_pre_residuals(
        cls,
        *,
        post_gap: np.ndarray,
        pre_resid: np.ndarray,
        post_synthetic: np.ndarray,
        alpha: float = 0.05,
        relative_baseline_min: float = 1e-8,
        hac_max_lag: int | None = None,
    ) -> dict[str, Any]:
        post_gap_arr = np.asarray(post_gap, dtype=float)
        pre_resid_arr = np.asarray(pre_resid, dtype=float)
        post_synthetic_arr = np.asarray(post_synthetic, dtype=float)

        if post_gap_arr.size < 1:
            raise ValueError("post_gap must contain at least one post-treatment period.")
        if pre_resid_arr.size < 1:
            raise ValueError("pre_resid must contain at least one pre-treatment period.")
        if post_synthetic_arr.size != post_gap_arr.size:
            raise ValueError("post_synthetic must have the same length as post_gap.")
        if hac_max_lag is not None and int(hac_max_lag) < 0:
            raise ValueError("hac_max_lag must be None or a non-negative integer.")

        att = float(np.mean(post_gap_arr))
        n_pre = int(pre_resid_arr.size)
        n_post = int(post_gap_arr.size)

        pre_lrv, hac_lag_used = cls._newey_west_long_run_variance(
            pre_resid_arr,
            max_lag=hac_max_lag,
        )
        se_abs = float(np.sqrt(pre_lrv / float(n_post)))
        df_t = max(n_pre - 1, 1)
        critical = cls._t_critical(alpha=alpha, df=df_t)
        ci_low_abs = float(att - critical * se_abs)
        ci_high_abs = float(att + critical * se_abs)
        if ci_low_abs > ci_high_abs:
            ci_low_abs, ci_high_abs = ci_high_abs, ci_low_abs

        if se_abs > 0.0 and np.isfinite(se_abs):
            t_stat = float(abs(att) / se_abs)
            p_value = cls._two_sided_t_p_value(stat_abs=t_stat, df=df_t)
        else:
            p_value = 1.0 if abs(att) < 1e-15 else 0.0
        p_value = float(np.clip(p_value, 0.0, 1.0))
        is_significant = bool(p_value < float(alpha))

        baseline = float(np.mean(post_synthetic_arr))
        value_relative: float | None = None
        ci_low_rel: float | None = None
        ci_high_rel: float | None = None
        if np.isfinite(baseline) and abs(baseline) >= float(relative_baseline_min):
            value_relative = float((att / baseline) * 100.0)
            se_rel = float((100.0 / abs(baseline)) * se_abs)
            ci_low_rel = float(value_relative - critical * se_rel)
            ci_high_rel = float(value_relative + critical * se_rel)
            if ci_low_rel > ci_high_rel:
                ci_low_rel, ci_high_rel = ci_high_rel, ci_low_rel

        return {
            "alpha": float(alpha),
            "se_absolute": se_abs,
            "ci_lower_absolute": ci_low_abs,
            "ci_upper_absolute": ci_high_abs,
            "p_value": p_value,
            "is_significant": is_significant,
            "baseline_post_synthetic": baseline,
            "value_relative": value_relative,
            "ci_lower_relative": ci_low_rel,
            "ci_upper_relative": ci_high_rel,
            "se_method": "pre_residual_hac_newey_west",
            "p_value_method": "pre_residual_hac_t",
            "critical_value": critical,
            "df_t": int(df_t),
            "pre_resid_n": int(n_pre),
            "hac_max_lag": int(hac_lag_used),
            "pre_long_run_variance": float(pre_lrv),
            "p_value_pre_residual": p_value,
            "is_significant_pre_residual": is_significant,
        }

    @staticmethod
    def _placebo_inverted_att_ci(
        *,
        treated_att: float,
        placebo_att: np.ndarray,
        alpha: float,
    ) -> tuple[float | None, float | None, float | None, bool]:
        if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be finite and in (0, 1).")

        placebo_att_arr = np.asarray(placebo_att, dtype=float)
        n_placebos = int(placebo_att_arr.size)
        if n_placebos < 1:
            return None, None, None, False

        min_possible_p = float(1.0 / float(n_placebos + 1))
        if float(alpha) < min_possible_p:
            return None, None, min_possible_p, True

        m = int(np.floor(float(alpha) * float(n_placebos + 1) - 1.0))
        m = int(np.clip(m, 0, n_placebos - 1))
        sorted_abs = np.sort(np.abs(placebo_att_arr))
        q = float(sorted_abs[n_placebos - 1 - m])
        ci_low_abs = float(treated_att - q)
        ci_high_abs = float(treated_att + q)
        if ci_low_abs > ci_high_abs:
            ci_low_abs, ci_high_abs = ci_high_abs, ci_low_abs
        return ci_low_abs, ci_high_abs, min_possible_p, False

    @classmethod
    def _summarize_placebo_distribution(
        cls,
        *,
        treated_gap: np.ndarray,
        n_pre: int,
        placebo_atts: List[float],
        placebo_rmspe_ratios: List[float],
        failed: int,
        alpha: float,
    ) -> dict[str, Any]:
        treated_gap_arr = np.asarray(treated_gap, dtype=float)
        n_placebos = int(len(placebo_atts))
        result: dict[str, Any] = {
            "available": False,
            "n_placebos": n_placebos,
            "n_failed_placebos": int(failed),
            "p_value_att": None,
            "p_value_rmspe_ratio": None,
            "treated_rmspe_ratio": None,
            "ci_lower_absolute": None,
            "ci_upper_absolute": None,
            "min_possible_p": None,
            "placebo_ci_is_unbounded": False,
            "placebo_att_distribution": [],
            "placebo_rmspe_ratio_distribution": [],
            "ci_method": "placebo_inversion_att",
            "p_value_method": "placebo_in_space_att",
        }
        if n_placebos < 1:
            return result

        treated_att = float(np.mean(treated_gap_arr[n_pre:]))
        treated_pre_rmspe = cls._rmse(treated_gap_arr[:n_pre])
        treated_post_rmspe = cls._rmse(treated_gap_arr[n_pre:])
        treated_rmspe_ratio = float(treated_post_rmspe / max(treated_pre_rmspe, 1e-12))

        placebo_att_arr = np.asarray(placebo_atts, dtype=float)
        placebo_ratio_arr = np.asarray(placebo_rmspe_ratios, dtype=float)
        p_value_att = float(
            (1.0 + float(np.sum(np.abs(placebo_att_arr) >= abs(treated_att))))
            / float(n_placebos + 1)
        )
        p_value_ratio = float(
            (1.0 + float(np.sum(placebo_ratio_arr >= treated_rmspe_ratio)))
            / float(n_placebos + 1)
        )
        ci_low_abs, ci_high_abs, min_possible_p, ci_unbounded = cls._placebo_inverted_att_ci(
            treated_att=treated_att,
            placebo_att=placebo_att_arr,
            alpha=alpha,
        )

        result.update(
            {
                "available": True,
                "p_value_att": float(np.clip(p_value_att, 0.0, 1.0)),
                "p_value_rmspe_ratio": float(np.clip(p_value_ratio, 0.0, 1.0)),
                "treated_rmspe_ratio": treated_rmspe_ratio,
                "ci_lower_absolute": ci_low_abs,
                "ci_upper_absolute": ci_high_abs,
                "min_possible_p": min_possible_p,
                "placebo_ci_is_unbounded": bool(ci_unbounded),
                "placebo_att_distribution": placebo_att_arr.tolist(),
                "placebo_rmspe_ratio_distribution": placebo_ratio_arr.tolist(),
            }
        )
        return result

    @staticmethod
    def _inference_from_placebo(
        *,
        att: float,
        baseline_post_synthetic: float,
        placebo: dict[str, Any],
        alpha: float = 0.05,
        relative_baseline_min: float = 1e-8,
    ) -> dict[str, Any]:
        p_value = placebo.get("p_value_att")
        p_value = float(p_value) if p_value is not None and np.isfinite(p_value) else None
        is_significant = None if p_value is None else bool(p_value <= float(alpha))
        p_value_fit_adjusted = placebo.get("p_value_rmspe_ratio")
        p_value_fit_adjusted = (
            float(p_value_fit_adjusted)
            if p_value_fit_adjusted is not None and np.isfinite(p_value_fit_adjusted)
            else None
        )
        is_significant_fit_adjusted = (
            None
            if p_value_fit_adjusted is None
            else bool(p_value_fit_adjusted <= float(alpha))
        )
        fit_adjusted_warning: str | None = None
        if is_significant is True and is_significant_fit_adjusted is False:
            fit_adjusted_warning = "Pre-fit is weak; consider RMSPE-ratio placebo result."

        ci_low_abs = placebo.get("ci_lower_absolute")
        ci_high_abs = placebo.get("ci_upper_absolute")
        baseline = float(baseline_post_synthetic)
        value_relative: float | None = None
        ci_low_rel: float | None = None
        ci_high_rel: float | None = None
        if np.isfinite(baseline) and abs(baseline) >= float(relative_baseline_min):
            value_relative = float((float(att) / baseline) * 100.0)
            if ci_low_abs is not None and ci_high_abs is not None:
                rel_low = float((float(ci_low_abs) / baseline) * 100.0)
                rel_high = float((float(ci_high_abs) / baseline) * 100.0)
                ci_low_rel = float(min(rel_low, rel_high))
                ci_high_rel = float(max(rel_low, rel_high))

        return {
            "alpha": float(alpha),
            "p_value": p_value,
            "is_significant": is_significant,
            "p_value_method": placebo.get("p_value_method", "placebo_in_space_att"),
            "ci_method": placebo.get("ci_method", "placebo_inversion_att"),
            "ci_lower_absolute": ci_low_abs,
            "ci_upper_absolute": ci_high_abs,
            "baseline_post_synthetic": baseline,
            "value_relative": value_relative,
            "ci_lower_relative": ci_low_rel,
            "ci_upper_relative": ci_high_rel,
            "n_placebos": placebo.get("n_placebos"),
            "n_failed_placebos": placebo.get("n_failed_placebos"),
            "p_value_placebo_att": placebo.get("p_value_att"),
            "p_value_placebo_rmspe_ratio": placebo.get("p_value_rmspe_ratio"),
            "is_significant_fit_adjusted": is_significant_fit_adjusted,
            "fit_adjusted_warning": fit_adjusted_warning,
            "treated_rmspe_ratio": placebo.get("treated_rmspe_ratio"),
            "placebo_min_possible_p": placebo.get("min_possible_p"),
            "placebo_ci_is_unbounded": placebo.get("placebo_ci_is_unbounded"),
            "placebo_att_distribution": placebo.get("placebo_att_distribution"),
            "placebo_rmspe_ratio_distribution": placebo.get("placebo_rmspe_ratio_distribution"),
        }

    def _placebo_in_space_inference(
        self,
        *,
        outcomes_by_unit: np.ndarray,
        treated_gap: np.ndarray,
        n_pre: int,
        alpha: float = 0.05,
        pre_weight_sqrt: np.ndarray | None = None,
        use_augmented: bool,
    ) -> dict[str, Any]:
        outcomes = np.asarray(outcomes_by_unit, dtype=float)
        treated_gap_arr = np.asarray(treated_gap, dtype=float)
        n_units, n_times = outcomes.shape

        result: dict[str, Any] = {
            "available": False,
            "n_placebos": 0,
            "n_failed_placebos": 0,
            "p_value_att": None,
            "p_value_rmspe_ratio": None,
            "treated_rmspe_ratio": None,
            "ci_lower_absolute": None,
            "ci_upper_absolute": None,
            "min_possible_p": None,
            "placebo_ci_is_unbounded": False,
            "ci_method": "placebo_inversion_att",
            "p_value_method": "placebo_in_space_att",
        }

        if n_units < 3:
            return result
        if n_pre < 1 or n_pre >= n_times:
            return result
        if treated_gap_arr.size != n_times:
            raise ValueError("treated_gap must have length equal to outcomes_by_unit.shape[1].")

        if pre_weight_sqrt is None:
            pre_weight_arr = np.ones(n_pre, dtype=float)
        else:
            pre_weight_arr = np.asarray(pre_weight_sqrt, dtype=float)
            if pre_weight_arr.shape != (n_pre,):
                raise ValueError("pre_weight_sqrt must have shape (n_pre,).")
            if (pre_weight_arr <= 0.0).any() or (not np.isfinite(pre_weight_arr).all()):
                raise ValueError("pre_weight_sqrt must be finite and strictly positive.")

        placebo_atts: List[float] = []
        placebo_rmspe_ratios: List[float] = []
        failed = 0
        donor_indices = list(range(1, n_units))
        for pseudo_treated_idx in donor_indices:
            donor_pool_idx = [idx for idx in donor_indices if idx != pseudo_treated_idx]
            if len(donor_pool_idx) < 1:
                failed += 1
                continue

            y_pseudo_pre = outcomes[pseudo_treated_idx, :n_pre]
            x_pseudo_pre = outcomes[donor_pool_idx, :n_pre].T
            y_pseudo_pre_fit = y_pseudo_pre * pre_weight_arr
            x_pseudo_pre_fit = x_pseudo_pre * pre_weight_arr[:, None]
            try:
                w_sc_pseudo = self._fit_simplex_weights(
                    x0_pre=x_pseudo_pre_fit,
                    y1_pre=y_pseudo_pre_fit,
                )
                w_pseudo = (
                    self._augment_weights(
                        x0_pre=x_pseudo_pre_fit,
                        y1_pre=y_pseudo_pre_fit,
                        w_sc=w_sc_pseudo,
                    )
                    if use_augmented
                    else w_sc_pseudo
                )
            except Exception:
                failed += 1
                continue

            y_pseudo_all = outcomes[pseudo_treated_idx, :]
            x_pseudo_all = outcomes[donor_pool_idx, :].T
            gap_pseudo = y_pseudo_all - (x_pseudo_all @ w_pseudo)
            placebo_atts.append(float(np.mean(gap_pseudo[n_pre:])))
            pre_rmspe = self._rmse(gap_pseudo[:n_pre])
            post_rmspe = self._rmse(gap_pseudo[n_pre:])
            placebo_rmspe_ratios.append(float(post_rmspe / max(pre_rmspe, 1e-12)))

        if not placebo_atts:
            return result
        return self._summarize_placebo_distribution(
            treated_gap=treated_gap_arr,
            n_pre=n_pre,
            placebo_atts=placebo_atts,
            placebo_rmspe_ratios=placebo_rmspe_ratios,
            failed=failed,
            alpha=alpha,
        )

    def _fit_simplex_weights(self, x0_pre: np.ndarray, y1_pre: np.ndarray) -> np.ndarray:
        x0_pre_arr = np.asarray(x0_pre, dtype=float)
        y1_pre_arr = np.asarray(y1_pre, dtype=float)

        if x0_pre_arr.ndim != 2:
            raise ValueError("x0_pre must be a 2D array with shape (n_pre, n_donors).")
        if y1_pre_arr.ndim != 1:
            raise ValueError("y1_pre must be a 1D array with shape (n_pre,).")
        if x0_pre_arr.shape[0] != y1_pre_arr.shape[0]:
            raise ValueError(
                "x0_pre and y1_pre must share the same pre-treatment length."
            )
        if not np.isfinite(x0_pre_arr).all() or not np.isfinite(y1_pre_arr).all():
            raise ValueError("SC pre-treatment arrays must contain only finite values.")

        n_donors = x0_pre_arr.shape[1]
        if n_donors < 1:
            raise ValueError("SC weight optimization requires at least one donor.")
        w0 = np.full(n_donors, 1.0 / n_donors, dtype=float)
        gram = x0_pre_arr.T @ x0_pre_arr + self.lambda_sc * np.eye(n_donors, dtype=float)
        rhs = x0_pre_arr.T @ y1_pre_arr

        def objective(w: np.ndarray) -> float:
            return float((w @ gram @ w) - 2.0 * (rhs @ w))

        def gradient(w: np.ndarray) -> np.ndarray:
            return 2.0 * (gram @ w - rhs)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n_donors)]

        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        if bool(result.success) and result.x is not None and np.isfinite(result.x).all():
            w = self._project_to_simplex(np.asarray(result.x, dtype=float))
            if np.isfinite(w).all():
                return w

        try:
            w = self._fit_simplex_weights_projected_gradient(
                gram=gram,
                rhs=rhs,
                w_init=w0,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback path.
            raise RuntimeError(
                f"SC weight optimization failed: {result.message}"
            ) from exc
        return w

    def _augment_weights(self, x0_pre: np.ndarray, y1_pre: np.ndarray, w_sc: np.ndarray) -> np.ndarray:
        n_donors = x0_pre.shape[1]

        gram = x0_pre.T @ x0_pre + self.lambda_aug * np.eye(n_donors, dtype=float)
        rhs = x0_pre.T @ y1_pre + self.lambda_aug * w_sc
        w_aug = self._solve_linear(gram, rhs)

        if self.enforce_sum_to_one_augmented:
            ones = np.ones(n_donors, dtype=float)
            gram_inv_ones = self._solve_linear(gram, ones)
            denom = float(ones @ gram_inv_ones)
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                raise RuntimeError("ASCM augmented constraint system is ill-conditioned.")
            correction = gram_inv_ones * ((float(np.sum(w_aug)) - 1.0) / denom)
            w_aug = w_aug - correction

        return np.asarray(w_aug, dtype=float)

    @staticmethod
    def _validate_pre_post_times(
        *,
        pre_times: List[Any],
        post_times: List[Any],
        model_name: str,
    ) -> None:
        if len(pre_times) < 1:
            raise ValueError(f"{model_name}.fit() requires at least one pre-treatment period.")
        if len(post_times) < 1:
            raise ValueError(f"{model_name}.fit() requires at least one post-treatment period.")

        overlap = set(pre_times).intersection(post_times)
        if overlap:
            raise ValueError(
                f"{model_name}.fit() requires disjoint pre/post periods; overlapping times found: "
                f"{sorted(overlap)}."
            )
        if max(pre_times) >= min(post_times):
            raise ValueError(
                f"{model_name}.fit() requires all pre-treatment times < all post-treatment times."
            )

    def _prepare_balanced_panel(
        self, data: PanelDataSCM
    ) -> tuple[pd.DataFrame, List[Hashable], List[Any], List[Any], List[Any]]:
        df = data.df_analysis()
        unit_col = data.unit_col
        time_col = data.time_col
        outcome_col = data.y

        donors = list(data.donor_pool())
        if len(donors) < 1:
            raise ValueError("ASCM.fit() requires at least one donor unit.")

        pre_times = list(data.pre_times())
        post_times = list(data.post_times())
        self._validate_pre_post_times(pre_times=pre_times, post_times=post_times, model_name="ASCM")

        all_times = list(pre_times) + list(post_times)
        keep_units = [data.treated_unit] + donors
        block = df[df[unit_col].isin(keep_units) & df[time_col].isin(all_times)].copy()

        has_dupes = block.duplicated([unit_col, time_col]).any()
        if has_dupes:
            raise ValueError(
                "ASCM.fit() requires unique (unit,time) rows in the analysis block. "
                "Aggregate duplicated rows before fitting."
            )

        panel = block.pivot(index=unit_col, columns=time_col, values=outcome_col)
        panel = panel.reindex(index=keep_units, columns=all_times)

        if panel.isna().any().any():
            mask = panel.isna().to_numpy()
            row_ids, col_ids = np.where(mask)
            examples: List[str] = []
            for r, c in zip(row_ids[:5], col_ids[:5]):
                examples.append(f"({panel.index[r]!r}, {panel.columns[c]!r})")

            raise ValueError(
                "ASCM.fit() requires a balanced block with fully observed outcomes in pre/post "
                f"periods. Missing unit-time cells include: {', '.join(examples)}."
            )

        return panel, donors, pre_times, post_times, all_times

    def fit(self, data: PanelDataSCM) -> "AugmentedSyntheticControl":
        """
        Fit ASCM on a balanced pre/post block extracted from PanelDataSCM.

        Fit-time ASCM checks enforced here:
        - balanced unit-time block for treated + donor units
        - no missing outcomes in pre/post periods
        """
        if not isinstance(data, PanelDataSCM):
            raise ValueError("Input must be a PanelDataSCM object.")

        panel, donors, pre_times, post_times, all_times = self._prepare_balanced_panel(data)

        treated = data.treated_unit
        y1_all = panel.loc[treated, all_times].to_numpy(dtype=float)
        y1_pre = panel.loc[treated, pre_times].to_numpy(dtype=float)

        y0_all = panel.loc[donors, all_times].to_numpy(dtype=float).T  # (T, J)
        x0_pre = panel.loc[donors, pre_times].to_numpy(dtype=float).T  # (T0, J)

        w_sc = self._fit_simplex_weights(x0_pre=x0_pre, y1_pre=y1_pre)
        w_aug = self._augment_weights(x0_pre=x0_pre, y1_pre=y1_pre, w_sc=w_sc)

        y0_hat_sc = y0_all @ w_sc
        y0_hat_aug = y0_all @ w_aug
        gap_sc = y1_all - y0_hat_sc
        gap_aug = y1_all - y0_hat_aug

        self._data = data
        self._donors = list(donors)
        self._pre_times = list(pre_times)
        self._post_times = list(post_times)
        self._all_times = list(all_times)

        self._w_sc = w_sc
        self._w_aug = w_aug

        self._observed = pd.Series(y1_all, index=self._all_times, name="observed_outcome")
        self._synthetic_sc = pd.Series(y0_hat_sc, index=self._all_times, name="synthetic_outcome_sc")
        self._synthetic_aug = pd.Series(y0_hat_aug, index=self._all_times, name="synthetic_outcome")
        self._gap_sc = pd.Series(gap_sc, index=self._all_times, name="gap_sc")
        self._gap_aug = pd.Series(gap_aug, index=self._all_times, name="gap_aug")
        att_sc = float(self._gap_sc.loc[self._post_times].mean())
        att_aug = float(self._gap_aug.loc[self._post_times].mean())

        pre_resid_sc = y1_pre - (x0_pre @ w_sc)
        pre_resid_aug = y1_pre - (x0_pre @ w_aug)
        gram_aug = x0_pre.T @ x0_pre + self.lambda_aug * np.eye(x0_pre.shape[1], dtype=float)
        cond_gram_aug = float(np.linalg.cond(gram_aug))
        l1_w_aug = float(np.sum(np.abs(w_aug)))
        max_abs_w_aug = float(np.max(np.abs(w_aug)))
        alpha = 0.05
        post_start = len(self._pre_times)
        outcomes_by_unit = np.vstack([y1_all[None, :], y0_all.T])
        placebo_aug = self._placebo_in_space_inference(
            outcomes_by_unit=outcomes_by_unit,
            treated_gap=gap_aug,
            n_pre=post_start,
            alpha=alpha,
            use_augmented=True,
        )
        placebo_sc = self._placebo_in_space_inference(
            outcomes_by_unit=outcomes_by_unit,
            treated_gap=gap_sc,
            n_pre=post_start,
            alpha=alpha,
            use_augmented=False,
        )
        inference_aug = self._inference_from_placebo(
            att=att_aug,
            baseline_post_synthetic=float(np.mean(y0_hat_aug[post_start:])),
            placebo=placebo_aug,
            alpha=alpha,
        )
        inference_sc = self._inference_from_placebo(
            att=att_sc,
            baseline_post_synthetic=float(np.mean(y0_hat_sc[post_start:])),
            placebo=placebo_sc,
            alpha=alpha,
        )
        if inference_aug["p_value"] is None:
            raise RuntimeError(
                "Placebo inference requires at least 2 donor units with successful placebo fits."
            )

        if cond_gram_aug > 1e10:
            warnings.warn(
                f"ASCM augmented normal equations are ill-conditioned (cond={cond_gram_aug:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        if l1_w_aug > 5.0 or max_abs_w_aug > 2.0:
            warnings.warn(
                "ASCM augmented donor weights are extreme; estimates may be unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._diagnostics = {
            "n_donors": len(self._donors),
            "n_pre_periods": len(self._pre_times),
            "n_post_periods": len(self._post_times),
            "enforce_sum_to_one_augmented": bool(self.enforce_sum_to_one_augmented),
            "pre_rmse_sc": self._rmse(pre_resid_sc),
            "pre_rmse_augmented": self._rmse(pre_resid_aug),
            "sum_weights_sc": float(np.sum(w_sc)),
            "sum_weights_augmented": float(np.sum(w_aug)),
            "min_weight_sc": float(np.min(w_sc)),
            "max_weight_sc": float(np.max(w_sc)),
            "min_weight_augmented": float(np.min(w_aug)),
            "max_weight_augmented": float(np.max(w_aug)),
            "l1_norm_weights_augmented": l1_w_aug,
            "max_abs_weight_augmented": max_abs_w_aug,
            "cond_augmented_gram": cond_gram_aug,
            "att_sc": att_sc,
            "att_augmented": att_aug,
            "ci_alpha": inference_aug["alpha"],
            "att_ci_method": inference_aug["ci_method"],
            "att_ci_lower_absolute": inference_aug["ci_lower_absolute"],
            "att_ci_upper_absolute": inference_aug["ci_upper_absolute"],
            "att_value_relative": inference_aug["value_relative"],
            "att_ci_lower_relative": inference_aug["ci_lower_relative"],
            "att_ci_upper_relative": inference_aug["ci_upper_relative"],
            "att_se_absolute": None,
            "att_p_value": inference_aug["p_value"],
            "att_is_significant": inference_aug["is_significant"],
            "att_p_value_method": inference_aug["p_value_method"],
            "att_se_method": None,
            "att_df_t": None,
            "att_hac_max_lag": None,
            "att_pre_resid_n": len(self._pre_times),
            "att_pre_long_run_variance": None,
            "att_p_value_pre_residual": None,
            "att_is_significant_pre_residual": None,
            "att_placebo_n": inference_aug["n_placebos"],
            "att_placebo_failed_n": inference_aug["n_failed_placebos"],
            "att_placebo_p_value": inference_aug["p_value_placebo_att"],
            "att_placebo_p_value_rmspe_ratio": inference_aug["p_value_placebo_rmspe_ratio"],
            "att_is_significant_fit_adjusted": inference_aug["is_significant_fit_adjusted"],
            "att_fit_adjusted_warning": inference_aug["fit_adjusted_warning"],
            "att_placebo_treated_rmspe_ratio": inference_aug["treated_rmspe_ratio"],
            "att_placebo_ci_lower_absolute": inference_aug["ci_lower_absolute"],
            "att_placebo_ci_upper_absolute": inference_aug["ci_upper_absolute"],
            "att_placebo_min_possible_p": inference_aug["placebo_min_possible_p"],
            "att_placebo_ci_is_unbounded": inference_aug["placebo_ci_is_unbounded"],
            "att_placebo_att_distribution": inference_aug["placebo_att_distribution"],
            "att_placebo_rmspe_ratio_distribution": inference_aug[
                "placebo_rmspe_ratio_distribution"
            ],
            "att_relative_baseline_post_synthetic": inference_aug["baseline_post_synthetic"],
            "att_sc_ci_method": inference_sc["ci_method"],
            "att_sc_ci_lower_absolute": inference_sc["ci_lower_absolute"],
            "att_sc_ci_upper_absolute": inference_sc["ci_upper_absolute"],
            "att_sc_value_relative": inference_sc["value_relative"],
            "att_sc_ci_lower_relative": inference_sc["ci_lower_relative"],
            "att_sc_ci_upper_relative": inference_sc["ci_upper_relative"],
            "att_sc_se_absolute": None,
            "att_sc_p_value": inference_sc["p_value"],
            "att_sc_is_significant": inference_sc["is_significant"],
            "att_sc_p_value_method": inference_sc["p_value_method"],
            "att_sc_se_method": None,
            "att_sc_df_t": None,
            "att_sc_hac_max_lag": None,
            "att_sc_pre_resid_n": len(self._pre_times),
            "att_sc_pre_long_run_variance": None,
            "att_sc_p_value_pre_residual": None,
            "att_sc_is_significant_pre_residual": None,
            "att_sc_placebo_n": inference_sc["n_placebos"],
            "att_sc_placebo_failed_n": inference_sc["n_failed_placebos"],
            "att_sc_placebo_p_value": inference_sc["p_value_placebo_att"],
            "att_sc_placebo_p_value_rmspe_ratio": inference_sc["p_value_placebo_rmspe_ratio"],
            "att_sc_is_significant_fit_adjusted": inference_sc["is_significant_fit_adjusted"],
            "att_sc_fit_adjusted_warning": inference_sc["fit_adjusted_warning"],
            "att_sc_placebo_treated_rmspe_ratio": inference_sc["treated_rmspe_ratio"],
            "att_sc_placebo_ci_lower_absolute": inference_sc["ci_lower_absolute"],
            "att_sc_placebo_ci_upper_absolute": inference_sc["ci_upper_absolute"],
            "att_sc_placebo_min_possible_p": inference_sc["placebo_min_possible_p"],
            "att_sc_placebo_ci_is_unbounded": inference_sc["placebo_ci_is_unbounded"],
            "att_sc_placebo_att_distribution": inference_sc["placebo_att_distribution"],
            "att_sc_placebo_rmspe_ratio_distribution": inference_sc[
                "placebo_rmspe_ratio_distribution"
            ],
            "att_sc_relative_baseline_post_synthetic": inference_sc["baseline_post_synthetic"],
            "inference_policy": self.inference_policy,
        }

        self._att_sc = att_sc
        self._att_aug = att_aug
        self._ci_upper_absolute = inference_aug["ci_upper_absolute"]
        self._ci_lower_absolute = inference_aug["ci_lower_absolute"]
        self._value_relative = inference_aug["value_relative"]
        self._ci_upper_relative = inference_aug["ci_upper_relative"]
        self._ci_lower_relative = inference_aug["ci_lower_relative"]
        self._ci_alpha = inference_aug["alpha"]
        self._p_value = inference_aug["p_value"]
        self._is_significant = inference_aug["is_significant"]
        self._is_fitted = True
        return self

    def estimate(self) -> PanelEstimate:
        """Return fitted ASCM estimate as a PanelEstimate contract."""
        if not self._is_fitted or self._data is None:
            raise RuntimeError("Model must be fitted with .fit(data) before calling .estimate().")

        return PanelEstimate(
            estimand="ATTE",
            model=self.__class__.__name__,
            treated_unit=self._data.treated_unit,
            treatment_start=self._data.treatment_start,
            pre_times=list(self._pre_times),
            post_times=list(self._post_times),
            att=float(self._att_aug),
            att_sc=float(self._att_sc),
            ci_upper_absolute=self._ci_upper_absolute,
            ci_lower_absolute=self._ci_lower_absolute,
            value_relative=self._value_relative,
            ci_upper_relative=self._ci_upper_relative,
            ci_lower_relative=self._ci_lower_relative,
            alpha=self._ci_alpha,
            p_value=self._p_value,
            is_significant=self._is_significant,
            att_by_time=self._gap_aug.loc[self._post_times].copy(),
            att_by_time_sc=self._gap_sc.loc[self._post_times].copy(),
            observed_outcome=self._observed.copy(),
            synthetic_outcome=self._synthetic_aug.copy(),
            synthetic_outcome_sc=self._synthetic_sc.copy(),
            donor_weights_augmented={
                donor: float(weight) for donor, weight in zip(self._donors, self._w_aug)
            },
            donor_weights_sc={
                donor: float(weight) for donor, weight in zip(self._donors, self._w_sc)
            },
            diagnostics=dict(self._diagnostics),
        )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(status='{status}')"


ASCM = AugmentedSyntheticControl


class RobustSyntheticControl(AugmentedSyntheticControl):
    """
    Robust Synthetic Control for missing panel outcomes.

    This model supports unit-time gaps and missing outcomes by first applying
    low-rank matrix completion (soft-impute style) on the treated + donor panel,
    then fitting standard simplex SC and ridge-augmented weights on the completed
    pre-treatment block.
    """

    def __init__(
        self,
        *,
        lambda_aug: float = 1.0,
        lambda_sc: float = 1e-6,
        max_iter: int = 2_000,
        tol: float = 1e-9,
        enforce_sum_to_one_augmented: bool = True,
        inference_policy: Literal["placebo"] = "placebo",
        completion_max_iter: int = 500,
        completion_tol: float = 1e-6,
        sv_threshold: float | None = None,
        sv_threshold_ratio: float = 0.5,
        max_rank: int | None = None,
        min_pre_observed: int = 1,
    ) -> None:
        super().__init__(
            lambda_aug=lambda_aug,
            lambda_sc=lambda_sc,
            max_iter=max_iter,
            tol=tol,
            enforce_sum_to_one_augmented=enforce_sum_to_one_augmented,
            inference_policy=inference_policy,
        )

        self.completion_max_iter = int(completion_max_iter)
        self.completion_tol = float(completion_tol)
        self.sv_threshold = None if sv_threshold is None else float(sv_threshold)
        self.sv_threshold_ratio = float(sv_threshold_ratio)
        self.max_rank = None if max_rank is None else int(max_rank)
        self.min_pre_observed = int(min_pre_observed)
        self.inference_policy = str(inference_policy)

        if self.completion_max_iter <= 0:
            raise ValueError("completion_max_iter must be a positive integer.")
        if not np.isfinite(self.completion_tol) or self.completion_tol <= 0.0:
            raise ValueError("completion_tol must be finite and > 0.")
        if self.sv_threshold is not None and (
            (not np.isfinite(self.sv_threshold)) or self.sv_threshold < 0.0
        ):
            raise ValueError("sv_threshold must be None or a finite value >= 0.")
        if not np.isfinite(self.sv_threshold_ratio) or self.sv_threshold_ratio < 0.0:
            raise ValueError("sv_threshold_ratio must be finite and >= 0.")
        if self.max_rank is not None and self.max_rank <= 0:
            raise ValueError("max_rank must be None or a positive integer.")
        if self.min_pre_observed <= 0:
            raise ValueError("min_pre_observed must be a positive integer.")

    @staticmethod
    def _safe_rmse(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    @staticmethod
    def _mean_with_fallback(values: np.ndarray, mask: np.ndarray, axis: int, fallback: float) -> np.ndarray:
        weighted_sum = np.where(mask, values, 0.0).sum(axis=axis, dtype=float)
        counts = mask.sum(axis=axis)
        out = np.full(weighted_sum.shape, fallback, dtype=float)
        np.divide(weighted_sum, counts, out=out, where=counts > 0)
        return out

    def _prepare_incomplete_panel(
        self, data: PanelDataSCM
    ) -> tuple[pd.DataFrame, np.ndarray, List[Hashable], List[Any], List[Any], List[Any]]:
        df = data.df_analysis()
        unit_col = data.unit_col
        time_col = data.time_col
        outcome_col = data.y

        donors = list(data.donor_pool())
        if len(donors) < 1:
            raise ValueError("RobustSyntheticControl.fit() requires at least one donor unit.")

        pre_times = list(data.pre_times())
        post_times = list(data.post_times())
        self._validate_pre_post_times(
            pre_times=pre_times,
            post_times=post_times,
            model_name="RobustSyntheticControl",
        )

        all_times = list(pre_times) + list(post_times)
        keep_units = [data.treated_unit] + donors
        block = df[df[unit_col].isin(keep_units) & df[time_col].isin(all_times)].copy()

        has_dupes = block.duplicated([unit_col, time_col]).any()
        if has_dupes:
            raise ValueError(
                "RobustSyntheticControl.fit() requires unique (unit,time) rows in the analysis "
                "block. Aggregate duplicated rows before fitting."
            )

        panel = block.pivot(index=unit_col, columns=time_col, values=outcome_col)
        panel = panel.reindex(index=keep_units, columns=all_times)
        y_matrix = panel.to_numpy(dtype=float)

        if data.observed_col is not None:
            observed_block = block[[unit_col, time_col, data.observed_col]].copy()
            observed_block["_observed"] = observed_block[data.observed_col].astype("boolean")
            observed_panel = observed_block.pivot(index=unit_col, columns=time_col, values="_observed")
            observed_panel = observed_panel.reindex(index=keep_units, columns=all_times)
            observed_mask = observed_panel.fillna(False).to_numpy(dtype=bool)
        else:
            observed_mask = np.isfinite(y_matrix)

        # Treat non-finite outcomes as missing even when observed mask is provided.
        observed_mask = observed_mask & np.isfinite(y_matrix)
        if not observed_mask.any():
            raise ValueError(
                "RobustSyntheticControl.fit() requires at least one observed outcome in the "
                "treated+donor analysis block."
            )

        return panel, observed_mask, donors, pre_times, post_times, all_times

    def _complete_low_rank_matrix(
        self,
        y_matrix: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        y = np.asarray(y_matrix, dtype=float)
        mask = np.asarray(observed_mask, dtype=bool)
        missing_mask = ~mask

        if not missing_mask.any():
            return y.copy(), {
                "converged": True,
                "iterations": 0,
                "sv_threshold_used": 0.0 if self.sv_threshold is None else float(self.sv_threshold),
                "effective_rank": int(np.linalg.matrix_rank(y)),
                "missing_cells": 0,
            }

        observed_vals = y[mask]
        global_mean = float(np.mean(observed_vals))
        row_means = self._mean_with_fallback(values=y, mask=mask, axis=1, fallback=global_mean)
        col_means = self._mean_with_fallback(values=y, mask=mask, axis=0, fallback=global_mean)
        init_missing = 0.5 * (row_means[:, None] + col_means[None, :])
        x = np.where(mask, y, init_missing)
        x = np.where(np.isfinite(x), x, global_mean)

        _, init_singular, _ = np.linalg.svd(x, full_matrices=False)
        if init_singular.size == 0:
            raise RuntimeError("RobustSyntheticControl matrix completion failed: empty SVD spectrum.")
        tau = (
            self.sv_threshold
            if self.sv_threshold is not None
            else self.sv_threshold_ratio * float(init_singular[0])
        )
        tau = float(max(tau, 0.0))

        converged = False
        rel_delta = np.inf
        rank_eff = 0
        n_iter = 0
        for iteration in range(1, self.completion_max_iter + 1):
            u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
            singular_shrunk = np.maximum(singular_values - tau, 0.0)
            if self.max_rank is not None and self.max_rank < singular_shrunk.size:
                singular_shrunk[self.max_rank :] = 0.0

            rank_eff = int(np.sum(singular_shrunk > 1e-12))
            x_low_rank = (u * singular_shrunk) @ vt
            x_new = np.where(mask, y, x_low_rank)

            delta = self._safe_rmse(x_new[missing_mask] - x[missing_mask])
            scale = self._safe_rmse(x[missing_mask]) + 1e-12
            rel_delta = delta / scale
            x = x_new
            n_iter = iteration
            if rel_delta < self.completion_tol:
                converged = True
                break

        if not converged:
            warnings.warn(
                "RobustSyntheticControl matrix completion did not converge within "
                f"{self.completion_max_iter} iterations (relative_change={rel_delta:.3e}).",
                RuntimeWarning,
                stacklevel=2,
            )

        x = np.where(mask, y, x)
        if not np.isfinite(x).all():
            raise RuntimeError(
                "RobustSyntheticControl matrix completion returned non-finite imputed values."
            )

        return x, {
            "converged": bool(converged),
            "iterations": int(n_iter),
            "sv_threshold_used": float(tau),
            "effective_rank": int(rank_eff),
            "missing_cells": int(np.sum(missing_mask)),
        }

    def _placebo_in_space_inference_robust(
        self,
        *,
        y_matrix: np.ndarray,
        observed_mask: np.ndarray,
        treated_gap: np.ndarray,
        n_pre: int,
        alpha: float = 0.05,
        use_augmented: bool,
    ) -> dict[str, Any]:
        """Compute placebo-in-space inference with robust completion semantics.

        For each donor-as-placebo run:
        - exclude the real treated unit from donor pools
        - hide pseudo-treated post outcomes during completion
        - require observed pseudo-treated post outcomes for ATT
        """
        y = np.asarray(y_matrix, dtype=float)
        mask = np.asarray(observed_mask, dtype=bool)
        n_units, n_times = y.shape
        treated_gap_arr = np.asarray(treated_gap, dtype=float)

        result: dict[str, Any] = {
            "available": False,
            "n_placebos": 0,
            "n_failed_placebos": 0,
            "p_value_att": None,
            "p_value_rmspe_ratio": None,
            "treated_rmspe_ratio": None,
            "ci_lower_absolute": None,
            "ci_upper_absolute": None,
            "min_possible_p": None,
            "placebo_ci_is_unbounded": False,
            "ci_method": "placebo_inversion_att",
            "p_value_method": "placebo_in_space_att",
        }
        if n_units < 3:
            return result
        if n_pre < 1 or n_pre >= n_times:
            return result
        if treated_gap_arr.size != n_times:
            raise ValueError("treated_gap must have length equal to y_matrix.shape[1].")

        placebo_atts: List[float] = []
        placebo_rmspe_ratios: List[float] = []
        failed = 0
        donor_indices = list(range(1, n_units))
        for pseudo_treated_idx in donor_indices:
            donor_pool_idx = [idx for idx in donor_indices if idx != pseudo_treated_idx]
            if len(donor_pool_idx) < 1:
                failed += 1
                continue

            # Keep pseudo-treated row first to mirror treated-first conventions.
            keep_rows = [pseudo_treated_idx] + donor_pool_idx
            y_placebo = y[np.asarray(keep_rows, dtype=int), :]
            mask_placebo = mask[np.asarray(keep_rows, dtype=int), :].copy()

            pseudo_pre_observed = int(np.sum(mask_placebo[0, :n_pre]))
            if pseudo_pre_observed < self.min_pre_observed:
                failed += 1
                continue
            donor_pre_counts = mask_placebo[1:, :n_pre].sum(axis=1)
            donor_keep_mask = donor_pre_counts >= self.min_pre_observed
            if not bool(np.any(donor_keep_mask)):
                failed += 1
                continue
            if not bool(np.all(donor_keep_mask)):
                donor_keep_idx = [idx + 1 for idx, keep in enumerate(donor_keep_mask) if bool(keep)]
                keep_local_rows = [0] + donor_keep_idx
                y_placebo = y_placebo[np.asarray(keep_local_rows, dtype=int), :]
                mask_placebo = mask_placebo[np.asarray(keep_local_rows, dtype=int), :]

            # Mirror robust ATT semantics: pseudo-treated post must be observed.
            if not bool(np.all(mask_placebo[0, n_pre:])):
                failed += 1
                continue

            completion_mask = mask_placebo.copy()
            completion_mask[0, n_pre:] = False
            try:
                completed_panel, _ = self._complete_low_rank_matrix(
                    y_matrix=y_placebo,
                    observed_mask=completion_mask,
                )
            except Exception:
                failed += 1
                continue

            y1_completed = completed_panel[0, :]
            y1_observed_raw = y_placebo[0, :]
            y1_all = np.where(np.isfinite(y1_observed_raw), y1_observed_raw, y1_completed)

            y0_all = completed_panel[1:, :].T
            x0_pre = y0_all[:n_pre, :]
            y1_pre = y1_all[:n_pre]
            pre_obs_rate = mask_placebo[:, :n_pre].mean(axis=0)
            pre_obs_rate = np.clip(pre_obs_rate, 1e-6, 1.0)
            pre_weight_sqrt = np.sqrt(pre_obs_rate)
            x0_pre_weighted = x0_pre * pre_weight_sqrt[:, None]
            y1_pre_weighted = y1_pre * pre_weight_sqrt

            try:
                w_sc_pseudo = self._fit_simplex_weights(
                    x0_pre=x0_pre_weighted,
                    y1_pre=y1_pre_weighted,
                )
                w_pseudo = (
                    self._augment_weights(
                        x0_pre=x0_pre_weighted,
                        y1_pre=y1_pre_weighted,
                        w_sc=w_sc_pseudo,
                    )
                    if use_augmented
                    else w_sc_pseudo
                )
            except Exception:
                failed += 1
                continue

            gap_pseudo = y1_all - (y0_all @ w_pseudo)
            placebo_atts.append(float(np.mean(gap_pseudo[n_pre:])))
            pre_rmspe = self._rmse(gap_pseudo[:n_pre])
            post_rmspe = self._rmse(gap_pseudo[n_pre:])
            placebo_rmspe_ratios.append(float(post_rmspe / max(pre_rmspe, 1e-12)))

        if not placebo_atts:
            result["n_failed_placebos"] = int(failed)
            return result

        return self._summarize_placebo_distribution(
            treated_gap=treated_gap_arr,
            n_pre=n_pre,
            placebo_atts=placebo_atts,
            placebo_rmspe_ratios=placebo_rmspe_ratios,
            failed=failed,
            alpha=alpha,
        )

    def fit(self, data: PanelDataSCM) -> "RobustSyntheticControl":
        """
        Fit robust SC on PanelDataSCM with optional missing outcomes/cells.

        Workflow:
        1) Build treated + donor panel over pre/post windows.
        2) Complete missing cells by low-rank soft-impute iterations.
        3) Fit simplex SC and ridge-augmented weights on completed pre-period.
        """
        if not isinstance(data, PanelDataSCM):
            raise ValueError("Input must be a PanelDataSCM object.")

        panel, observed_mask, donors, pre_times, post_times, all_times = self._prepare_incomplete_panel(data)

        n_pre = len(pre_times)
        treated_pre_observed_mask = observed_mask[0, :n_pre]
        treated_pre_observed = int(np.sum(treated_pre_observed_mask))
        if treated_pre_observed < self.min_pre_observed:
            raise ValueError(
                "RobustSyntheticControl.fit() requires at least "
                f"{self.min_pre_observed} observed treated pre-treatment outcomes."
            )

        donor_pre_counts = observed_mask[1:, :n_pre].sum(axis=1)
        donor_keep_mask = donor_pre_counts >= self.min_pre_observed
        if not bool(np.any(donor_keep_mask)):
            raise ValueError(
                "RobustSyntheticControl.fit() could not find donors with enough observed "
                "pre-treatment outcomes."
            )
        if not bool(np.all(donor_keep_mask)):
            dropped = [donor for donor, keep in zip(donors, donor_keep_mask) if not bool(keep)]
            warnings.warn(
                "Dropping donors with too few observed pre-treatment outcomes: "
                f"{dropped}.",
                RuntimeWarning,
                stacklevel=2,
            )
            keep_rows = [0] + [idx + 1 for idx, keep in enumerate(donor_keep_mask) if bool(keep)]
            panel = panel.iloc[keep_rows, :]
            observed_mask = observed_mask[np.asarray(keep_rows, dtype=int), :]
            donors = [donor for donor, keep in zip(donors, donor_keep_mask) if bool(keep)]
        treated_pre_observed_mask = observed_mask[0, :n_pre]

        treated_post_observed_mask = observed_mask[0, n_pre:]
        if not bool(np.all(treated_post_observed_mask)):
            missing_post_times = [
                time_value
                for time_value, is_observed in zip(post_times, treated_post_observed_mask)
                if not bool(is_observed)
            ]
            raise ValueError(
                "RobustSyntheticControl.fit() requires observed treated post-treatment outcomes "
                f"to compute ATTE. Missing treated post periods: {missing_post_times}."
            )

        y_matrix = panel.to_numpy(dtype=float)
        # Prevent post-treatment leakage: the treated unit's post outcomes are
        # withheld from matrix completion even when observed.
        completion_mask = observed_mask.copy()
        completion_mask[0, n_pre:] = False
        completed_panel, completion_diag = self._complete_low_rank_matrix(
            y_matrix=y_matrix,
            observed_mask=completion_mask,
        )

        y1_completed = completed_panel[0, :]
        y1_observed_raw = y_matrix[0, :]
        y1_all = np.where(np.isfinite(y1_observed_raw), y1_observed_raw, y1_completed)

        y0_all = completed_panel[1:, :].T  # (T, J)
        x0_pre = y0_all[:n_pre, :]
        y1_pre = y1_all[:n_pre]

        pre_obs_rate = observed_mask[:, :n_pre].mean(axis=0)
        pre_obs_rate = np.clip(pre_obs_rate, 1e-6, 1.0)
        pre_weight_sqrt = np.sqrt(pre_obs_rate)
        x0_pre_weighted = x0_pre * pre_weight_sqrt[:, None]
        y1_pre_weighted = y1_pre * pre_weight_sqrt

        w_sc = self._fit_simplex_weights(x0_pre=x0_pre_weighted, y1_pre=y1_pre_weighted)
        w_aug = self._augment_weights(x0_pre=x0_pre_weighted, y1_pre=y1_pre_weighted, w_sc=w_sc)

        y0_hat_sc = y0_all @ w_sc
        y0_hat_aug = y0_all @ w_aug
        gap_sc = y1_all - y0_hat_sc
        gap_aug = y1_all - y0_hat_aug

        self._data = data
        self._donors = list(donors)
        self._pre_times = list(pre_times)
        self._post_times = list(post_times)
        self._all_times = list(all_times)

        self._w_sc = w_sc
        self._w_aug = w_aug

        self._observed = pd.Series(y1_all, index=self._all_times, name="observed_outcome")
        self._synthetic_sc = pd.Series(y0_hat_sc, index=self._all_times, name="synthetic_outcome_sc")
        self._synthetic_aug = pd.Series(y0_hat_aug, index=self._all_times, name="synthetic_outcome")
        self._gap_sc = pd.Series(gap_sc, index=self._all_times, name="gap_sc")
        self._gap_aug = pd.Series(gap_aug, index=self._all_times, name="gap_aug")
        att_sc = float(self._gap_sc.loc[self._post_times].mean())
        att_aug = float(self._gap_aug.loc[self._post_times].mean())

        pre_resid_sc = y1_pre - (x0_pre @ w_sc)
        pre_resid_aug = y1_pre - (x0_pre @ w_aug)
        pre_resid_sc_obs = pre_resid_sc[treated_pre_observed_mask]
        pre_resid_aug_obs = pre_resid_aug[treated_pre_observed_mask]
        pre_resid_sc_weighted = y1_pre_weighted - (x0_pre_weighted @ w_sc)
        pre_resid_aug_weighted = y1_pre_weighted - (x0_pre_weighted @ w_aug)
        gram_aug = x0_pre_weighted.T @ x0_pre_weighted + self.lambda_aug * np.eye(
            x0_pre_weighted.shape[1], dtype=float
        )
        cond_gram_aug = float(np.linalg.cond(gram_aug))
        l1_w_aug = float(np.sum(np.abs(w_aug)))
        max_abs_w_aug = float(np.max(np.abs(w_aug)))
        alpha = 0.05
        post_start = len(self._pre_times)
        placebo_aug = self._placebo_in_space_inference_robust(
            y_matrix=y_matrix,
            observed_mask=observed_mask,
            treated_gap=gap_aug,
            n_pre=post_start,
            alpha=alpha,
            use_augmented=True,
        )
        placebo_sc = self._placebo_in_space_inference_robust(
            y_matrix=y_matrix,
            observed_mask=observed_mask,
            treated_gap=gap_sc,
            n_pre=post_start,
            alpha=alpha,
            use_augmented=False,
        )
        inference_aug = self._inference_from_placebo(
            att=att_aug,
            baseline_post_synthetic=float(np.mean(y0_hat_aug[post_start:])),
            placebo=placebo_aug,
            alpha=alpha,
        )
        inference_sc = self._inference_from_placebo(
            att=att_sc,
            baseline_post_synthetic=float(np.mean(y0_hat_sc[post_start:])),
            placebo=placebo_sc,
            alpha=alpha,
        )
        if inference_aug["p_value"] is None:
            raise RuntimeError(
                "Placebo inference requires at least 2 donor units with successful placebo fits."
            )

        if cond_gram_aug > 1e10:
            warnings.warn(
                f"RobustSyntheticControl augmented normal equations are ill-conditioned (cond={cond_gram_aug:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        if l1_w_aug > 5.0 or max_abs_w_aug > 2.0:
            warnings.warn(
                "RobustSyntheticControl augmented donor weights are extreme; estimates may be unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

        treated_missing_pre = int(np.sum(~np.isfinite(y1_observed_raw[:n_pre])))
        treated_missing_post = int(np.sum(~np.isfinite(y1_observed_raw[n_pre:])))
        missing_mask = ~observed_mask
        self._diagnostics = {
            "n_donors": len(self._donors),
            "n_pre_periods": len(self._pre_times),
            "n_post_periods": len(self._post_times),
            "enforce_sum_to_one_augmented": bool(self.enforce_sum_to_one_augmented),
            "pre_rmse_sc": self._rmse(pre_resid_sc),
            "pre_rmse_augmented": self._rmse(pre_resid_aug),
            "pre_rmse_sc_observed_treated_pre": self._rmse(pre_resid_sc_obs),
            "pre_rmse_augmented_observed_treated_pre": self._rmse(pre_resid_aug_obs),
            "pre_rmse_sc_weighted": self._rmse(pre_resid_sc_weighted),
            "pre_rmse_augmented_weighted": self._rmse(pre_resid_aug_weighted),
            "sum_weights_sc": float(np.sum(w_sc)),
            "sum_weights_augmented": float(np.sum(w_aug)),
            "min_weight_sc": float(np.min(w_sc)),
            "max_weight_sc": float(np.max(w_sc)),
            "min_weight_augmented": float(np.min(w_aug)),
            "max_weight_augmented": float(np.max(w_aug)),
            "l1_norm_weights_augmented": l1_w_aug,
            "max_abs_weight_augmented": max_abs_w_aug,
            "cond_augmented_gram": cond_gram_aug,
            "att_sc": att_sc,
            "att_augmented": att_aug,
            "n_missing_cells": int(np.sum(missing_mask)),
            "missing_cell_fraction": float(np.mean(missing_mask)),
            "treated_missing_pre": treated_missing_pre,
            "treated_missing_post": treated_missing_post,
            "treated_observed_pre_for_inference": int(np.sum(treated_pre_observed_mask)),
            "completion_converged": bool(completion_diag["converged"]),
            "completion_iterations": int(completion_diag["iterations"]),
            "completion_effective_rank": int(completion_diag["effective_rank"]),
            "completion_sv_threshold": float(completion_diag["sv_threshold_used"]),
            "ci_alpha": inference_aug["alpha"],
            "att_ci_method": inference_aug["ci_method"],
            "att_ci_lower_absolute": inference_aug["ci_lower_absolute"],
            "att_ci_upper_absolute": inference_aug["ci_upper_absolute"],
            "att_value_relative": inference_aug["value_relative"],
            "att_ci_lower_relative": inference_aug["ci_lower_relative"],
            "att_ci_upper_relative": inference_aug["ci_upper_relative"],
            "att_se_absolute": None,
            "att_p_value": inference_aug["p_value"],
            "att_is_significant": inference_aug["is_significant"],
            "att_p_value_method": inference_aug["p_value_method"],
            "att_se_method": None,
            "att_df_t": None,
            "att_hac_max_lag": None,
            "att_pre_resid_n": int(np.sum(treated_pre_observed_mask)),
            "att_pre_long_run_variance": None,
            "att_p_value_pre_residual": None,
            "att_is_significant_pre_residual": None,
            "att_placebo_n": inference_aug["n_placebos"],
            "att_placebo_failed_n": inference_aug["n_failed_placebos"],
            "att_placebo_p_value": inference_aug["p_value_placebo_att"],
            "att_placebo_p_value_rmspe_ratio": inference_aug["p_value_placebo_rmspe_ratio"],
            "att_is_significant_fit_adjusted": inference_aug["is_significant_fit_adjusted"],
            "att_fit_adjusted_warning": inference_aug["fit_adjusted_warning"],
            "att_placebo_treated_rmspe_ratio": inference_aug["treated_rmspe_ratio"],
            "att_placebo_ci_lower_absolute": inference_aug["ci_lower_absolute"],
            "att_placebo_ci_upper_absolute": inference_aug["ci_upper_absolute"],
            "att_placebo_min_possible_p": inference_aug["placebo_min_possible_p"],
            "att_placebo_ci_is_unbounded": inference_aug["placebo_ci_is_unbounded"],
            "att_placebo_att_distribution": inference_aug["placebo_att_distribution"],
            "att_placebo_rmspe_ratio_distribution": inference_aug[
                "placebo_rmspe_ratio_distribution"
            ],
            "att_relative_baseline_post_synthetic": inference_aug["baseline_post_synthetic"],
            "att_sc_ci_method": inference_sc["ci_method"],
            "att_sc_ci_lower_absolute": inference_sc["ci_lower_absolute"],
            "att_sc_ci_upper_absolute": inference_sc["ci_upper_absolute"],
            "att_sc_value_relative": inference_sc["value_relative"],
            "att_sc_ci_lower_relative": inference_sc["ci_lower_relative"],
            "att_sc_ci_upper_relative": inference_sc["ci_upper_relative"],
            "att_sc_se_absolute": None,
            "att_sc_p_value": inference_sc["p_value"],
            "att_sc_is_significant": inference_sc["is_significant"],
            "att_sc_p_value_method": inference_sc["p_value_method"],
            "att_sc_se_method": None,
            "att_sc_df_t": None,
            "att_sc_hac_max_lag": None,
            "att_sc_pre_resid_n": int(np.sum(treated_pre_observed_mask)),
            "att_sc_pre_long_run_variance": None,
            "att_sc_p_value_pre_residual": None,
            "att_sc_is_significant_pre_residual": None,
            "att_sc_placebo_n": inference_sc["n_placebos"],
            "att_sc_placebo_failed_n": inference_sc["n_failed_placebos"],
            "att_sc_placebo_p_value": inference_sc["p_value_placebo_att"],
            "att_sc_placebo_p_value_rmspe_ratio": inference_sc["p_value_placebo_rmspe_ratio"],
            "att_sc_is_significant_fit_adjusted": inference_sc["is_significant_fit_adjusted"],
            "att_sc_fit_adjusted_warning": inference_sc["fit_adjusted_warning"],
            "att_sc_placebo_treated_rmspe_ratio": inference_sc["treated_rmspe_ratio"],
            "att_sc_placebo_ci_lower_absolute": inference_sc["ci_lower_absolute"],
            "att_sc_placebo_ci_upper_absolute": inference_sc["ci_upper_absolute"],
            "att_sc_placebo_min_possible_p": inference_sc["placebo_min_possible_p"],
            "att_sc_placebo_ci_is_unbounded": inference_sc["placebo_ci_is_unbounded"],
            "att_sc_placebo_att_distribution": inference_sc["placebo_att_distribution"],
            "att_sc_placebo_rmspe_ratio_distribution": inference_sc[
                "placebo_rmspe_ratio_distribution"
            ],
            "att_sc_relative_baseline_post_synthetic": inference_sc["baseline_post_synthetic"],
            "inference_policy": self.inference_policy,
        }

        self._att_sc = att_sc
        self._att_aug = att_aug
        self._ci_upper_absolute = inference_aug["ci_upper_absolute"]
        self._ci_lower_absolute = inference_aug["ci_lower_absolute"]
        self._value_relative = inference_aug["value_relative"]
        self._ci_upper_relative = inference_aug["ci_upper_relative"]
        self._ci_lower_relative = inference_aug["ci_lower_relative"]
        self._ci_alpha = inference_aug["alpha"]
        self._p_value = inference_aug["p_value"]
        self._is_significant = inference_aug["is_significant"]
        self._is_fitted = True
        return self


RSCM = RobustSyntheticControl


class SyntheticControl:
    """Auto-selecting Synthetic Control estimator.

    Parameters
    ----------
    lambda_aug : float, default=1.0
        Ridge penalty used by the augmented donor-weight step.
    lambda_sc : float, default=1e-6
        L2 regularization used in simplex-constrained SC weight optimization.
    max_iter : int, default=2000
        Maximum optimizer iterations for simplex-constrained SC weights.
    tol : float, default=1e-9
        Numerical tolerance used by the SC optimizer.
    enforce_sum_to_one_augmented : bool, default=True
        If ``True``, project augmented donor weights to sum to one.
    completion_max_iter : int, default=500
        Maximum matrix-completion iterations used by robust SC.
    completion_tol : float, default=1e-6
        Relative convergence tolerance for robust SC matrix completion.
    sv_threshold : float or None, default=None
        Absolute singular-value shrinkage threshold for robust completion.
        If ``None``, ``sv_threshold_ratio`` times the leading singular value
        of the initialized matrix is used.
    sv_threshold_ratio : float, default=0.5
        Relative threshold used when ``sv_threshold`` is ``None``.
    max_rank : int or None, default=None
        Optional cap on effective rank in robust matrix completion.
    min_pre_observed : int, default=1
        Minimum number of observed pre-treatment outcomes required for the
        treated unit and each retained donor in robust SC.

    Notes
    -----
    Model selection is performed during :meth:`fit`. If any treated/donor
    analysis-block cell is missing or marked unobserved, robust SC is used;
    otherwise augmented SC is used.
    """

    def __init__(
        self,
        *,
        lambda_aug: float = 1.0,
        lambda_sc: float = 1e-6,
        max_iter: int = 2_000,
        tol: float = 1e-9,
        enforce_sum_to_one_augmented: bool = True,
        completion_max_iter: int = 500,
        completion_tol: float = 1e-6,
        sv_threshold: float | None = None,
        sv_threshold_ratio: float = 0.5,
        max_rank: int | None = None,
        min_pre_observed: int = 1,
        inference_policy: Literal["placebo"] = "placebo",
    ) -> None:
        """Initialize synthetic-control selection and delegate hyperparameters.

        Parameters
        ----------
        lambda_aug : float, default=1.0
            Ridge penalty used by the augmented donor-weight step.
        lambda_sc : float, default=1e-6
            L2 regularization used in simplex-constrained SC weight optimization.
        max_iter : int, default=2000
            Maximum optimizer iterations for simplex-constrained SC weights.
        tol : float, default=1e-9
            Numerical tolerance used by the SC optimizer.
        enforce_sum_to_one_augmented : bool, default=True
            If ``True``, project augmented donor weights to sum to one.
        completion_max_iter : int, default=500
            Maximum matrix-completion iterations used by robust SC.
        completion_tol : float, default=1e-6
            Relative convergence tolerance for robust SC matrix completion.
        sv_threshold : float or None, default=None
            Absolute singular-value shrinkage threshold for robust completion.
        sv_threshold_ratio : float, default=0.5
            Relative threshold used when ``sv_threshold`` is ``None``.
        max_rank : int or None, default=None
            Optional cap on effective rank in robust matrix completion.
        min_pre_observed : int, default=1
            Minimum observed treated/donor pre-period outcomes for robust SC.
        """
        self.lambda_aug = float(lambda_aug)
        self.lambda_sc = float(lambda_sc)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.enforce_sum_to_one_augmented = bool(enforce_sum_to_one_augmented)
        self.completion_max_iter = int(completion_max_iter)
        self.completion_tol = float(completion_tol)
        self.sv_threshold = None if sv_threshold is None else float(sv_threshold)
        self.sv_threshold_ratio = float(sv_threshold_ratio)
        self.max_rank = None if max_rank is None else int(max_rank)
        self.min_pre_observed = int(min_pre_observed)
        self.inference_policy = str(inference_policy)

        # Reuse delegate constructors to run one source of truth for
        # hyperparameter validation (and keep guardrails aligned).
        self._build_ascm()
        self._build_rscm()

        self._is_fitted: bool = False
        self._delegate: AugmentedSyntheticControl | RobustSyntheticControl | None = None
        self._selected_model: str | None = None
        self._selection_reason: str | None = None

    def _build_ascm(self) -> AugmentedSyntheticControl:
        return AugmentedSyntheticControl(
            lambda_aug=self.lambda_aug,
            lambda_sc=self.lambda_sc,
            max_iter=self.max_iter,
            tol=self.tol,
            enforce_sum_to_one_augmented=self.enforce_sum_to_one_augmented,
            inference_policy=self.inference_policy,
        )

    def _build_rscm(self) -> RobustSyntheticControl:
        return RobustSyntheticControl(
            lambda_aug=self.lambda_aug,
            lambda_sc=self.lambda_sc,
            max_iter=self.max_iter,
            tol=self.tol,
            enforce_sum_to_one_augmented=self.enforce_sum_to_one_augmented,
            completion_max_iter=self.completion_max_iter,
            completion_tol=self.completion_tol,
            sv_threshold=self.sv_threshold,
            sv_threshold_ratio=self.sv_threshold_ratio,
            max_rank=self.max_rank,
            min_pre_observed=self.min_pre_observed,
            inference_policy=self.inference_policy,
        )

    def fit(self, data: PanelDataSCM) -> "SyntheticControl":
        """Fit by selecting augmented SC or robust SC from observed missingness.

        Parameters
        ----------
        data : PanelDataSCM
            Panel data contract containing treated unit, donors, and pre/post
            time windows.

        Returns
        -------
        SyntheticControl
            Fitted estimator with an internally selected delegate model.

        Raises
        ------
        ValueError
            If ``data`` is not a ``PanelDataSCM`` object or delegate fit-time
            validation fails.
        """
        if not isinstance(data, PanelDataSCM):
            raise ValueError("Input must be a PanelDataSCM object.")

        # Reset fit state before starting so a failed refit cannot leak a
        # previous successful estimate through .estimate().
        self._is_fitted = False
        self._delegate = None
        self._selected_model = None
        self._selection_reason = None

        # Probe the analysis block via robust preparation because it has the
        # complete missingness semantics (NaN and optional observed_col flags).
        probe_model = self._build_rscm()
        _, observed_mask, _, _, _, _ = probe_model._prepare_incomplete_panel(data)
        has_missing = bool((~observed_mask).any())

        if has_missing:
            # Reuse the probe instance to avoid preparing the incomplete panel twice.
            self._delegate = probe_model.fit(data)
            self._selected_model = "RobustSyntheticControl"
            self._selection_reason = (
                "Selected RobustSyntheticControl because missing or unobserved "
                "analysis-block outcomes were detected."
            )
        else:
            self._delegate = self._build_ascm().fit(data)
            self._selected_model = "AugmentedSyntheticControl"
            self._selection_reason = (
                "Selected AugmentedSyntheticControl because the analysis block "
                "is fully observed."
            )

        self._is_fitted = True
        return self

    def estimate(self) -> PanelEstimate:
        """Return estimate from the selected synthetic-control delegate model.

        Returns
        -------
        PanelEstimate
            Delegate estimate with selection diagnostics appended:
            ``selected_model`` and ``selection_reason``.

        Raises
        ------
        RuntimeError
            If the estimator has not been successfully fitted.
        """
        if not self._is_fitted or self._delegate is None:
            raise RuntimeError("Model must be fitted with .fit(data) before calling .estimate().")

        estimate = self._delegate.estimate()
        # Copy diagnostics to avoid mutating nested state returned by delegates
        # across repeated calls.
        diagnostics = dict(estimate.diagnostics)
        diagnostics["selected_model"] = self._selected_model
        diagnostics["selection_reason"] = self._selection_reason
        estimate.diagnostics = diagnostics
        return estimate

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        selected = self._selected_model if self._selected_model is not None else "n/a"
        return f"{self.__class__.__name__}(status='{status}', selected='{selected}')"


SCM = SyntheticControl
