from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Hashable, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.data_contracts.panel_estimate import PanelEstimate


@dataclass(frozen=True)
class _PointwiseConformalResult:
    effect_by_time: pd.Series
    ci_lower_by_time: pd.Series
    ci_upper_by_time: pd.Series
    p_value_by_time: pd.Series
    is_significant_by_time: pd.Series
    confidence_set_by_time: dict[Any, list[tuple[float, float]]]
    grid_by_time: dict[Any, list[float]]
    grid_p_values_by_time: dict[Any, list[float]]


class AugmentedSyntheticControl:
    """
    Augmented Synthetic Control with CWZ pointwise conformal inference only.

    Breaking changes:
    - only one model path (balanced panel, no robust/matrix-completion route)
    - only one inference path (CWZ moving-block permutation conformal pointwise inversion)
    - only one estimand output shape: dynamic effect path

    Estimand
    --------
    Dynamic post-treatment effect path theta_t for each post-treatment period t.

    Point estimate
    --------------
    Plug-in post-period gap path from pre-period-fitted augmented SC weights.

    Inference
    ---------
    For each post-treatment period t and candidate theta_t^0 on a grid:
    1) Build reduced sample: [all pre periods] + [target post period t].
    2) Impose sharp null on only target post outcome.
    3) Refit ASCM under the null on reduced sample.
    4) Compute residuals and permutation p-value.
    5) Invert test to obtain pointwise CI for theta_t.

    Notes
    -----
    - Inference is based on circular moving-block permutations, so validity is
      approximate under weak dependence rather than exact finite-sample.
    - Returned confidence sets are grid-approximated contiguous accepted segments.
    """

    def __init__(
        self,
        *,
        lambda_aug: float = 1.0,
        lambda_sc: float = 1e-6,
        max_iter: int = 2_000,
        tol: float = 1e-9,
        enforce_sum_to_one_augmented: bool = True,
        alpha: float = 0.05,
        conformal_grid_size: int = 401,
        conformal_grid_min: float | None = None,
        conformal_grid_max: float | None = None,
        conformal_grid_scale_mult: float = 6.0,
    ) -> None:
        self.lambda_aug = float(lambda_aug)
        self.lambda_sc = float(lambda_sc)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.enforce_sum_to_one_augmented = bool(enforce_sum_to_one_augmented)

        self.alpha = float(alpha)
        self.conformal_grid_size = int(conformal_grid_size)
        self.conformal_grid_min = None if conformal_grid_min is None else float(conformal_grid_min)
        self.conformal_grid_max = None if conformal_grid_max is None else float(conformal_grid_max)
        self.conformal_grid_scale_mult = float(conformal_grid_scale_mult)

        if not np.isfinite(self.lambda_aug) or self.lambda_aug < 0.0:
            raise ValueError("lambda_aug must be finite and >= 0.")
        if not np.isfinite(self.lambda_sc) or self.lambda_sc < 0.0:
            raise ValueError("lambda_sc must be finite and >= 0.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and > 0.")
        if not np.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be finite and in (0, 1).")
        if self.conformal_grid_size < 3 or self.conformal_grid_size % 2 == 0:
            raise ValueError("conformal_grid_size must be an odd integer >= 3.")
        if not np.isfinite(self.conformal_grid_scale_mult) or self.conformal_grid_scale_mult <= 0.0:
            raise ValueError("conformal_grid_scale_mult must be finite and > 0.")
        if (
            self.conformal_grid_min is not None
            and self.conformal_grid_max is not None
            and self.conformal_grid_min >= self.conformal_grid_max
        ):
            raise ValueError("conformal_grid_min must be < conformal_grid_max.")

        self._is_fitted: bool = False
        self._data: PanelDataSCM | None = None

    # ---------------------------------------------------------------------
    # Core numerics
    # ---------------------------------------------------------------------

    @staticmethod
    def _rmse(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(arr))))

    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(a, b, rcond=None)[0]

    def _fit_simplex_weights_projected_gradient(
        self,
        *,
        gram: np.ndarray,
        rhs: np.ndarray,
        w_init: np.ndarray,
    ) -> np.ndarray:
        gram_arr = np.asarray(gram, dtype=float)
        rhs_arr = np.asarray(rhs, dtype=float)
        w = self._project_to_simplex(np.asarray(w_init, dtype=float))

        if gram_arr.shape[0] != gram_arr.shape[1]:
            raise ValueError("gram must be square.")
        if rhs_arr.shape != (gram_arr.shape[0],):
            raise ValueError("rhs must have shape (n_donors,).")

        try:
            eigvals = np.linalg.eigvalsh(gram_arr)
            lmax = float(np.max(eigvals)) if eigvals.size > 0 else 0.0
        except np.linalg.LinAlgError:
            lmax = float(np.linalg.norm(gram_arr, ord=2))
        lipschitz = float(max(2.0 * lmax, 1e-12))
        step = 1.0 / lipschitz

        max_iter_pg = int(max(2_000, 5 * self.max_iter))
        tol_eff = float(max(self.tol, 1e-10))
        for _ in range(max_iter_pg):
            grad = 2.0 * (gram_arr @ w - rhs_arr)
            if not np.isfinite(grad).all():
                break
            w_next = self._project_to_simplex(w - step * grad)
            if float(np.max(np.abs(w_next - w))) <= tol_eff:
                return w_next
            w = w_next

        return self._project_to_simplex(w)

    def _fit_simplex_weights(self, *, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.ndim != 2:
            raise ValueError("x must be 2D with shape (n_periods, n_donors).")
        if y_arr.ndim != 1:
            raise ValueError("y must be 1D with shape (n_periods,).")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must have the same number of rows.")
        if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
            raise ValueError("x and y must contain only finite values.")

        n_donors = int(x_arr.shape[1])
        if n_donors < 1:
            raise ValueError("At least one donor is required.")

        w0 = np.full(n_donors, 1.0 / float(n_donors), dtype=float)
        gram = x_arr.T @ x_arr + self.lambda_sc * np.eye(n_donors, dtype=float)
        rhs = x_arr.T @ y_arr

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
            return self._project_to_simplex(np.asarray(result.x, dtype=float))

        return self._fit_simplex_weights_projected_gradient(
            gram=gram,
            rhs=rhs,
            w_init=w0,
        )

    def _augment_weights(self, *, x: np.ndarray, y: np.ndarray, w_sc: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        w_sc_arr = np.asarray(w_sc, dtype=float)

        n_donors = int(x_arr.shape[1])
        gram = x_arr.T @ x_arr + self.lambda_aug * np.eye(n_donors, dtype=float)
        rhs = x_arr.T @ y_arr + self.lambda_aug * w_sc_arr
        w_aug = self._solve_linear(gram, rhs)

        if self.enforce_sum_to_one_augmented:
            ones = np.ones(n_donors, dtype=float)
            gram_inv_ones = self._solve_linear(gram, ones)
            denom = float(ones @ gram_inv_ones)
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                raise RuntimeError("Augmented constraint system is ill-conditioned.")
            correction = gram_inv_ones * ((float(np.sum(w_aug)) - 1.0) / denom)
            w_aug = w_aug - correction

        return np.asarray(w_aug, dtype=float)

    def _fit_augmented_weights(self, *, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w_sc = self._fit_simplex_weights(x=x, y=y)
        w_aug = self._augment_weights(x=x, y=y, w_sc=w_sc)
        return w_sc, w_aug

    # ---------------------------------------------------------------------
    # Panel preparation
    # ---------------------------------------------------------------------

    @staticmethod
    def _validate_pre_post_times(
        *,
        pre_times: List[Any],
        post_times: List[Any],
    ) -> None:
        if len(pre_times) < 1:
            raise ValueError("fit() requires at least one pre-treatment period.")
        if len(post_times) < 1:
            raise ValueError("fit() requires at least one post-treatment period.")

        overlap = set(pre_times).intersection(post_times)
        if overlap:
            raise ValueError(
                f"fit() requires disjoint pre/post periods; overlapping times found: {sorted(overlap)}."
            )
        try:
            pre_sorted = sorted(pre_times)
            post_sorted = sorted(post_times)
        except TypeError as exc:
            raise ValueError("fit() requires comparable pre/post period labels.") from exc
        if list(pre_times) != pre_sorted or list(post_times) != post_sorted:
            raise ValueError("fit() requires pre/post periods sorted in ascending order.")
        if max(pre_times) >= min(post_times):
            raise ValueError("fit() requires all pre-treatment times < all post-treatment times.")

    def _prepare_balanced_panel(
        self,
        data: PanelDataSCM,
    ) -> tuple[pd.DataFrame, list[Hashable], list[Any], list[Any], list[Any]]:
        df = data.df_analysis()
        unit_col = data.unit_col
        time_col = data.time_col
        outcome_col = data.y

        donors = list(data.donor_pool())
        if len(donors) < 1:
            raise ValueError("At least one donor unit is required.")

        pre_times = list(data.pre_times())
        post_times = list(data.post_times())
        self._validate_pre_post_times(pre_times=pre_times, post_times=post_times)

        all_times = list(pre_times) + list(post_times)
        keep_units = [data.treated_unit] + donors
        block = df[df[unit_col].isin(keep_units) & df[time_col].isin(all_times)].copy()

        if bool(block.duplicated([unit_col, time_col]).any()):
            raise ValueError(
                "fit() requires unique (unit, time) rows in the analysis block. "
                "Aggregate duplicated rows before fitting."
            )

        panel = block.pivot(index=unit_col, columns=time_col, values=outcome_col)
        panel = panel.reindex(index=keep_units, columns=all_times)

        if bool(panel.isna().any().any()):
            mask = panel.isna().to_numpy()
            row_ids, col_ids = np.where(mask)
            examples = [f"({panel.index[r]!r}, {panel.columns[c]!r})" for r, c in zip(row_ids[:5], col_ids[:5])]
            raise ValueError(
                "This estimator requires a fully observed balanced block for treated + donors. "
                f"Missing unit-time cells include: {', '.join(examples)}."
            )

        return panel, donors, pre_times, post_times, all_times

    # ---------------------------------------------------------------------
    # Conformal inference
    # ---------------------------------------------------------------------

    @staticmethod
    def _moving_block_permutation_indices(n_total: int) -> list[np.ndarray]:
        base = np.arange(n_total, dtype=int)
        return [np.roll(base, -shift) for shift in range(n_total)]

    @staticmethod
    def _conformal_stat_from_residuals(
        residuals: np.ndarray,
        *,
        n_pre: int,
    ) -> float:
        """
        Average-effect conformal statistic:
            S(u_hat) = |sum_{post} u_hat_t| / sqrt(T_post)
        """
        resid = np.asarray(residuals, dtype=float).reshape(-1)
        n_total = int(resid.size)
        if n_pre < 1 or n_pre >= n_total:
            raise ValueError("n_pre must satisfy 1 <= n_pre < len(residuals).")
        post = resid[n_pre:]
        n_post = int(post.size)
        return float(abs(np.sum(post)) / np.sqrt(float(n_post)))

    def _permutation_p_value(
        self,
        *,
        residuals: np.ndarray,
        n_pre: int,
    ) -> float:
        resid = np.asarray(residuals, dtype=float).reshape(-1)
        observed = self._conformal_stat_from_residuals(resid, n_pre=n_pre)

        perm_stats = np.empty(resid.size, dtype=float)
        for k, idx in enumerate(self._moving_block_permutation_indices(resid.size)):
            perm_stats[k] = self._conformal_stat_from_residuals(resid[idx], n_pre=n_pre)

        p_value = float(np.mean(perm_stats >= (observed - 1e-12)))
        return float(np.clip(p_value, 0.0, 1.0))

    @staticmethod
    def _accepted_segments(grid: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        grid_arr = np.asarray(grid, dtype=float)
        mask_arr = np.asarray(mask, dtype=bool)
        if grid_arr.ndim != 1 or mask_arr.ndim != 1 or grid_arr.size != mask_arr.size:
            raise ValueError("grid and mask must be 1D with the same length.")
        if grid_arr.size == 0:
            return []

        segments: list[tuple[float, float]] = []
        start_idx: int | None = None
        for i, accepted in enumerate(mask_arr):
            if accepted and start_idx is None:
                start_idx = i
            elif (not accepted) and start_idx is not None:
                segments.append((float(grid_arr[start_idx]), float(grid_arr[i - 1])))
                start_idx = None
        if start_idx is not None:
            segments.append((float(grid_arr[start_idx]), float(grid_arr[-1])))
        return segments

    def _build_conformal_grid(
        self,
        *,
        point_estimate: float,
        pre_residuals: np.ndarray,
    ) -> np.ndarray:
        if self.conformal_grid_min is not None and self.conformal_grid_max is not None:
            return np.linspace(
                self.conformal_grid_min,
                self.conformal_grid_max,
                self.conformal_grid_size,
                dtype=float,
            )

        pre_resid = np.asarray(pre_residuals, dtype=float).reshape(-1)
        scale = max(
            self._rmse(pre_resid),
            float(np.std(pre_resid, ddof=1)) if pre_resid.size > 1 else 0.0,
            1e-8,
        )
        half_width = max(
            2.0 * abs(float(point_estimate)),
            self.conformal_grid_scale_mult * scale,
            1e-6,
        )

        grid_min = (
            self.conformal_grid_min
            if self.conformal_grid_min is not None
            else float(point_estimate) - half_width
        )
        grid_max = (
            self.conformal_grid_max
            if self.conformal_grid_max is not None
            else float(point_estimate) + half_width
        )
        if not np.isfinite(grid_min) or not np.isfinite(grid_max) or grid_min >= grid_max:
            raise RuntimeError("Invalid conformal grid bounds.")
        return np.linspace(grid_min, grid_max, self.conformal_grid_size, dtype=float)

    def _pointwise_null_residuals_for_theta_t(
        self,
        *,
        y1_observed: np.ndarray,
        y0_all: np.ndarray,
        n_pre: int,
        target_post_offset: int,
        theta_t_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        CWZ pointwise null for a single post-treatment period.

        Reduced sample:
            [all pre periods] + [target post period]
        with null imposed only on the target post outcome.
        """
        target_abs_idx = n_pre + int(target_post_offset)
        if target_abs_idx < n_pre or target_abs_idx >= y1_observed.shape[0]:
            raise ValueError("target_post_offset is out of range.")

        reduced_idx = np.concatenate(
            [np.arange(n_pre, dtype=int), np.array([target_abs_idx], dtype=int)]
        )

        y1_reduced = np.asarray(y1_observed[reduced_idx], dtype=float).copy()
        x_reduced = np.asarray(y0_all[reduced_idx, :], dtype=float)

        y1_reduced[-1] = y1_reduced[-1] - float(theta_t_value)

        _, w_aug_null = self._fit_augmented_weights(x=x_reduced, y=y1_reduced)
        residuals = y1_reduced - (x_reduced @ w_aug_null)
        return residuals, w_aug_null

    def _invert_pointwise_ci_for_post_period(
        self,
        *,
        y1_observed: np.ndarray,
        y0_all: np.ndarray,
        n_pre: int,
        target_post_offset: int,
        point_estimate_t: float,
        pre_residuals: np.ndarray,
    ) -> tuple[float | None, float | None, float, list[tuple[float, float]], np.ndarray, np.ndarray]:
        grid = self._build_conformal_grid(
            point_estimate=point_estimate_t,
            pre_residuals=pre_residuals,
        )

        p_values = np.empty(grid.size, dtype=float)
        for i, theta0 in enumerate(grid):
            residuals_theta, _ = self._pointwise_null_residuals_for_theta_t(
                y1_observed=y1_observed,
                y0_all=y0_all,
                n_pre=n_pre,
                target_post_offset=target_post_offset,
                theta_t_value=float(theta0),
            )
            p_values[i] = self._permutation_p_value(residuals=residuals_theta, n_pre=n_pre)

        accepted_mask = p_values > float(self.alpha)
        accepted_segments = self._accepted_segments(grid, accepted_mask)

        ci_lower: float | None = None
        ci_upper: float | None = None
        if len(accepted_segments) == 1:
            ci_lower = float(accepted_segments[0][0])
            ci_upper = float(accepted_segments[0][1])

        if len(accepted_segments) == 0:
            warnings.warn(
                "No grid point was accepted for this post period; widen or refine the conformal grid.",
                RuntimeWarning,
                stacklevel=2,
            )

        if bool(np.any(accepted_mask)):
            if bool(accepted_mask[0]) or bool(accepted_mask[-1]):
                warnings.warn(
                    "Pointwise conformal confidence set touches the grid boundary; widen the grid.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        zero_residuals, _ = self._pointwise_null_residuals_for_theta_t(
            y1_observed=y1_observed,
            y0_all=y0_all,
            n_pre=n_pre,
            target_post_offset=target_post_offset,
            theta_t_value=0.0,
        )
        p_value_at_zero = self._permutation_p_value(residuals=zero_residuals, n_pre=n_pre)

        return (
            ci_lower,
            ci_upper,
            float(p_value_at_zero),
            accepted_segments,
            grid,
            p_values,
        )

    def _invert_pointwise_conformal_tests(
        self,
        *,
        y1_observed: np.ndarray,
        y0_all: np.ndarray,
        n_pre: int,
        post_times: list[Any],
        effect_by_time: pd.Series,
        pre_residuals: np.ndarray,
    ) -> _PointwiseConformalResult:
        post_index = pd.Index(post_times)
        ci_lower_map: dict[Any, float] = {}
        ci_upper_map: dict[Any, float] = {}
        p_value_map: dict[Any, float] = {}
        significant_map: dict[Any, bool] = {}
        confidence_set_map: dict[Any, list[tuple[float, float]]] = {}
        grid_map: dict[Any, list[float]] = {}
        grid_p_map: dict[Any, list[float]] = {}

        for post_offset, time_key in enumerate(post_times):
            point_estimate_t = float(effect_by_time.loc[time_key])

            (
                ci_low_t,
                ci_high_t,
                p_zero_t,
                accepted_segments_t,
                grid_t,
                pvals_t,
            ) = self._invert_pointwise_ci_for_post_period(
                y1_observed=y1_observed,
                y0_all=y0_all,
                n_pre=n_pre,
                target_post_offset=post_offset,
                point_estimate_t=point_estimate_t,
                pre_residuals=pre_residuals,
            )

            ci_lower_map[time_key] = np.nan if ci_low_t is None else float(ci_low_t)
            ci_upper_map[time_key] = np.nan if ci_high_t is None else float(ci_high_t)
            p_value_map[time_key] = float(p_zero_t)
            significant_map[time_key] = bool(p_zero_t <= float(self.alpha))
            confidence_set_map[time_key] = accepted_segments_t
            grid_map[time_key] = grid_t.tolist()
            grid_p_map[time_key] = pvals_t.tolist()

        return _PointwiseConformalResult(
            effect_by_time=effect_by_time.reindex(post_index).copy(),
            ci_lower_by_time=pd.Series(
                ci_lower_map,
                index=post_index,
                dtype=float,
                name="ci_lower_by_time",
            ),
            ci_upper_by_time=pd.Series(
                ci_upper_map,
                index=post_index,
                dtype=float,
                name="ci_upper_by_time",
            ),
            p_value_by_time=pd.Series(
                p_value_map,
                index=post_index,
                dtype=float,
                name="p_value_by_time",
            ),
            is_significant_by_time=pd.Series(
                significant_map,
                index=post_index,
                dtype=bool,
                name="is_significant_by_time",
            ),
            confidence_set_by_time=confidence_set_map,
            grid_by_time=grid_map,
            grid_p_values_by_time=grid_p_map,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit(self, data: PanelDataSCM) -> "AugmentedSyntheticControl":
        # Reset fit state so failed refits cannot leak stale estimates.
        self._is_fitted = False
        self._data = None

        if not isinstance(data, PanelDataSCM):
            raise ValueError("Input must be a PanelDataSCM object.")

        panel, donors, pre_times, post_times, all_times = self._prepare_balanced_panel(data)

        treated = data.treated_unit
        y1_all = panel.loc[treated, all_times].to_numpy(dtype=float)
        y1_pre = panel.loc[treated, pre_times].to_numpy(dtype=float)

        y0_all = panel.loc[donors, all_times].to_numpy(dtype=float).T
        x0_pre = panel.loc[donors, pre_times].to_numpy(dtype=float).T

        _, w_aug = self._fit_augmented_weights(x=x0_pre, y=y1_pre)

        y0_hat = y0_all @ w_aug
        gap = y1_all - y0_hat
        n_pre = len(pre_times)
        pre_residuals = y1_pre - (x0_pre @ w_aug)
        min_possible_p = 1.0 / float(n_pre + 1)

        if self.alpha < min_possible_p:
            warnings.warn(
                f"alpha={self.alpha:.3f} is smaller than the minimum attainable pointwise p-value "
                f"{min_possible_p:.3f}; rejection is impossible with the current number of "
                "pre-treatment periods.",
                RuntimeWarning,
                stacklevel=2,
            )
        if n_pre < 10:
            warnings.warn(
                "Very short pre-treatment window detected; moving-block pointwise p-values will be "
                "highly discrete and inference may be unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

        observed_series = pd.Series(y1_all, index=all_times, name="observed_outcome")
        synthetic_series = pd.Series(y0_hat, index=all_times, name="synthetic_outcome")
        gap_series = pd.Series(gap, index=all_times, name="gap")
        effect_by_time = gap_series.loc[post_times].copy()
        effect_by_time.name = "effect_by_time"

        pointwise = self._invert_pointwise_conformal_tests(
            y1_observed=y1_all,
            y0_all=y0_all,
            n_pre=n_pre,
            post_times=post_times,
            effect_by_time=effect_by_time,
            pre_residuals=pre_residuals,
        )

        gram_aug = x0_pre.T @ x0_pre + self.lambda_aug * np.eye(x0_pre.shape[1], dtype=float)
        cond_gram_aug = float(np.linalg.cond(gram_aug))
        l1_w_aug = float(np.sum(np.abs(w_aug)))
        max_abs_w_aug = float(np.max(np.abs(w_aug)))

        if cond_gram_aug > 1e10:
            warnings.warn(
                f"Augmented normal equations are ill-conditioned (cond={cond_gram_aug:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        if l1_w_aug > 5.0 or max_abs_w_aug > 2.0:
            warnings.warn(
                "Augmented donor weights are extreme; estimates may be unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._data = data
        self._donors = list(donors)
        self._pre_times = list(pre_times)
        self._post_times = list(post_times)
        self._all_times = list(all_times)
        self._w_aug = np.asarray(w_aug, dtype=float)

        self._observed = observed_series
        self._synthetic = synthetic_series
        self._gap = gap_series

        self._effect_by_time = pointwise.effect_by_time
        self._ci_lower_by_time = pointwise.ci_lower_by_time
        self._ci_upper_by_time = pointwise.ci_upper_by_time
        self._p_value_by_time = pointwise.p_value_by_time
        self._is_significant_by_time = pointwise.is_significant_by_time
        self._confidence_set_by_time = {
            key: [(float(lo), float(hi)) for lo, hi in segments]
            for key, segments in pointwise.confidence_set_by_time.items()
        }

        self._diagnostics = {
            "n_donors": len(self._donors),
            "n_pre_periods": len(self._pre_times),
            "n_post_periods": len(self._post_times),
            "enforce_sum_to_one_augmented": bool(self.enforce_sum_to_one_augmented),
            "pre_rmse_augmented": self._rmse(pre_residuals),
            "sum_weights_augmented": float(np.sum(w_aug)),
            "min_weight_augmented": float(np.min(w_aug)),
            "max_weight_augmented": float(np.max(w_aug)),
            "l1_norm_weights_augmented": l1_w_aug,
            "max_abs_weight_augmented": max_abs_w_aug,
            "cond_augmented_gram": cond_gram_aug,
            "estimand": "dynamic_effect_path",
            "ci_alpha": float(self.alpha),
            "pointwise_ci_method": "cwz_permutation_conformal_pointwise_moving_block_approximate",
            "pointwise_min_possible_p_value": min_possible_p,
            "pointwise_rejection_possible_at_alpha": bool(self.alpha >= min_possible_p),
            "pointwise_confidence_set_representation": "grid_approximated_contiguous_segments",
            "pointwise_grid_by_time": pointwise.grid_by_time,
            "pointwise_grid_p_values_by_time": pointwise.grid_p_values_by_time,
            "pointwise_confidence_set_by_time": self._confidence_set_by_time,
            "pointwise_accepted_sets_by_time": self._confidence_set_by_time,
        }

        self._is_fitted = True
        return self

    def estimate(self) -> PanelEstimate:
        if not self._is_fitted or self._data is None:
            raise RuntimeError("Model must be fitted with .fit(data) before calling .estimate().")

        return PanelEstimate(
            estimand="dynamic_effect_path",
            model=self.__class__.__name__,
            treated_unit=self._data.treated_unit,
            treatment_start=self._data.treatment_start,
            pre_times=list(self._pre_times),
            post_times=list(self._post_times),
            effect_by_time=self._effect_by_time.copy(),
            ci_lower_by_time=self._ci_lower_by_time.copy(),
            ci_upper_by_time=self._ci_upper_by_time.copy(),
            p_value_by_time=self._p_value_by_time.copy(),
            is_significant_by_time=self._is_significant_by_time.copy(),
            confidence_set_by_time={
                key: [(float(lo), float(hi)) for lo, hi in segments]
                for key, segments in self._confidence_set_by_time.items()
            },
            alpha=float(self.alpha),
            observed_outcome=self._observed.copy(),
            synthetic_outcome=self._synthetic.copy(),
            donor_weights_augmented={
                donor: float(weight) for donor, weight in zip(self._donors, self._w_aug)
            },
            diagnostics=dict(self._diagnostics),
        )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(status='{status}')"


ASCM = AugmentedSyntheticControl
