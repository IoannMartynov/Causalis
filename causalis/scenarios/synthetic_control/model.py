from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Hashable, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.data_contracts.panel_estimate import PanelEstimate
from causalis.scenarios.synthetic_control._utils import (
    accepted_segments,
    build_average_att_blocks,
    circular_shift_indices,
    cwz_stat_from_residuals,
    project_to_simplex,
    rmse,
    solve_linear_system,
)


@dataclass(frozen=True)
class _PointwiseConformalResult:
    """Internal container for post-period pointwise inference outputs."""

    effect_by_time: pd.Series
    ci_lower_by_time: pd.Series
    ci_upper_by_time: pd.Series
    # Per-post-period p-values (pointwise), not a joint post-window p-value.
    p_value_by_time: pd.Series
    is_significant_by_time: pd.Series
    confidence_set_by_time: dict[Any, list[tuple[float, float]]]
    grid_by_time: dict[Any, list[float]]
    grid_p_values_by_time: dict[Any, list[float]]


@dataclass(frozen=True)
class _AverageATTTTestResult:
    """Internal container for post-window ATT t-test inference."""

    available: bool
    message: str | None
    att: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    t_stat: float | None
    standard_error: float | None
    sigma_hat: float | None
    fold_estimates: pd.Series
    fold_blocks: dict[str, list[Any]]
    n_folds: int
    block_length: int


class AugmentedSyntheticControl:
    """Augmented Synthetic Control with aggregate-first inference.

    Notes
    -----
    Average ATT t-test inference is the default post-treatment inference layer.
    Pointwise conformal intervals/p-values are optional and can be enabled for
    dynamic path uncertainty quantification.
    """

    _AUGMENTED_GRAM_COND_WARN_THRESHOLD = 1e10
    _AUGMENTED_WEIGHT_L1_WARN_THRESHOLD = 5.0
    _AUGMENTED_WEIGHT_MAX_ABS_WARN_THRESHOLD = 2.0
    _DEGENERATE_SIGMA_TOL = 1e-15

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
        average_att_n_folds: int = 3,
        compute_average_att_ttest: bool = True,
        compute_pointwise_conformal: bool = False,
    ) -> None:
        """Initialize ASCM hyperparameters.

        Parameters
        ----------
        lambda_aug : float, default=1.0
            Ridge regularization for augmented weights.
        lambda_sc : float, default=1e-6
            Numerical regularization for simplex SCM weights.
        max_iter : int, default=2000
            Maximum iterations for constrained optimization routines.
        tol : float, default=1e-9
            Optimization tolerance.
        enforce_sum_to_one_augmented : bool, default=True
            Enforce sum-to-one constraint on augmented weights.
        alpha : float, default=0.05
            Significance level for confidence intervals and tests.
        conformal_grid_size : int, default=401
            Number of grid points used in pointwise conformal inversion.
        conformal_grid_min : float or None, default=None
            Optional fixed lower bound for conformal grid.
        conformal_grid_max : float or None, default=None
            Optional fixed upper bound for conformal grid.
        conformal_grid_scale_mult : float, default=6.0
            Scale multiplier for automatic conformal grid width.
        average_att_n_folds : int, default=3
            Requested number of folds for average ATT t-test inference.
        compute_average_att_ttest : bool, default=True
            Whether to run average ATT t-test inference.
        compute_pointwise_conformal : bool, default=False
            Whether to run pointwise conformal CIs/p-values for each post period.

        Raises
        ------
        ValueError
            If any hyperparameter is invalid.
        """
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

        self.average_att_n_folds = int(average_att_n_folds)
        self.compute_average_att_ttest = bool(compute_average_att_ttest)
        self.compute_pointwise_conformal = bool(compute_pointwise_conformal)

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
        if self.average_att_n_folds < 2:
            raise ValueError("average_att_n_folds must be an integer >= 2.")
        if self.compute_pointwise_conformal:
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
        return rmse(x)

    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        return project_to_simplex(v)

    @staticmethod
    def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return solve_linear_system(a, b)

    def _fit_simplex_weights_projected_gradient(
        self,
        *,
        gram: np.ndarray,
        rhs: np.ndarray,
        w_init: np.ndarray,
    ) -> np.ndarray:
        """Solve simplex-constrained quadratic program via projected gradient.

        Parameters
        ----------
        gram : numpy.ndarray
            Positive semi-definite matrix in the quadratic objective.
        rhs : numpy.ndarray
            Linear term of the objective.
        w_init : numpy.ndarray
            Initial weight vector.

        Returns
        -------
        numpy.ndarray
            Simplex-constrained donor weights.
        """
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
        # Fixed step from the Lipschitz constant of the gradient.
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
        """Estimate donor weights on the simplex.

        Parameters
        ----------
        x : numpy.ndarray
            Donor matrix with shape ``(n_periods, n_donors)``.
        y : numpy.ndarray
            Treated outcomes with shape ``(n_periods,)``.

        Returns
        -------
        numpy.ndarray
            Simplex-constrained donor weights.

        Notes
        -----
        Uses SLSQP first, then falls back to projected-gradient if SLSQP does
        not converge.
        """
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
        """Compute augmented donor weights around simplex baseline weights.

        Parameters
        ----------
        x : numpy.ndarray
            Donor matrix with shape ``(n_periods, n_donors)``.
        y : numpy.ndarray
            Treated outcomes with shape ``(n_periods,)``.
        w_sc : numpy.ndarray
            Baseline simplex SCM weights.

        Returns
        -------
        numpy.ndarray
            Augmented donor weights.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        w_sc_arr = np.asarray(w_sc, dtype=float)

        n_donors = int(x_arr.shape[1])
        gram = x_arr.T @ x_arr + self.lambda_aug * np.eye(n_donors, dtype=float)
        if self.enforce_sum_to_one_augmented and self.lambda_aug == 0.0:
            if int(np.linalg.matrix_rank(gram)) < n_donors:
                raise ValueError(
                    "lambda_aug=0 with enforce_sum_to_one_augmented=True requires a full-rank donor "
                    "Gram matrix; received a singular Gram matrix. Use lambda_aug>0."
                )
        rhs = x_arr.T @ y_arr + self.lambda_aug * w_sc_arr
        w_aug = self._solve_linear(gram, rhs)

        if self.enforce_sum_to_one_augmented:
            ones = np.ones(n_donors, dtype=float)
            gram_inv_ones = self._solve_linear(gram, ones)
            denom = float(ones @ gram_inv_ones)
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                raise RuntimeError("Augmented constraint system is ill-conditioned.")
            # Closed-form Lagrange correction to satisfy sum(w_aug) == 1.
            correction = gram_inv_ones * ((float(np.sum(w_aug)) - 1.0) / denom)
            w_aug = w_aug - correction

        return np.asarray(w_aug, dtype=float)

    def _fit_augmented_weights(self, *, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit both simplex and augmented donor weights.

        Parameters
        ----------
        x : numpy.ndarray
            Donor matrix with shape ``(n_periods, n_donors)``.
        y : numpy.ndarray
            Treated outcomes with shape ``(n_periods,)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(w_sc, w_aug)`` baseline simplex and augmented weights.
        """
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
        """Validate pre/post period partitions.

        Parameters
        ----------
        pre_times : list[Any]
            Sorted pre-treatment time labels.
        post_times : list[Any]
            Sorted post-treatment time labels.

        Raises
        ------
        ValueError
            If periods are empty, overlapping, unsorted, or not strictly split
            by treatment boundary.
        """
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
        """Prepare a fully balanced treated-plus-donors panel block.

        Parameters
        ----------
        data : PanelDataSCM
            Validated SCM panel contract.

        Returns
        -------
        tuple[pandas.DataFrame, list[Hashable], list[Any], list[Any], list[Any]]
            ``(panel, donors, pre_times, post_times, all_times)`` where
            ``panel`` is unit-by-time and fully observed.

        Raises
        ------
        ValueError
            If required donor rows are missing, duplicated, or unbalanced.
        """
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
    # Added: average ATT t-test inference (1812.10820)
    # ---------------------------------------------------------------------

    @staticmethod
    def _empty_average_att_ttest_result(message: str) -> _AverageATTTTestResult:
        """Build a standardized unavailable-result payload.

        Parameters
        ----------
        message : str
            Reason why average ATT inference was not available.

        Returns
        -------
        _AverageATTTTestResult
            Result object with ``available=False`` and empty payload fields.
        """
        return _AverageATTTTestResult(
            available=False,
            message=str(message),
            att=None,
            ci_lower=None,
            ci_upper=None,
            p_value=None,
            t_stat=None,
            standard_error=None,
            sigma_hat=None,
            fold_estimates=pd.Series(dtype=float, name="average_att_fold_estimates"),
            fold_blocks={},
            n_folds=0,
            block_length=0,
        )

    def _build_average_att_blocks(
        self,
        *,
        n_pre: int,
        n_post: int,
    ) -> tuple[list[np.ndarray], int, int]:
        """Create fold holdout blocks for average ATT inference.

        Parameters
        ----------
        n_pre : int
            Number of pre-treatment periods.
        n_post : int
            Number of post-treatment periods.

        Returns
        -------
        tuple[list[numpy.ndarray], int, int]
            ``(blocks, k_used, block_length)``.
        """
        return build_average_att_blocks(
            n_pre=n_pre,
            n_post=n_post,
            n_folds=self.average_att_n_folds,
        )

    def _compute_average_att_ttest(
        self,
        *,
        y1_observed: np.ndarray,
        y0_all: np.ndarray,
        pre_times: list[Any],
        post_times: list[Any],
    ) -> _AverageATTTTestResult:
        """Compute average ATT using a debiased self-normalized t-test.

        Parameters
        ----------
        y1_observed : numpy.ndarray
            Treated outcomes across all analysis periods.
        y0_all : numpy.ndarray
            Donor outcomes matrix with shape ``(n_periods, n_donors)``.
        pre_times : list[Any]
            Pre-treatment time labels.
        post_times : list[Any]
            Post-treatment time labels.

        Returns
        -------
        _AverageATTTTestResult
            Fold-level and aggregate ATT inference results.

        Notes
        -----
        Fold estimator:

        ``tau_hat_k = mean_post(gap_k) - mean_holdout_pre(gap_k)``

        where ``gap_k`` is computed with fold-specific weights fit on the
        complement pre-period sample.
        """
        y1 = np.asarray(y1_observed, dtype=float).reshape(-1)
        x = np.asarray(y0_all, dtype=float)
        n_pre = int(len(pre_times))
        n_post = int(len(post_times))

        if y1.ndim != 1:
            return self._empty_average_att_ttest_result("y1_observed must be one-dimensional.")
        if x.ndim != 2:
            return self._empty_average_att_ttest_result("y0_all must be two-dimensional.")
        if x.shape[0] != y1.size:
            return self._empty_average_att_ttest_result("y1_observed and y0_all must have the same number of rows.")
        if not np.isfinite(y1).all() or not np.isfinite(x).all():
            return self._empty_average_att_ttest_result("Average ATT t-test inputs must be finite.")

        try:
            blocks, k_used, r = self._build_average_att_blocks(n_pre=n_pre, n_post=n_post)
        except ValueError as exc:
            return self._empty_average_att_ttest_result(str(exc))

        all_pre_idx = np.arange(n_pre, dtype=int)
        post_idx = np.arange(n_pre, n_pre + n_post, dtype=int)

        y_post = y1[post_idx]
        x_post = x[post_idx, :]
        tau_hat_k = np.empty(k_used, dtype=float)
        fold_blocks: dict[str, list[Any]] = {}

        for k, holdout_idx in enumerate(blocks, start=1):
            train_mask = np.ones(n_pre, dtype=bool)
            train_mask[holdout_idx] = False
            train_idx = all_pre_idx[train_mask]

            if train_idx.size < 1:
                return self._empty_average_att_ttest_result(
                    "A fold left no pre-treatment periods for training."
                )

            _, w_aug_k = self._fit_augmented_weights(
                x=x[train_idx, :],
                y=y1[train_idx],
            )

            # Debias post mean gap by the fold-specific pre holdout mean gap.
            post_gap_mean = float(np.mean(y_post - (x_post @ w_aug_k)))
            holdout_gap_mean = float(np.mean(y1[holdout_idx] - (x[holdout_idx, :] @ w_aug_k)))
            tau_hat_k[k - 1] = post_gap_mean - holdout_gap_mean
            fold_blocks[f"fold_{k}"] = list(pre_times[int(holdout_idx[0]) : int(holdout_idx[-1]) + 1])

        tau_hat = float(np.mean(tau_hat_k))
        if k_used < 2:
            return self._empty_average_att_ttest_result("Average ATT t-test requires at least two folds.")

        sd_fold = float(np.sqrt(np.sum((tau_hat_k - tau_hat) ** 2) / float(k_used - 1)))
        sigma_hat_tau = float(np.sqrt(1.0 + (k_used * r) / float(n_post)) * sd_fold)
        se_hat = float(sigma_hat_tau / np.sqrt(float(k_used)))

        if not np.isfinite(sigma_hat_tau):
            return self._empty_average_att_ttest_result("Average ATT t-test scale is not finite.")

        df = int(k_used - 1)
        crit = float(student_t.ppf(1.0 - self.alpha / 2.0, df=df))

        if sigma_hat_tau <= self._DEGENERATE_SIGMA_TOL:
            warnings.warn(
                "Average ATT t-test fold dispersion is numerically zero; returning a degenerate interval.",
                RuntimeWarning,
                stacklevel=2,
            )
            t_stat = 0.0 if abs(tau_hat) <= self._DEGENERATE_SIGMA_TOL else float(np.sign(tau_hat) * np.inf)
            p_value = 1.0 if abs(tau_hat) <= self._DEGENERATE_SIGMA_TOL else 0.0
            ci_lower = tau_hat
            ci_upper = tau_hat
        else:
            t_stat = float(np.sqrt(float(k_used)) * tau_hat / sigma_hat_tau)
            p_value = float(2.0 * student_t.sf(abs(t_stat), df=df))
            ci_lower = float(tau_hat - crit * se_hat)
            ci_upper = float(tau_hat + crit * se_hat)

        fold_estimates = pd.Series(
            {f"fold_{k + 1}": float(val) for k, val in enumerate(tau_hat_k)},
            dtype=float,
            name="average_att_fold_estimates",
        )

        return _AverageATTTTestResult(
            available=True,
            message=None,
            att=tau_hat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            t_stat=t_stat,
            standard_error=se_hat,
            sigma_hat=sigma_hat_tau,
            fold_estimates=fold_estimates,
            fold_blocks=fold_blocks,
            n_folds=k_used,
            block_length=r,
        )

    # ---------------------------------------------------------------------
    # Conformal inference
    # ---------------------------------------------------------------------

    @staticmethod
    def _build_effect_only_pointwise_result(
        *,
        effect_by_time: pd.Series,
        post_times: list[Any],
    ) -> _PointwiseConformalResult:
        """Build point-estimate-only post-period outputs without conformal inversion.

        Parameters
        ----------
        effect_by_time : pandas.Series
            Point estimates by post-treatment period.
        post_times : list[Any]
            Ordered post-treatment period labels.

        Returns
        -------
        _PointwiseConformalResult
            Result container with effect path populated and inference fields set
            to neutral placeholders.
        """
        post_index = pd.Index(post_times)
        return _PointwiseConformalResult(
            effect_by_time=effect_by_time.reindex(post_index).copy(),
            ci_lower_by_time=pd.Series(
                np.nan,
                index=post_index,
                dtype=float,
                name="ci_lower_by_time",
            ),
            ci_upper_by_time=pd.Series(
                np.nan,
                index=post_index,
                dtype=float,
                name="ci_upper_by_time",
            ),
            p_value_by_time=pd.Series(
                1.0,
                index=post_index,
                dtype=float,
                name="p_value_by_time",
            ),
            is_significant_by_time=pd.Series(
                False,
                index=post_index,
                dtype=bool,
                name="is_significant_by_time",
            ),
            confidence_set_by_time={time_key: [] for time_key in post_times},
            grid_by_time={time_key: [] for time_key in post_times},
            grid_p_values_by_time={time_key: [] for time_key in post_times},
        )

    @staticmethod
    def _cwz_overlapping_moving_block(n_total: int) -> list[np.ndarray]:
        """Return circular-shift permutations used by CWZ p-values.

        Parameters
        ----------
        n_total : int
            Number of residual periods.

        Returns
        -------
        list[numpy.ndarray]
            Circular-shift index arrays.
        """
        return circular_shift_indices(n_total)

    @staticmethod
    def _conformal_stat_from_residuals(
        residuals: np.ndarray,
        *,
        n_pre: int,
    ) -> float:
        """Compute CWZ residual aggregation statistic.

        Parameters
        ----------
        residuals : numpy.ndarray
            Residual vector with pre-period values followed by post-period
            values.
        n_pre : int
            Number of pre-period residuals.

        Returns
        -------
        float
            ``|sum(post_residuals)| / sqrt(T_post)``.

        Notes
        -----
        In this model's pointwise inversion, the reduced sample has a single
        post period, so the statistic collapses to ``|u_post|``.
        """
        return cwz_stat_from_residuals(residuals, n_pre=n_pre)

    def _cwz_overlapping_moving_block_p_value(
        self,
        *,
        residuals: np.ndarray,
        n_pre: int,
    ) -> float:
        """Compute CWZ circular-shift p-value for residual statistic.

        Parameters
        ----------
        residuals : numpy.ndarray
            Residual vector with pre and post entries.
        n_pre : int
            Number of pre-period residuals.

        Returns
        -------
        float
            One-sided permutation p-value clipped to ``[0, 1]``.
        """
        resid = np.asarray(residuals, dtype=float).reshape(-1)
        observed = self._conformal_stat_from_residuals(resid, n_pre=n_pre)

        perm_stats = np.empty(resid.size, dtype=float)
        for k, idx in enumerate(self._cwz_overlapping_moving_block(resid.size)):
            perm_stats[k] = self._conformal_stat_from_residuals(resid[idx], n_pre=n_pre)

        p_value = float(np.mean(perm_stats >= (observed - 1e-12)))
        return float(np.clip(p_value, 0.0, 1.0))

    @staticmethod
    def _accepted_segments(grid: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        """Map accepted grid points to contiguous parameter intervals.

        Parameters
        ----------
        grid : numpy.ndarray
            Parameter grid.
        mask : numpy.ndarray
            Acceptance mask over the grid.

        Returns
        -------
        list[tuple[float, float]]
            Contiguous accepted intervals.
        """
        return accepted_segments(grid, mask)

    def _build_conformal_grid(
        self,
        *,
        point_estimate: float,
        pre_residuals: np.ndarray,
    ) -> np.ndarray:
        """Build conformal inversion grid for one post-treatment period.

        Parameters
        ----------
        point_estimate : float
            Point estimate of period-specific treatment effect.
        pre_residuals : numpy.ndarray
            Pre-treatment residuals used to scale automatic bounds.

        Returns
        -------
        numpy.ndarray
            One-dimensional candidate grid for null effect inversion.
        """
        if self.conformal_grid_min is not None and self.conformal_grid_max is not None:
            return np.linspace(
                self.conformal_grid_min,
                self.conformal_grid_max,
                self.conformal_grid_size,
                dtype=float,
            )

        pre_resid = np.asarray(pre_residuals, dtype=float).reshape(-1)
        # Scale is stabilized by both RMSE and sample standard deviation.
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
        """Refit null-constrained residuals for one post-treatment period.

        Parameters
        ----------
        y1_observed : numpy.ndarray
            Treated outcomes over all periods.
        y0_all : numpy.ndarray
            Donor outcomes matrix over all periods.
        n_pre : int
            Number of pre-treatment periods.
        target_post_offset : int
            Offset of target post period relative to post-period start.
        theta_t_value : float
            Candidate null effect for the target post period.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Residuals and fitted augmented weights under the null.

        Notes
        -----
        Uses reduced sample ``[all pre] + [target post]`` and imposes the null
        only on the target post outcome.
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
        """Invert pointwise conformal test for one post-treatment period.

        Parameters
        ----------
        y1_observed : numpy.ndarray
            Treated outcomes over all periods.
        y0_all : numpy.ndarray
            Donor outcomes matrix over all periods.
        n_pre : int
            Number of pre-treatment periods.
        target_post_offset : int
            Offset of target post period in post-period index.
        point_estimate_t : float
            Point estimate for target period effect.
        pre_residuals : numpy.ndarray
            Pre-treatment residuals for grid scaling.

        Returns
        -------
        tuple
            ``(ci_lower, ci_upper, p_zero, segments, grid, p_values)``.
        """
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
            p_values[i] = self._cwz_overlapping_moving_block_p_value(
                residuals=residuals_theta,
                n_pre=n_pre,
            )

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
        p_value_at_zero = self._cwz_overlapping_moving_block_p_value(
            residuals=zero_residuals,
            n_pre=n_pre,
        )

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
        """Run pointwise conformal inversion for all post-treatment periods.

        Parameters
        ----------
        y1_observed : numpy.ndarray
            Treated outcomes over all periods.
        y0_all : numpy.ndarray
            Donor outcomes matrix over all periods.
        n_pre : int
            Number of pre-treatment periods.
        post_times : list[Any]
            Post-treatment period labels.
        effect_by_time : pandas.Series
            Point estimates by post-treatment period.
        pre_residuals : numpy.ndarray
            Pre-treatment residuals used in grid construction.

        Returns
        -------
        _PointwiseConformalResult
            Pointwise confidence intervals, p-values, and accepted sets.
        """
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

    def _compute_pointwise_conformal_if_requested(
        self,
        *,
        y1_observed: np.ndarray,
        y0_all: np.ndarray,
        n_pre: int,
        post_times: list[Any],
        effect_by_time: pd.Series,
        pre_residuals: np.ndarray,
    ) -> _PointwiseConformalResult:
        """Run optional pointwise conformal inference with safe fallback.

        Parameters
        ----------
        y1_observed : numpy.ndarray
            Treated outcomes over all periods.
        y0_all : numpy.ndarray
            Donor outcomes matrix over all periods.
        n_pre : int
            Number of pre-treatment periods.
        post_times : list[Any]
            Post-treatment period labels.
        effect_by_time : pandas.Series
            Point estimates by post-treatment period.
        pre_residuals : numpy.ndarray
            Pre-treatment residuals used in grid construction.

        Returns
        -------
        _PointwiseConformalResult
            Conformal outputs if requested and successful; otherwise point-only
            placeholders.
        """
        if not self.compute_pointwise_conformal:
            return self._build_effect_only_pointwise_result(
                effect_by_time=effect_by_time,
                post_times=post_times,
            )

        try:
            return self._invert_pointwise_conformal_tests(
                y1_observed=y1_observed,
                y0_all=y0_all,
                n_pre=n_pre,
                post_times=post_times,
                effect_by_time=effect_by_time,
                pre_residuals=pre_residuals,
            )
        except Exception as exc:
            warnings.warn(
                "Pointwise conformal inference was unavailable; returning effect path without "
                f"pointwise CI/p-values. Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._build_effect_only_pointwise_result(
                effect_by_time=effect_by_time,
                post_times=post_times,
            )

    def _compute_average_att_ttest_if_requested(
        self,
        *,
        y1_observed: np.ndarray,
        y0_all: np.ndarray,
        pre_times: list[Any],
        post_times: list[Any],
    ) -> _AverageATTTTestResult:
        """Run average ATT t-test if enabled, with safe error containment.

        Parameters
        ----------
        y1_observed : numpy.ndarray
            Treated outcomes over all periods.
        y0_all : numpy.ndarray
            Donor outcomes matrix over all periods.
        pre_times : list[Any]
            Pre-treatment period labels.
        post_times : list[Any]
            Post-treatment period labels.

        Returns
        -------
        _AverageATTTTestResult
            Available result or structured unavailable payload.
        """
        result = self._empty_average_att_ttest_result("Average ATT t-test not requested.")
        if not self.compute_average_att_ttest:
            return result

        try:
            result = self._compute_average_att_ttest(
                y1_observed=y1_observed,
                y0_all=y0_all,
                pre_times=pre_times,
                post_times=post_times,
            )
        except Exception as exc:
            result = self._empty_average_att_ttest_result(str(exc))

        if not result.available:
            warnings.warn(
                "Average ATT self-normalized t-test inference was unavailable; the dynamic effect "
                f"path was still computed. Reason: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    def _compute_augmented_weight_metrics(
        self,
        *,
        x0_pre: np.ndarray,
        w_aug: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute diagnostics for augmented weights.

        Parameters
        ----------
        x0_pre : numpy.ndarray
            Pre-period donor matrix.
        w_aug : numpy.ndarray
            Augmented donor weights.

        Returns
        -------
        tuple[float, float, float]
            ``(condition_number, l1_norm, max_abs_weight)``.
        """
        gram_aug = x0_pre.T @ x0_pre + self.lambda_aug * np.eye(x0_pre.shape[1], dtype=float)
        cond_gram_aug = float(np.linalg.cond(gram_aug))
        l1_w_aug = float(np.sum(np.abs(w_aug)))
        max_abs_w_aug = float(np.max(np.abs(w_aug)))
        return cond_gram_aug, l1_w_aug, max_abs_w_aug

    def _warn_on_augmented_weight_metrics(
        self,
        *,
        cond_gram_aug: float,
        l1_w_aug: float,
        max_abs_w_aug: float,
    ) -> None:
        """Emit warnings when augmented weights look numerically unstable.

        Parameters
        ----------
        cond_gram_aug : float
            Condition number of augmented normal-equation matrix.
        l1_w_aug : float
            L1 norm of augmented weights.
        max_abs_w_aug : float
            Maximum absolute augmented weight.
        """
        if cond_gram_aug > self._AUGMENTED_GRAM_COND_WARN_THRESHOLD:
            warnings.warn(
                f"Augmented normal equations are ill-conditioned (cond={cond_gram_aug:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        if (
            l1_w_aug > self._AUGMENTED_WEIGHT_L1_WARN_THRESHOLD
            or max_abs_w_aug > self._AUGMENTED_WEIGHT_MAX_ABS_WARN_THRESHOLD
        ):
            warnings.warn(
                "Augmented donor weights are extreme; estimates may be unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _build_diagnostics(
        self,
        *,
        pointwise: _PointwiseConformalResult,
        average_att_ttest: _AverageATTTTestResult,
        pre_residuals: np.ndarray,
        w_sc: np.ndarray,
        w_aug: np.ndarray,
        min_possible_p: float,
        cond_gram_aug: float,
        l1_w_aug: float,
        max_abs_w_aug: float,
        full_sample_post_mean_gap: float,
    ) -> dict[str, Any]:
        """Assemble public diagnostics payload returned in ``PanelEstimate``.

        Parameters
        ----------
        pointwise : _PointwiseConformalResult
            Pointwise path inference results.
        average_att_ttest : _AverageATTTTestResult
            Aggregate ATT inference results.
        pre_residuals : numpy.ndarray
            Pre-period residuals under fitted augmented weights.
        w_sc : numpy.ndarray
            Simplex donor weights.
        w_aug : numpy.ndarray
            Augmented donor weights.
        min_possible_p : float
            Minimum attainable pointwise p-value under circular-shift test.
        cond_gram_aug : float
            Condition number of augmented Gram matrix.
        l1_w_aug : float
            L1 norm of augmented weights.
        max_abs_w_aug : float
            Maximum absolute augmented weight.
        full_sample_post_mean_gap : float
            Mean post-treatment gap using full-sample augmented weights.

        Returns
        -------
        dict[str, Any]
            Diagnostics dictionary exposed through ``PanelEstimate``.
        """
        pointwise_conformal_available = bool(
            self.compute_pointwise_conformal
            and any(len(grid) > 0 for grid in pointwise.grid_by_time.values())
        )
        return {
            "n_donors": len(self._donors),
            "n_pre_periods": len(self._pre_times),
            "n_post_periods": len(self._post_times),
            "enforce_sum_to_one_augmented": bool(self.enforce_sum_to_one_augmented),
            "lambda_sc": float(self.lambda_sc),
            "lambda_sc_role": "numerical_regularizer",
            "w_sc_is_approximate_scm": bool(self.lambda_sc != 0.0),
            "w_sc_equals_exact_simplex_scm_only_if_lambda_sc_zero": True,
            "pre_rmse_augmented": self._rmse(pre_residuals),
            "sum_weights_augmented": float(np.sum(w_aug)),
            "min_weight_augmented": float(np.min(w_aug)),
            "max_weight_augmented": float(np.max(w_aug)),
            "l1_norm_weights_augmented": l1_w_aug,
            "max_abs_weight_augmented": max_abs_w_aug,
            "sum_weights_sc": float(np.sum(w_sc)),
            "min_weight_sc": float(np.min(w_sc)),
            "max_weight_sc": float(np.max(w_sc)),
            "cond_augmented_gram": cond_gram_aug,
            "estimand": "dynamic_effect_path",
            "ci_alpha": float(self.alpha),
            "inference_default": "average_att_ttest",
            "pointwise_conformal_requested": bool(self.compute_pointwise_conformal),
            "pointwise_conformal_available": pointwise_conformal_available,
            "pointwise_ci_method": (
                "cwz_overlapping_moving_block"
                if pointwise_conformal_available
                else "average_att_ttest"
            ),
            "pointwise_min_possible_p_value": (
                min_possible_p if self.compute_pointwise_conformal else np.nan
            ),
            "pointwise_rejection_possible_at_alpha": (
                bool(self.alpha >= min_possible_p) if self.compute_pointwise_conformal else None
            ),
            "p_value_by_time_is_pointwise_not_joint": (
                True if pointwise_conformal_available else None
            ),
            "multiple_testing_adjusted": False,
            "pointwise_confidence_set_representation": (
                "grid_approximated_contiguous_segments"
                if pointwise_conformal_available
                else None
            ),
            "pointwise_grid_by_time": pointwise.grid_by_time,
            "pointwise_grid_p_values_by_time": pointwise.grid_p_values_by_time,
            "pointwise_confidence_set_by_time": self._confidence_set_by_time,
            "pointwise_accepted_sets_by_time": self._confidence_set_by_time,
            "effect_by_time": [
                {
                    "period": time_key,
                    "estimate": float(pointwise.effect_by_time.loc[time_key]),
                }
                for time_key in self._post_times
            ],
            "average_att_ttest_requested": bool(self.compute_average_att_ttest),
            "average_att_ttest_available": bool(average_att_ttest.available),
            "average_att_ttest_method": "cwz_2018_debiased_self_normalized_t",
            "average_att_ttest_message": average_att_ttest.message,
            "average_att_n_folds_requested": int(self.average_att_n_folds),
            "average_att_n_folds_used": int(average_att_ttest.n_folds),
            "average_att_block_length": int(average_att_ttest.block_length),
            "average_att_estimate": (
                np.nan if average_att_ttest.att is None else float(average_att_ttest.att)
            ),
            "average_att_ci_lower": (
                np.nan if average_att_ttest.ci_lower is None else float(average_att_ttest.ci_lower)
            ),
            "average_att_ci_upper": (
                np.nan if average_att_ttest.ci_upper is None else float(average_att_ttest.ci_upper)
            ),
            "average_att_p_value": (
                np.nan if average_att_ttest.p_value is None else float(average_att_ttest.p_value)
            ),
            "average_att_t_stat": (
                np.nan if average_att_ttest.t_stat is None else float(average_att_ttest.t_stat)
            ),
            "average_att_standard_error": (
                np.nan
                if average_att_ttest.standard_error is None
                else float(average_att_ttest.standard_error)
            ),
            "average_att_sigma_hat": (
                np.nan if average_att_ttest.sigma_hat is None else float(average_att_ttest.sigma_hat)
            ),
            "average_att_fold_estimates": average_att_ttest.fold_estimates.to_dict(),
            "average_att_fold_blocks": dict(average_att_ttest.fold_blocks),
            "average_att_full_sample_post_mean_gap": full_sample_post_mean_gap,
            "average_att_crossfit_minus_full_sample_gap": (
                np.nan
                if average_att_ttest.att is None
                else float(average_att_ttest.att - full_sample_post_mean_gap)
            ),
        }

    def _assign_fitted_state(
        self,
        *,
        data: PanelDataSCM,
        donors: list[Hashable],
        pre_times: list[Any],
        post_times: list[Any],
        all_times: list[Any],
        w_aug: np.ndarray,
        observed_series: pd.Series,
        synthetic_series: pd.Series,
        gap_series: pd.Series,
        pointwise: _PointwiseConformalResult,
        average_att_ttest: _AverageATTTTestResult,
    ) -> None:
        """Persist fitted model state before diagnostics export.

        Parameters
        ----------
        data : PanelDataSCM
            Contract object used for fitting.
        donors : list[Hashable]
            Donor unit identifiers.
        pre_times : list[Any]
            Pre-treatment period labels.
        post_times : list[Any]
            Post-treatment period labels.
        all_times : list[Any]
            Combined analysis period labels.
        w_aug : numpy.ndarray
            Fitted augmented donor weights.
        observed_series : pandas.Series
            Observed treated outcomes over all periods.
        synthetic_series : pandas.Series
            Fitted synthetic outcomes over all periods.
        gap_series : pandas.Series
            Observed minus synthetic over all periods.
        pointwise : _PointwiseConformalResult
            Pointwise inference result container.
        average_att_ttest : _AverageATTTTestResult
            Average ATT inference result container.
        """
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

        self._average_att_ttest = average_att_ttest

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit(self, data: PanelDataSCM) -> "AugmentedSyntheticControl":
        """Fit ASCM and compute inference outputs.

        Parameters
        ----------
        data : PanelDataSCM
            Validated synthetic-control panel data.

        Returns
        -------
        AugmentedSyntheticControl
            Fitted estimator instance.

        Raises
        ------
        ValueError
            If input type is invalid or panel requirements are violated.
        """
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

        w_sc, w_aug = self._fit_augmented_weights(x=x0_pre, y=y1_pre)

        y0_hat = y0_all @ w_aug
        gap = y1_all - y0_hat
        n_pre = len(pre_times)
        min_possible_p = 1.0 / float(n_pre + 1)

        # Pre residuals are reused in conformal-grid scale heuristics.
        pre_residuals = y1_pre - (x0_pre @ w_aug)

        if self.compute_pointwise_conformal:
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

        # Stage 1: default inference, aggregate post-window ATT t-test.
        average_att_ttest = self._compute_average_att_ttest_if_requested(
            y1_observed=y1_all,
            y0_all=y0_all,
            pre_times=pre_times,
            post_times=post_times,
        )

        # Stage 2: optional dynamic pointwise conformal inference.
        pointwise = self._compute_pointwise_conformal_if_requested(
            y1_observed=y1_all,
            y0_all=y0_all,
            n_pre=n_pre,
            post_times=post_times,
            effect_by_time=effect_by_time,
            pre_residuals=pre_residuals,
        )

        # Stage 3: stability checks on final augmented fit.
        cond_gram_aug, l1_w_aug, max_abs_w_aug = self._compute_augmented_weight_metrics(
            x0_pre=x0_pre,
            w_aug=w_aug,
        )
        self._warn_on_augmented_weight_metrics(
            cond_gram_aug=cond_gram_aug,
            l1_w_aug=l1_w_aug,
            max_abs_w_aug=max_abs_w_aug,
        )

        # Stage 4: persist fitted state and publish diagnostics.
        self._assign_fitted_state(
            data=data,
            donors=donors,
            pre_times=pre_times,
            post_times=post_times,
            all_times=all_times,
            w_aug=w_aug,
            observed_series=observed_series,
            synthetic_series=synthetic_series,
            gap_series=gap_series,
            pointwise=pointwise,
            average_att_ttest=average_att_ttest,
        )

        full_sample_post_mean_gap = float(np.mean(effect_by_time.to_numpy(dtype=float)))
        self._diagnostics = self._build_diagnostics(
            pointwise=pointwise,
            average_att_ttest=average_att_ttest,
            pre_residuals=pre_residuals,
            w_sc=w_sc,
            w_aug=w_aug,
            min_possible_p=min_possible_p,
            cond_gram_aug=cond_gram_aug,
            l1_w_aug=l1_w_aug,
            max_abs_w_aug=max_abs_w_aug,
            full_sample_post_mean_gap=full_sample_post_mean_gap,
        )

        self._is_fitted = True
        return self

    def estimate(self) -> PanelEstimate:
        """Return dynamic-path estimate object.

        Returns
        -------
        PanelEstimate
            Dynamic path estimates with pointwise inference fields. Aggregate
            average ATT t-test outputs are provided in ``diagnostics``.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
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
