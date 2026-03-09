from __future__ import annotations

import numpy as np


def rmse(values: np.ndarray) -> float:
    """Compute root mean squared error.

    Parameters
    ----------
    values : numpy.ndarray
        Input array-like values. The input is flattened before computation.

    Returns
    -------
    float
        Root mean squared error. Returns ``0.0`` for empty input.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex.

    The simplex is defined as non-negative vectors whose entries sum to one.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional coefficient vector.

    Returns
    -------
    numpy.ndarray
        Projected vector with non-negative entries summing to one.

    Raises
    ------
    ValueError
        If the input is empty or contains non-finite values.
    """
    vec = np.asarray(values, dtype=float).reshape(-1)
    if vec.size < 1:
        raise ValueError("Simplex projection requires at least one coefficient.")
    if not np.isfinite(vec).all():
        raise ValueError("Simplex projection input must be finite.")

    # Wang & Carreira-Perpiñán sorting-based Euclidean projection.
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


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a linear system with least-squares fallback.

    Parameters
    ----------
    a : numpy.ndarray
        Coefficient matrix.
    b : numpy.ndarray
        Right-hand side vector or matrix.

    Returns
    -------
    numpy.ndarray
        Exact solution when ``a`` is non-singular, otherwise least-squares
        solution.
    """
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def circular_shift_indices(n_total: int) -> list[np.ndarray]:
    """Generate circular-shift index permutations.

    Parameters
    ----------
    n_total : int
        Sequence length.

    Returns
    -------
    list[numpy.ndarray]
        All ``n_total`` circular shifts of ``arange(n_total)``.
    """
    base = np.arange(n_total, dtype=int)
    return [np.roll(base, -shift) for shift in range(n_total)]


def cwz_stat_from_residuals(residuals: np.ndarray, *, n_pre: int) -> float:
    """Compute CWZ post-window residual aggregation statistic.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residual vector containing pre-period entries first, then post-period
        entries.
    n_pre : int
        Number of pre-period residuals.

    Returns
    -------
    float
        ``|sum(post_residuals)| / sqrt(n_post)``.

    Raises
    ------
    ValueError
        If ``n_pre`` does not satisfy ``1 <= n_pre < len(residuals)``.
    """
    resid = np.asarray(residuals, dtype=float).reshape(-1)
    n_total = int(resid.size)
    if n_pre < 1 or n_pre >= n_total:
        raise ValueError("n_pre must satisfy 1 <= n_pre < len(residuals).")
    post = resid[n_pre:]
    n_post = int(post.size)
    return float(abs(np.sum(post)) / np.sqrt(float(n_post)))


def accepted_segments(grid: np.ndarray, accepted_mask: np.ndarray) -> list[tuple[float, float]]:
    """Convert a boolean acceptance mask into contiguous grid segments.

    Parameters
    ----------
    grid : numpy.ndarray
        One-dimensional, ordered grid of candidate parameter values.
    accepted_mask : numpy.ndarray
        Boolean mask with the same length as ``grid``.

    Returns
    -------
    list[tuple[float, float]]
        Closed intervals corresponding to contiguous accepted regions.

    Raises
    ------
    ValueError
        If ``grid`` and ``accepted_mask`` are not one-dimensional arrays of the
        same length.
    """
    grid_arr = np.asarray(grid, dtype=float)
    mask_arr = np.asarray(accepted_mask, dtype=bool)
    if grid_arr.ndim != 1 or mask_arr.ndim != 1 or grid_arr.size != mask_arr.size:
        raise ValueError("grid and mask must be 1D with the same length.")
    if grid_arr.size == 0:
        return []

    segments: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, is_accepted in enumerate(mask_arr):
        if is_accepted and start_idx is None:
            start_idx = i
        elif (not is_accepted) and start_idx is not None:
            segments.append((float(grid_arr[start_idx]), float(grid_arr[i - 1])))
            start_idx = None
    if start_idx is not None:
        segments.append((float(grid_arr[start_idx]), float(grid_arr[-1])))
    return segments


def build_average_att_blocks(
    *,
    n_pre: int,
    n_post: int,
    n_folds: int,
) -> tuple[list[np.ndarray], int, int]:
    """Build consecutive pre-period holdout blocks for average ATT inference.

    Parameters
    ----------
    n_pre : int
        Number of pre-treatment periods ``T0``.
    n_post : int
        Number of post-treatment periods ``T1``.
    n_folds : int
        Requested number of folds ``K``.

    Returns
    -------
    tuple[list[numpy.ndarray], int, int]
        ``(blocks, k_used, block_length)`` where ``blocks`` are consecutive
        holdout indices over the pre-period.

    Notes
    -----
    If ``n_pre`` is not divisible by ``k_used``, the rule
    ``r = min(floor(T0 / K), T1)`` is applied on the first ``K * r`` periods.
    Remaining pre-periods are included in every fold's training subset.

    Raises
    ------
    ValueError
        If the configuration cannot produce at least two folds and a positive
        holdout block length.
    """
    if n_pre < 2:
        raise ValueError("Average ATT t-test requires at least two pre-treatment periods.")
    if n_post < 1:
        raise ValueError("Average ATT t-test requires at least one post-treatment period.")

    k_used = int(min(n_folds, n_pre))
    if k_used < 2:
        raise ValueError("Average ATT t-test requires at least two folds.")

    block_length = int(min(n_pre // k_used, n_post))
    if block_length < 1:
        raise ValueError(
            "Average ATT t-test requires block length r >= 1; increase data or reduce average_att_n_folds."
        )

    # Holdout blocks are contiguous to mirror the time-series split in the paper.
    blocks = [np.arange(k * block_length, (k + 1) * block_length, dtype=int) for k in range(k_used)]
    return blocks, k_used, block_length
