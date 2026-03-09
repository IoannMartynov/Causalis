import numpy as np

from causalis.scenarios.synthetic_control._utils import (
    accepted_segments,
    build_average_att_blocks,
    cwz_stat_from_residuals,
    project_to_simplex,
)


def test_project_to_simplex_returns_valid_weights():
    vec = np.array([0.5, -0.2, 2.0, 0.1], dtype=float)
    projected = project_to_simplex(vec)

    assert projected.shape == vec.shape
    assert np.all(projected >= 0.0)
    assert np.isclose(float(np.sum(projected)), 1.0)


def test_build_average_att_blocks_contiguous_and_bounded():
    blocks, k_used, block_length = build_average_att_blocks(
        n_pre=10,
        n_post=3,
        n_folds=4,
    )

    assert k_used == 4
    assert block_length == 2
    assert len(blocks) == 4
    assert np.array_equal(blocks[0], np.array([0, 1], dtype=int))
    assert np.array_equal(blocks[3], np.array([6, 7], dtype=int))


def test_cwz_stat_from_residuals_uses_post_window_only():
    residuals = np.array([1.0, -1.0, 2.0, 4.0], dtype=float)
    stat = cwz_stat_from_residuals(residuals, n_pre=2)
    expected = abs(2.0 + 4.0) / np.sqrt(2.0)
    assert np.isclose(stat, expected)


def test_accepted_segments_returns_contiguous_ranges():
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    mask = np.array([False, True, True, False, True], dtype=bool)
    segments = accepted_segments(grid, mask)
    assert segments == [(-1.0, 0.0), (2.0, 2.0)]
