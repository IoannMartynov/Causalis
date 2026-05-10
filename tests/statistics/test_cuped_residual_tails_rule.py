from __future__ import annotations

from causalis.scenarios.cuped.refutation.regression_checks import RegressionChecks
from causalis.scenarios.cuped.refutation import (
    assumption_cooks,
    assumption_leverage,
    assumption_vif,
    FLAG_GREEN,
    FLAG_RED,
    FLAG_YELLOW,
    assumption_residual_tails,
)


def _checks(
    max_abs_std_resid: float,
    max_leverage: float = 0.1,
    leverage_cutoff: float = 0.2,
    n_high_leverage: int = 0,
    max_cooks: float = 0.05,
    n_high_cooks: int = 0,
    cooks_cutoff: float = 0.1,
    p_main_covariates: int = 1,
    vif: dict[str, float] | None = None,
    near_duplicate_pairs: list[tuple[str, str, float]] | None = None,
) -> RegressionChecks:
    return RegressionChecks(
        ate_naive=1.0,
        ate_adj=1.0,
        ate_gap=0.0,
        ate_gap_over_se_naive=0.0,
        k=3,
        rank=3,
        full_rank=True,
        condition_number=10.0,
        p_main_covariates=p_main_covariates,
        near_duplicate_pairs=near_duplicate_pairs or [],
        vif=vif,
        resid_scale_mad=1.0,
        n_std_resid_gt_3=270,
        n_std_resid_gt_4=6,
        max_abs_std_resid=max_abs_std_resid,
        max_leverage=max_leverage,
        leverage_cutoff=leverage_cutoff,
        n_high_leverage=n_high_leverage,
        max_cooks=max_cooks,
        cooks_cutoff=cooks_cutoff,
        n_high_cooks=n_high_cooks,
        min_one_minus_h=0.8,
        n_tiny_one_minus_h=0,
        winsor_q=0.01,
        ate_adj_winsor=1.0,
        ate_adj_winsor_gap=0.0,
    )


def test_assumption_vif_unavailable_is_green_when_not_applicable():
    row = assumption_vif(_checks(max_abs_std_resid=5.0, p_main_covariates=1, vif=None))
    assert row["flag"] == FLAG_GREEN


def test_assumption_vif_unavailable_is_yellow_with_multiple_covariates():
    row = assumption_vif(_checks(max_abs_std_resid=5.0, p_main_covariates=2, vif=None))
    assert row["flag"] == FLAG_YELLOW


def test_assumption_vif_unavailable_is_yellow_with_near_duplicates():
    row = assumption_vif(
        _checks(
            max_abs_std_resid=5.0,
            p_main_covariates=1,
            vif=None,
            near_duplicate_pairs=[("x1__centered", "x2__centered", 0.999999999)],
        )
    )
    assert row["flag"] == FLAG_YELLOW


def test_assumption_residual_tails_is_green_when_max_is_small():
    row = assumption_residual_tails(_checks(max_abs_std_resid=5.0))
    assert row["flag"] == FLAG_GREEN
    assert row["test"] == "Residual extremes"


def test_assumption_residual_tails_is_yellow_when_max_exceeds_yellow_cutoff():
    row = assumption_residual_tails(_checks(max_abs_std_resid=8.0))
    assert row["flag"] == FLAG_YELLOW


def test_assumption_residual_tails_is_red_when_max_exceeds_red_cutoff():
    row = assumption_residual_tails(_checks(max_abs_std_resid=11.0))
    assert row["flag"] == FLAG_RED


def test_assumption_cooks_is_green_when_only_count_cutoff_is_triggered():
    row = assumption_cooks(
        _checks(
            max_abs_std_resid=5.0,
            max_cooks=0.05,
            n_high_cooks=25,
            cooks_cutoff=4.0 / 100000.0,
        )
    )
    assert row["flag"] == FLAG_GREEN


def test_assumption_cooks_is_yellow_when_max_exceeds_yellow_cutoff():
    row = assumption_cooks(_checks(max_abs_std_resid=5.0, max_cooks=0.2, n_high_cooks=0))
    assert row["flag"] == FLAG_YELLOW


def test_assumption_cooks_is_red_when_max_exceeds_red_cutoff():
    row = assumption_cooks(_checks(max_abs_std_resid=5.0, max_cooks=1.2, n_high_cooks=0))
    assert row["flag"] == FLAG_RED


def test_assumption_leverage_is_green_when_max_not_large_relative_to_cutoff():
    row = assumption_leverage(
        _checks(
            max_abs_std_resid=5.0,
            max_leverage=0.024,
            leverage_cutoff=0.005,
            n_high_leverage=15,
        )
    )
    assert row["flag"] == FLAG_GREEN


def test_assumption_leverage_is_yellow_when_max_exceeds_5x_cutoff():
    row = assumption_leverage(
        _checks(
            max_abs_std_resid=5.0,
            max_leverage=0.03,
            leverage_cutoff=0.005,
            n_high_leverage=1,
        )
    )
    assert row["flag"] == FLAG_YELLOW


def test_assumption_leverage_is_red_when_max_exceeds_red_rule():
    row = assumption_leverage(
        _checks(
            max_abs_std_resid=5.0,
            max_leverage=0.6,
            leverage_cutoff=0.005,
            n_high_leverage=1,
        )
    )
    assert row["flag"] == FLAG_RED
