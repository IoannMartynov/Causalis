from datetime import datetime, timezone

import numpy as np
import pandas as pd

from causalis.data_contracts import PanelEstimate


def _build_panel_estimate(
    *, disconnected_second_ci: bool = False, include_average_atte: bool = False
) -> PanelEstimate:
    pre = ["2020-01", "2020-02"]
    post = ["2020-03", "2020-04"]
    all_times = pre + post

    ci_low = [0.5, 1.1]
    ci_high = [1.5, 2.9]
    confidence_set = {
        "2020-03": [(0.5, 1.5)],
        "2020-04": [(1.1, 2.9)],
    }
    if disconnected_second_ci:
        ci_low[1] = np.nan
        ci_high[1] = np.nan
        confidence_set["2020-04"] = [(0.9, 1.4), (2.0, 2.8)]

    diagnostics = {
        "pointwise_ci_method": "block_bootstrap",
        "n_donors": 2,
        "n_pre_periods": 2,
    }
    if include_average_atte:
        diagnostics.update(
            {
                "average_att_ttest_available": True,
                "average_att_estimate": 1.7,
                "average_att_ci_lower": 1.2,
                "average_att_ci_upper": 2.2,
                "average_att_p_value": 0.01,
                "average_att_n_folds_used": 3,
                "average_att_fold_estimates": {"fold_1": 1.6, "fold_2": 1.8, "fold_3": 1.7},
            }
        )

    return PanelEstimate(
        model="AugmentedSyntheticControl",
        treated_unit="T",
        treatment_start="2020-03",
        pre_times=pre,
        post_times=post,
        effect_by_time=pd.Series([1.0, 2.0], index=post),
        ci_lower_by_time=pd.Series(ci_low, index=post),
        ci_upper_by_time=pd.Series(ci_high, index=post),
        p_value_by_time=pd.Series([0.04, 0.20], index=post),
        is_significant_by_time=pd.Series([True, False], index=post),
        confidence_set_by_time=confidence_set,
        alpha=0.05,
        observed_outcome=pd.Series([10.0, 11.0, 13.0, 14.0], index=all_times),
        synthetic_outcome=pd.Series([9.0, 10.0, 12.0, 12.0], index=all_times),
        donor_weights_augmented={"C1": 0.8, "C2": 0.2},
        diagnostics=diagnostics,
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )


def test_panel_estimate_summary_is_compact_dataframe():
    est = _build_panel_estimate()

    summary = est.summary()

    assert isinstance(summary, pd.DataFrame)
    assert summary.loc["estimand", "value"] == "dynamic_effect_path"
    assert summary.loc["value", "value"] == "1.5000 (post_period_average)"
    assert summary.loc["value_relative", "value"] == "12.5000"
    assert summary.loc["p_value", "value"] is None
    assert bool(summary.loc["is_significant", "value"]) is True
    assert int(summary.loc["n_donors", "value"]) == 2
    assert int(summary.loc["n_pre_periods", "value"]) == 2
    assert summary.loc["post_outcome_d_mean", "value"] == "13.5000"
    assert int(summary.loc["n_post_periods", "value"]) == 2
    assert int(summary.loc["n_significant_periods", "value"]) == 1
    assert summary.loc["inference", "value"] == "block_bootstrap"
    assert summary.loc["time", "value"] == "2026-03-01"
    assert summary.loc["cumulative_effect", "value"] == "3.0000"


def test_panel_estimate_summary_poinwise_returns_expected_dataframe():
    est = _build_panel_estimate(disconnected_second_ci=True)

    details = est.summary_poinwise()

    assert isinstance(details, pd.DataFrame)
    assert list(details.columns) == [
        "time",
        "effect",
        "ci_lower",
        "ci_upper",
        "p_value",
        "is_significant",
    ]
    assert details.shape == (2, 6)
    assert details.iloc[0]["time"] == "2020-03"
    assert details.iloc[0]["effect"] == 1.0
    assert details.iloc[0]["ci_lower"] == 0.5
    assert pd.isna(details.iloc[1]["ci_lower"])
    assert bool(details.iloc[0]["is_significant"]) is True
    assert bool(details.iloc[1]["is_significant"]) is False


def test_panel_estimate_summary_uses_average_atte_when_available():
    est = _build_panel_estimate(include_average_atte=True)

    summary = est.summary()

    assert summary.loc["estimand", "value"] == "average_post_effect"
    assert summary.loc["value", "value"] == "1.7000 (ci_abs: 1.2000, 2.2000)"
    assert summary.loc["value_relative", "value"] == "14.1667 (ci_rel: 10.0000, 18.3333)"
    assert summary.loc["p_value", "value"] == "0.0100"
    assert bool(summary.loc["is_significant", "value"]) is True
    assert summary.loc["pointwise_post_period_average", "value"] == "1.5000"
    assert summary.loc["cumulative_effect", "value"] == "3.0000"
