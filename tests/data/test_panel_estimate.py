from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelEstimate


def _base_estimate_kwargs() -> dict:
    pre = ["2020-01", "2020-02", "2020-03"]
    post = ["2020-04", "2020-05"]
    all_times = pre + post
    return {
        "model": "AugmentedSyntheticControl",
        "treated_unit": "T",
        "treatment_start": "2020-04",
        "pre_times": pre,
        "post_times": post,
        "effect_by_time": pd.Series([1.2, 1.3], index=post),
        "ci_lower_by_time": pd.Series([0.7, 0.8], index=post),
        "ci_upper_by_time": pd.Series([1.8, 1.9], index=post),
        "p_value_by_time": pd.Series([0.0033, 0.2], index=post),
        "is_significant_by_time": pd.Series([True, False], index=post),
        "confidence_set_by_time": {
            "2020-04": [(0.7, 1.8)],
            "2020-05": [(0.8, 1.9)],
        },
        "alpha": 0.05,
        "observed_outcome": pd.Series([10, 11, 12, 13, 14], index=all_times),
        "synthetic_outcome": pd.Series([9.9, 10.8, 11.7, 11.8, 12.6], index=all_times),
        "donor_weights_augmented": {"C1": 1.1, "C2": -0.1},
        "diagnostics": {"enforce_sum_to_one_augmented": True},
    }


def test_panel_estimate_valid_contract_and_summary():
    est = PanelEstimate(**_base_estimate_kwargs())
    summary = est.summary()

    assert est.estimand == "dynamic_effect_path"
    assert isinstance(est.created_at, datetime)
    assert est.created_at.tzinfo is not None
    assert str(est.treatment_start) == "2020-04"
    assert est.alpha == 0.05
    assert summary.loc["estimand", "value"] == "dynamic_effect_path"
    assert summary.loc["value", "value"] == "1.2500 (post_period_average)"
    assert summary.loc["value_relative", "value"] == "10.2459"
    assert summary.loc["alpha", "value"] == "0.0500"
    assert summary.loc["p_value", "value"] is None
    assert bool(summary.loc["is_significant", "value"]) is True


def test_effect_by_time_index_must_match_post_times():
    kwargs = _base_estimate_kwargs()
    kwargs["effect_by_time"] = pd.Series([1.2, 1.3], index=["2020-05", "2020-04"])

    with pytest.raises(ValueError, match="effect_by_time index must exactly equal post_times"):
        PanelEstimate(**kwargs)


def test_outcome_path_index_must_match_pre_plus_post():
    kwargs = _base_estimate_kwargs()
    kwargs["observed_outcome"] = pd.Series(
        [10, 11, 12, 13, 14], index=["2020-01", "2020-02", "2020-03", "2020-05", "2020-04"]
    )

    with pytest.raises(ValueError, match="observed_outcome index must exactly equal"):
        PanelEstimate(**kwargs)


def test_pre_post_must_be_disjoint_and_ordered_and_sorted():
    kwargs_overlap = _base_estimate_kwargs()
    kwargs_overlap["post_times"] = ["2020-03", "2020-04"]
    kwargs_overlap["effect_by_time"] = pd.Series([1.2, 1.3], index=["2020-03", "2020-04"])
    kwargs_overlap["ci_lower_by_time"] = pd.Series([0.7, 0.8], index=["2020-03", "2020-04"])
    kwargs_overlap["ci_upper_by_time"] = pd.Series([1.8, 1.9], index=["2020-03", "2020-04"])
    kwargs_overlap["p_value_by_time"] = pd.Series([0.1, 0.2], index=["2020-03", "2020-04"])
    kwargs_overlap["is_significant_by_time"] = pd.Series([False, False], index=["2020-03", "2020-04"])
    kwargs_overlap["confidence_set_by_time"] = {
        "2020-03": [(0.7, 1.8)],
        "2020-04": [(0.8, 1.9)],
    }
    kwargs_overlap["observed_outcome"] = pd.Series(
        [10, 11, 12, 13, 14], index=["2020-01", "2020-02", "2020-03", "2020-03", "2020-04"]
    )
    kwargs_overlap["synthetic_outcome"] = pd.Series(
        [9, 10, 11, 12, 13], index=["2020-01", "2020-02", "2020-03", "2020-03", "2020-04"]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        PanelEstimate(**kwargs_overlap)

    kwargs_unsorted = _base_estimate_kwargs()
    kwargs_unsorted["pre_times"] = ["2020-02", "2020-01", "2020-03"]
    kwargs_unsorted["observed_outcome"] = pd.Series(
        [10, 11, 12, 13, 14], index=["2020-02", "2020-01", "2020-03", "2020-04", "2020-05"]
    )
    kwargs_unsorted["synthetic_outcome"] = pd.Series(
        [9, 10, 11, 12, 13], index=["2020-02", "2020-01", "2020-03", "2020-04", "2020-05"]
    )

    with pytest.raises(ValueError, match="must be sorted ascending"):
        PanelEstimate(**kwargs_unsorted)


def test_numeric_finite_checks():
    kwargs_effect = _base_estimate_kwargs()
    kwargs_effect["effect_by_time"] = pd.Series([1.2, float("inf")], index=["2020-04", "2020-05"])
    with pytest.raises(ValueError, match="effect_by_time must contain only finite numeric values"):
        PanelEstimate(**kwargs_effect)

    kwargs_series = _base_estimate_kwargs()
    kwargs_series["p_value_by_time"] = pd.Series([0.2, "bad"], index=["2020-04", "2020-05"])
    with pytest.raises(ValueError, match="p_value_by_time must contain only finite numeric values"):
        PanelEstimate(**kwargs_series)


def test_ci_bounds_must_be_paired_and_ordered_and_match_confidence_set():
    kwargs_pair = _base_estimate_kwargs()
    kwargs_pair["ci_lower_by_time"] = pd.Series([None, 0.8], index=["2020-04", "2020-05"])
    with pytest.raises(ValueError, match="ci_lower_by_time and ci_upper_by_time must be paired per period"):
        PanelEstimate(**kwargs_pair)

    kwargs_order = _base_estimate_kwargs()
    kwargs_order["ci_lower_by_time"] = pd.Series([2.0, 0.8], index=["2020-04", "2020-05"])
    kwargs_order["ci_upper_by_time"] = pd.Series([1.0, 1.9], index=["2020-04", "2020-05"])
    kwargs_order["confidence_set_by_time"] = {
        "2020-04": [(1.0, 2.0)],
        "2020-05": [(0.8, 1.9)],
    }
    with pytest.raises(ValueError, match="ci_lower_by_time must be <= ci_upper_by_time at post index 0"):
        PanelEstimate(**kwargs_order)


def test_alpha_and_pvalue_by_time_must_be_valid():
    kwargs_alpha = _base_estimate_kwargs()
    kwargs_alpha["alpha"] = 1.0
    with pytest.raises(ValueError, match="alpha must be finite and in \\(0, 1\\)"):
        PanelEstimate(**kwargs_alpha)

    kwargs_pvalue = _base_estimate_kwargs()
    kwargs_pvalue["p_value_by_time"] = pd.Series([0.2, 1.5], index=["2020-04", "2020-05"])
    with pytest.raises(ValueError, match="p_value_by_time values must be in \\[0, 1\\]"):
        PanelEstimate(**kwargs_pvalue)


def test_augmented_weight_sum_enforced_only_when_configured():
    kwargs_enforced = _base_estimate_kwargs()
    kwargs_enforced["donor_weights_augmented"] = {"C1": 0.8, "C2": 0.5}
    kwargs_enforced["diagnostics"] = {"enforce_sum_to_one_augmented": True}
    with pytest.raises(ValueError, match="donor_weights_augmented must sum to 1"):
        PanelEstimate(**kwargs_enforced)

    kwargs_not_enforced = _base_estimate_kwargs()
    kwargs_not_enforced["donor_weights_augmented"] = {"C1": 0.8, "C2": 0.5}
    kwargs_not_enforced["diagnostics"] = {"enforce_sum_to_one_augmented": False}
    est = PanelEstimate(**kwargs_not_enforced)
    assert isinstance(est, PanelEstimate)


def test_legacy_fields_are_rejected():
    kwargs = _base_estimate_kwargs()
    kwargs["intervention_time"] = 4
    kwargs["time"] = "2026-02-22"
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PanelEstimate(**kwargs)


def test_created_at_must_be_timezone_aware():
    kwargs = _base_estimate_kwargs()
    kwargs["created_at"] = datetime(2026, 2, 22)
    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        PanelEstimate(**kwargs)


def test_at_least_one_donor_weight_required():
    kwargs = _base_estimate_kwargs()
    kwargs["donor_weights_augmented"] = {}
    with pytest.raises(ValueError, match="At least one donor weight is required"):
        PanelEstimate(**kwargs)
