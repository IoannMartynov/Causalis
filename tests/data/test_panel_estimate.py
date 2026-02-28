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
        "att": 1.25,
        "att_sc": 1.1,
        "ci_upper_absolute": 1.8,
        "ci_lower_absolute": 0.7,
        "value_relative": 9.5,
        "ci_upper_relative": 13.2,
        "ci_lower_relative": 5.8,
        "alpha": 0.05,
        "p_value": 0.0033,
        "is_significant": True,
        "att_by_time": pd.Series([1.2, 1.3], index=post),
        "att_by_time_sc": pd.Series([1.0, 1.2], index=post),
        "observed_outcome": pd.Series([10, 11, 12, 13, 14], index=all_times),
        "synthetic_outcome": pd.Series([9.9, 10.8, 11.7, 11.8, 12.6], index=all_times),
        "synthetic_outcome_sc": pd.Series([9.8, 10.7, 11.5, 11.6, 12.4], index=all_times),
        "donor_weights_augmented": {"C1": 1.1, "C2": -0.1},
        "donor_weights_sc": {"C1": 0.7, "C2": 0.3},
        "diagnostics": {"enforce_sum_to_one_augmented": True},
    }


def test_panel_estimate_valid_contract_and_summary():
    est = PanelEstimate(**_base_estimate_kwargs())
    summary = est.summary()

    assert est.estimand == "ATTE"
    assert isinstance(est.created_at, datetime)
    assert est.created_at.tzinfo is not None
    assert str(est.treatment_start) == "2020-04"
    assert est.ci_lower_absolute <= est.ci_upper_absolute
    assert est.ci_lower_relative <= est.ci_upper_relative
    assert est.alpha == 0.05
    assert est.p_value == 0.0033
    assert est.is_significant is True
    assert list(summary.index) == [
        "estimand",
        "model",
        "value",
        "value_relative",
        "alpha",
        "p_value",
        "is_significant",
    ]
    assert summary.loc["value", "value"] == "1.2500 (ci_abs: 0.7000, 1.8000)"
    assert summary.loc["value_relative", "value"] == "9.5000 (ci_rel: 5.8000, 13.2000)"
    assert summary.loc["alpha", "value"] == "0.0500"
    assert summary.loc["p_value", "value"] == "0.0033"
    assert bool(summary.loc["is_significant", "value"]) is True


def test_att_by_time_index_must_match_post_times():
    kwargs = _base_estimate_kwargs()
    kwargs["att_by_time"] = pd.Series([1.2, 1.3], index=["2020-05", "2020-04"])

    with pytest.raises(ValueError, match="att_by_time index must exactly equal post_times"):
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
    kwargs_overlap["att_by_time"] = pd.Series([1.2, 1.3], index=["2020-03", "2020-04"])
    kwargs_overlap["att_by_time_sc"] = pd.Series([1.1, 1.2], index=["2020-03", "2020-04"])
    kwargs_overlap["observed_outcome"] = pd.Series(
        [10, 11, 12, 13, 14], index=["2020-01", "2020-02", "2020-03", "2020-03", "2020-04"]
    )
    kwargs_overlap["synthetic_outcome"] = pd.Series(
        [9, 10, 11, 12, 13], index=["2020-01", "2020-02", "2020-03", "2020-03", "2020-04"]
    )
    kwargs_overlap["synthetic_outcome_sc"] = pd.Series(
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
    kwargs_unsorted["synthetic_outcome_sc"] = pd.Series(
        [9, 10, 11, 12, 13], index=["2020-02", "2020-01", "2020-03", "2020-04", "2020-05"]
    )

    with pytest.raises(ValueError, match="must be sorted ascending"):
        PanelEstimate(**kwargs_unsorted)


def test_numeric_finite_checks():
    kwargs_att = _base_estimate_kwargs()
    kwargs_att["att"] = float("inf")
    with pytest.raises(ValueError, match="att and att_sc must be finite"):
        PanelEstimate(**kwargs_att)

    kwargs_series = _base_estimate_kwargs()
    kwargs_series["att_by_time"] = pd.Series([1.2, "bad"], index=["2020-04", "2020-05"])
    with pytest.raises(ValueError, match="att_by_time must contain only numeric values"):
        PanelEstimate(**kwargs_series)


def test_ci_bounds_must_be_paired_and_ordered():
    kwargs_pair = _base_estimate_kwargs()
    kwargs_pair["ci_lower_absolute"] = None
    with pytest.raises(ValueError, match="ci_lower_absolute and ci_upper_absolute must be provided together"):
        PanelEstimate(**kwargs_pair)

    kwargs_order = _base_estimate_kwargs()
    kwargs_order["ci_lower_relative"] = 10.0
    kwargs_order["ci_upper_relative"] = 5.0
    with pytest.raises(ValueError, match="ci_lower_relative must be <= ci_upper_relative"):
        PanelEstimate(**kwargs_order)


def test_relative_ci_requires_relative_value_and_alpha_must_be_valid():
    kwargs_rel = _base_estimate_kwargs()
    kwargs_rel["value_relative"] = None
    with pytest.raises(ValueError, match="value_relative must be provided when relative confidence interval"):
        PanelEstimate(**kwargs_rel)

    kwargs_alpha = _base_estimate_kwargs()
    kwargs_alpha["alpha"] = 1.0
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        PanelEstimate(**kwargs_alpha)

    kwargs_pvalue = _base_estimate_kwargs()
    kwargs_pvalue["p_value"] = 1.5
    with pytest.raises(ValueError, match="p_value must be in \\[0, 1\\]"):
        PanelEstimate(**kwargs_pvalue)


def test_sc_weights_must_be_nonnegative_and_sum_to_one():
    kwargs_negative = _base_estimate_kwargs()
    kwargs_negative["donor_weights_sc"] = {"C1": 1.1, "C2": -0.1}
    with pytest.raises(ValueError, match="must be nonnegative"):
        PanelEstimate(**kwargs_negative)

    kwargs_sum = _base_estimate_kwargs()
    kwargs_sum["donor_weights_sc"] = {"C1": 0.6, "C2": 0.3}
    with pytest.raises(ValueError, match="must sum to 1"):
        PanelEstimate(**kwargs_sum)


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
    kwargs["donor_weights_sc"] = {}
    kwargs["donor_weights_augmented"] = {}
    with pytest.raises(ValueError, match="At least one donor weight is required"):
        PanelEstimate(**kwargs)
