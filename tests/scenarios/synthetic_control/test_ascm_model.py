import numpy as np
import pandas as pd
import pytest

import causalis.scenarios.synthetic_control.model as sc_model
from causalis.data_contracts import PanelDataSCM, PanelEstimate
from causalis.scenarios.synthetic_control import ASCM, AugmentedSyntheticControl


def _make_panel_with_effect(effect: float = 2.5) -> pd.DataFrame:
    rows = []
    for idx, t in enumerate(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"],
        start=1,
    ):
        y_c1 = 10.0 + 0.5 * idx
        y_c2 = 12.0 + 0.2 * idx
        y_treat = 0.65 * y_c1 + 0.35 * y_c2
        if idx >= 4:
            y_treat += effect

        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": y_treat, "treated_time": 1 if idx >= 4 else 0},
                {"unit_id": "C1", "time_id": t, "y": y_c1, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": y_c2, "treated_time": 0},
            ]
        )
    return pd.DataFrame(rows)


def _panel(df: pd.DataFrame, **overrides) -> PanelDataSCM:
    kwargs = {
        "unit_col": "unit_id",
        "time_col": "time_id",
        "y": "y",
        "treated_time": "treated_time",
        "df": df,
    }
    kwargs.update(overrides)
    return PanelDataSCM(**kwargs)


def test_ascm_fit_and_estimate_interface():
    df = _make_panel_with_effect(effect=3.0)
    data = _panel(df)

    model = AugmentedSyntheticControl(lambda_aug=0.5).fit(data)
    estimate = model.estimate()

    assert isinstance(estimate, PanelEstimate)
    assert estimate.estimand == "dynamic_effect_path"
    assert estimate.model == "AugmentedSyntheticControl"
    assert len(estimate.pre_times) == 3
    assert len(estimate.post_times) == 3
    assert len(estimate.effect_by_time) == 3
    assert set(estimate.donor_weights_augmented.keys()) == {"C1", "C2"}
    assert float(estimate.effect_by_time.mean()) > 2.0
    assert len(estimate.ci_lower_by_time) == len(estimate.post_times)
    assert len(estimate.ci_upper_by_time) == len(estimate.post_times)
    assert estimate.alpha == 0.05
    pvals_non_missing = estimate.p_value_by_time.dropna()
    assert ((pvals_non_missing >= 0.0) & (pvals_non_missing <= 1.0)).all()
    assert estimate.is_significant_by_time.isin([True, False]).all()
    assert estimate.diagnostics["n_pre_periods"] == len(estimate.pre_times)
    assert estimate.diagnostics["n_post_periods"] == len(estimate.post_times)
    assert estimate.diagnostics["estimand"] == "dynamic_effect_path"
    assert estimate.diagnostics["average_att_ttest_available"] in {True, False}


def test_ascm_estimate_inference_overrides_are_supported():
    df = _make_panel_with_effect(effect=3.0)
    data = _panel(df)

    model = AugmentedSyntheticControl(alpha=0.2, compute_average_att_ttest=True).fit(data)
    default_estimate = model.estimate()
    overridden = model.estimate(alpha=0.1, compute_average_att_ttest=False)
    default_estimate_again = model.estimate()

    assert default_estimate.alpha == 0.2
    assert overridden.alpha == 0.1
    assert overridden.diagnostics["ci_alpha"] == pytest.approx(0.1)
    assert overridden.diagnostics["average_att_ttest_requested"] is False
    assert overridden.diagnostics["average_att_ttest_available"] is False
    assert default_estimate_again.alpha == 0.2
    assert default_estimate_again.diagnostics["average_att_ttest_requested"] is True


def test_ascm_estimate_overrides_do_not_mutate_fit_warning_state(monkeypatch):
    class FailedResult:
        success = False
        message = "Positive directional derivative for linesearch"
        x = None

    def _always_fail(*args, **kwargs):
        return FailedResult()

    data = _panel(_make_panel_with_effect(effect=3.0))
    model = AugmentedSyntheticControl(alpha=0.2, compute_average_att_ttest=True).fit(data)

    baseline = model.estimate()
    base_count = int(model._slsqp_fallback_count)
    base_reasons = list(model._slsqp_fallback_reasons)
    base_warnings = list(model._stability_warning_messages)

    monkeypatch.setattr(sc_model, "minimize", _always_fail)
    overridden = model.estimate(alpha=0.1)
    overridden_again = model.estimate(alpha=0.15)

    assert model._slsqp_fallback_count == base_count
    assert model._slsqp_fallback_reasons == base_reasons
    assert model._stability_warning_messages == base_warnings

    assert overridden.diagnostics["slsqp_fallback_count"] == baseline.diagnostics["slsqp_fallback_count"]
    assert overridden.diagnostics["slsqp_fallback_reasons"] == baseline.diagnostics["slsqp_fallback_reasons"]
    assert overridden.diagnostics["stability_warning_messages"] == baseline.diagnostics[
        "stability_warning_messages"
    ]
    assert overridden.diagnostics["suppressed_fit_warnings"] == baseline.diagnostics["suppressed_fit_warnings"]

    assert overridden_again.diagnostics["slsqp_fallback_count"] == baseline.diagnostics[
        "slsqp_fallback_count"
    ]


def test_ascm_estimate_inference_override_validation():
    data = _panel(_make_panel_with_effect(effect=2.0))
    model = AugmentedSyntheticControl().fit(data)

    with pytest.raises(ValueError, match="alpha must be finite and in \\(0, 1\\)"):
        model.estimate(alpha=1.0)
    with pytest.raises(ValueError, match="average_att_n_folds"):
        model.estimate(average_att_n_folds=1)
    with pytest.raises(ValueError, match="conformal_grid_size"):
        model.estimate(compute_pointwise_conformal=True, conformal_grid_size=100)


def test_ascm_alias_and_not_fitted_guard():
    model = ASCM()
    with pytest.raises(RuntimeError, match="fit"):
        model.estimate()


def test_ascm_requires_balanced_block_no_missing_cells():
    df = _make_panel_with_effect()
    df = df[~((df["unit_id"] == "C2") & (df["time_id"] == "2020-02-01"))].copy()
    data = _panel(df)

    with pytest.raises(ValueError, match="balanced block"):
        AugmentedSyntheticControl().fit(data)


def test_ascm_requires_balanced_block_no_missing_outcomes():
    df = _make_panel_with_effect()
    df.loc[(df["unit_id"] == "C1") & (df["time_id"] == "2020-03-01"), "y"] = np.nan
    with pytest.raises(ValueError, match="contains nulls"):
        _panel(df)


def test_ascm_requires_nonempty_pre_and_post():
    df = _make_panel_with_effect()
    df.loc[df["unit_id"] == "T", "treated_time"] = 1
    with pytest.raises(ValueError, match="No pre-treatment periods available"):
        _panel(df)


def test_ascm_contract_requires_at_least_two_donors():
    df = _make_panel_with_effect(effect=2.0)
    df = df[df["unit_id"].isin(["T", "C1"])].copy()
    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        _panel(df)


def test_ascm_rejects_invalid_explicit_periods():
    df = _make_panel_with_effect()
    with pytest.raises(ValueError, match="pre_periods"):
        _panel(
            df,
            pre_periods=["2020-01-01", "2020-02-01", "2020-04-01"],
            post_periods=["2020-04-01", "2020-05-01", "2020-06-01"],
        )


def test_augmented_weights_satisfy_constrained_kkt_conditions():
    df = _make_panel_with_effect(effect=2.0)
    data = _panel(df)
    panel = data.df_analysis().pivot(index="unit_id", columns="time_id", values="y")
    donors = list(data.donor_pool())
    pre = list(data.pre_times())
    x0_pre = panel.loc[donors, pre].to_numpy(dtype=float).T
    y1_pre = panel.loc[data.treated_unit, pre].to_numpy(dtype=float)

    model = AugmentedSyntheticControl(lambda_aug=0.75, enforce_sum_to_one_augmented=True)
    w_sc = model._fit_simplex_weights(x0_pre=x0_pre, y1_pre=y1_pre)
    w_aug = model._augment_weights(x0_pre=x0_pre, y1_pre=y1_pre, w_sc=w_sc)

    gram = x0_pre.T @ x0_pre + model.lambda_aug * np.eye(x0_pre.shape[1], dtype=float)
    rhs = x0_pre.T @ y1_pre + model.lambda_aug * w_sc
    stationarity = gram @ w_aug - rhs

    assert abs(float(np.sum(w_aug)) - 1.0) < 1e-10
    assert np.max(np.abs(stationarity - np.mean(stationarity))) < 1e-10


def test_augmented_weights_satisfy_unconstrained_normal_equations():
    df = _make_panel_with_effect(effect=2.0)
    data = _panel(df)
    panel = data.df_analysis().pivot(index="unit_id", columns="time_id", values="y")
    donors = list(data.donor_pool())
    pre = list(data.pre_times())
    x0_pre = panel.loc[donors, pre].to_numpy(dtype=float).T
    y1_pre = panel.loc[data.treated_unit, pre].to_numpy(dtype=float)

    model = AugmentedSyntheticControl(lambda_aug=0.75, enforce_sum_to_one_augmented=False)
    w_sc = model._fit_simplex_weights(x0_pre=x0_pre, y1_pre=y1_pre)
    w_aug = model._augment_weights(x0_pre=x0_pre, y1_pre=y1_pre, w_sc=w_sc)

    gram = x0_pre.T @ x0_pre + model.lambda_aug * np.eye(x0_pre.shape[1], dtype=float)
    rhs = x0_pre.T @ y1_pre + model.lambda_aug * w_sc
    stationarity = gram @ w_aug - rhs

    assert np.max(np.abs(stationarity)) < 1e-10


def test_simplex_weights_fallback_if_slsqp_fails(monkeypatch):
    class FailedResult:
        success = False
        message = "Positive directional derivative for linesearch"
        x = None

    def _always_fail(*args, **kwargs):
        return FailedResult()

    monkeypatch.setattr(sc_model, "minimize", _always_fail)

    model = AugmentedSyntheticControl(lambda_sc=1e-6, max_iter=200, tol=1e-9)
    x0_pre = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    y1_pre = np.array([1.5, 2.5, 3.5, 4.5], dtype=float)

    w = model._fit_simplex_weights(x0_pre=x0_pre, y1_pre=y1_pre)

    assert np.isfinite(w).all()
    assert np.all(w >= -1e-12)
    assert abs(float(np.sum(w)) - 1.0) < 1e-10
