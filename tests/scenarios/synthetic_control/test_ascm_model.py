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
                {"unit_id": "T", "time_id": t, "y": y_treat},
                {"unit_id": "C1", "time_id": t, "y": y_c1},
                {"unit_id": "C2", "time_id": t, "y": y_c2},
            ]
        )
    return pd.DataFrame(rows)


def _panel(df: pd.DataFrame, **overrides) -> PanelDataSCM:
    kwargs = {
        "unit_col": "unit_id",
        "time_col": "time_id",
        "y": "y",
        "df": df,
        "treated_unit": "T",
        "treatment_start": "2020-04-01",
    }
    kwargs.update(overrides)
    return PanelDataSCM(**kwargs)


def test_ascm_fit_and_estimate_interface():
    df = _make_panel_with_effect(effect=3.0)
    data = _panel(df)

    model = AugmentedSyntheticControl(lambda_aug=0.5).fit(data)
    estimate = model.estimate()

    assert isinstance(estimate, PanelEstimate)
    assert estimate.estimand == "ATTE"
    assert estimate.model == "AugmentedSyntheticControl"
    assert len(estimate.pre_times) == 3
    assert len(estimate.post_times) == 3
    assert len(estimate.att_by_time) == 3
    assert set(estimate.donor_weights_sc.keys()) == {"C1", "C2"}
    assert estimate.att > 2.0
    if estimate.ci_lower_absolute is not None and estimate.ci_upper_absolute is not None:
        assert estimate.ci_lower_absolute <= estimate.ci_upper_absolute
    else:
        assert estimate.diagnostics["att_placebo_ci_is_unbounded"] is True
        assert estimate.diagnostics["att_placebo_min_possible_p"] > estimate.alpha
    assert estimate.alpha == 0.05
    assert estimate.p_value is not None
    assert 0.0 <= estimate.p_value <= 1.0
    assert isinstance(estimate.is_significant, bool)
    assert estimate.diagnostics["att_pre_resid_n"] == len(estimate.pre_times)
    assert estimate.diagnostics["att_p_value_method"] == "placebo_in_space_att"
    assert estimate.diagnostics["att_placebo_n"] >= 1
    assert estimate.diagnostics["att_is_significant_fit_adjusted"] in {True, False, None}
    warning = estimate.diagnostics["att_fit_adjusted_warning"]
    assert warning is None or isinstance(warning, str)
    if estimate.is_significant is True and estimate.diagnostics["att_is_significant_fit_adjusted"] is False:
        assert warning is not None
    if estimate.value_relative is not None and estimate.ci_lower_relative is not None:
        assert estimate.ci_upper_relative is not None
        assert estimate.ci_lower_relative <= estimate.ci_upper_relative


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
    data = _panel(df)

    with pytest.raises(ValueError, match="balanced block"):
        AugmentedSyntheticControl().fit(data)


def test_ascm_requires_nonempty_pre_and_post():
    df = _make_panel_with_effect()
    with pytest.raises(ValueError, match="No post-treatment periods available"):
        _panel(df, treatment_start="2020-10-01")


def test_ascm_contract_requires_at_least_two_donors():
    df = _make_panel_with_effect(effect=2.0)
    df = df[df["unit_id"].isin(["T", "C1"])].copy()
    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        _panel(df)


def test_ascm_rejects_invalid_explicit_periods():
    df = _make_panel_with_effect()
    with pytest.raises(ValueError, match="pre_periods must contain only periods < treatment_start"):
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
