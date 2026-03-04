import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelDataSCM, PanelEstimate
from causalis.scenarios.synthetic_control import ASCM, RSCM, RobustSyntheticControl


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
                {"unit_id": "T", "time_id": t, "y": y_treat, "treated_time": int(idx >= 4)},
                {"unit_id": "C1", "time_id": t, "y": y_c1, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": y_c2, "treated_time": 0},
            ]
        )
    return pd.DataFrame(rows)


def test_rscm_fit_and_estimate_handles_missing_cells_and_outcomes():
    df = _make_panel_with_effect(effect=3.0)
    df = df[~((df["unit_id"] == "C2") & (df["time_id"] == "2020-02-01"))].copy()
    df.loc[(df["unit_id"] == "C1") & (df["time_id"] == "2020-03-01"), "y"] = np.nan
    df["observed"] = (~df["y"].isna()).astype(int)

    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )

    estimate = RobustSyntheticControl(lambda_aug=0.5, completion_max_iter=250).fit(data).estimate()

    assert isinstance(estimate, PanelEstimate)
    assert estimate.model == "RobustSyntheticControl"
    assert np.isfinite(estimate.att)
    assert not estimate.observed_outcome.isna().any()
    assert estimate.diagnostics["n_missing_cells"] >= 2
    assert estimate.diagnostics["treated_missing_post"] == 0
    assert estimate.diagnostics["completion_iterations"] >= 0
    assert set(estimate.donor_weights_sc.keys()) == {"C1", "C2"}
    if estimate.ci_lower_absolute is not None and estimate.ci_upper_absolute is not None:
        assert estimate.ci_lower_absolute <= estimate.ci_upper_absolute
    else:
        assert estimate.diagnostics["att_placebo_ci_is_unbounded"] is True
        assert estimate.diagnostics["att_placebo_min_possible_p"] > estimate.alpha
    assert estimate.alpha == 0.05
    assert estimate.p_value is not None
    assert 0.0 <= estimate.p_value <= 1.0
    assert isinstance(estimate.is_significant, bool)
    assert estimate.diagnostics["att_pre_resid_n"] >= 1
    assert estimate.diagnostics["att_p_value_method"] == "placebo_in_space_att"
    assert estimate.diagnostics["att_is_significant_fit_adjusted"] in {True, False, None}
    warning = estimate.diagnostics["att_fit_adjusted_warning"]
    assert warning is None or isinstance(warning, str)
    if estimate.is_significant is True and estimate.diagnostics["att_is_significant_fit_adjusted"] is False:
        assert warning is not None
    if estimate.value_relative is not None and estimate.ci_lower_relative is not None:
        assert estimate.ci_upper_relative is not None
        assert estimate.ci_lower_relative <= estimate.ci_upper_relative


def test_rscm_contract_requires_observed_treated_post_for_att():
    df = _make_panel_with_effect(effect=3.0)
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == "2020-06-01"), "y"] = np.nan
    with pytest.raises(ValueError, match="treated_unit must have observed y"):
        PanelDataSCM(
            unit_col="unit_id",
            time_col="time_id",
            treated_time="treated_time",
            y="y",
            df=df,
        )


def test_rscm_contract_requires_at_least_two_donors():
    df = _make_panel_with_effect(effect=2.0)
    df = df[df["unit_id"].isin(["T", "C1"])].copy()
    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        PanelDataSCM(
            unit_col="unit_id",
            time_col="time_id",
            treated_time="treated_time",
            y="y",
            df=df,
        )


def test_rscm_alias_and_not_fitted_guard():
    model = RSCM()
    with pytest.raises(RuntimeError, match="fit"):
        model.estimate()


def test_rscm_matches_ascm_on_fully_observed_panel():
    df = _make_panel_with_effect(effect=2.0)
    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )

    ascm = ASCM(lambda_aug=0.5, inference_policy="placebo").fit(data).estimate()
    rscm = RSCM(lambda_aug=0.5).fit(data).estimate()

    assert abs(float(ascm.att) - float(rscm.att)) < 1e-9
    assert abs(float(ascm.att_sc) - float(rscm.att_sc)) < 1e-9


def test_rscm_requires_enough_observed_treated_pre_periods():
    df = _make_panel_with_effect(effect=2.0)
    df.loc[
        (df["unit_id"] == "T") & (df["time_id"].isin(["2020-01-01", "2020-02-01", "2020-03-01"])),
        "y",
    ] = np.nan
    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )

    with pytest.raises(ValueError, match="observed treated pre-treatment outcomes"):
        RSCM(min_pre_observed=1).fit(data)


def test_rscm_inference_uses_only_observed_treated_pre_residuals():
    df = _make_panel_with_effect(effect=2.0)
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == "2020-02-01"), "y"] = np.nan
    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )

    estimate = RSCM(lambda_aug=0.5, completion_max_iter=250).fit(data).estimate()
    assert estimate.diagnostics["treated_observed_pre_for_inference"] == 2
    assert estimate.diagnostics["att_pre_resid_n"] == 2
    assert estimate.diagnostics["att_sc_pre_resid_n"] == 2


def test_rscm_masks_treated_post_during_completion(monkeypatch):
    df = _make_panel_with_effect(effect=2.0)
    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )

    captured: dict[str, np.ndarray] = {}
    original_complete = RobustSyntheticControl._complete_low_rank_matrix

    def spy_complete(self, y_matrix: np.ndarray, observed_mask: np.ndarray):
        captured["mask"] = observed_mask.copy()
        return original_complete(self, y_matrix=y_matrix, observed_mask=observed_mask)

    monkeypatch.setattr(RobustSyntheticControl, "_complete_low_rank_matrix", spy_complete)
    RSCM(lambda_aug=0.5).fit(data)

    assert "mask" in captured
    n_pre = len(data.pre_times())
    assert not captured["mask"][0, n_pre:].any()
