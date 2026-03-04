import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelDataSCM
from causalis.scenarios.synthetic_control import ASCM, RSCM, SCM, SyntheticControl


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


def test_scm_defaults_to_ascm_when_fully_observed():
    df = _make_panel_with_effect(effect=2.0)
    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )

    estimate_auto = SyntheticControl(lambda_aug=0.5).fit(data).estimate()
    estimate_ascm = ASCM(lambda_aug=0.5).fit(data).estimate()

    assert estimate_auto.estimand == "ATTE"
    assert estimate_auto.model == "AugmentedSyntheticControl"
    assert abs(float(estimate_auto.att) - float(estimate_ascm.att)) < 1e-9
    assert abs(float(estimate_auto.att_sc) - float(estimate_ascm.att_sc)) < 1e-9
    assert estimate_auto.diagnostics["selected_model"] == "AugmentedSyntheticControl"


def test_scm_forces_rscm_when_missing_outcomes_present():
    df = _make_panel_with_effect(effect=3.0)
    df = df[~((df["unit_id"] == "C2") & (df["time_id"] == "2020-02-01"))].copy()
    df.loc[(df["unit_id"] == "C1") & (df["time_id"] == "2020-03-01"), "y"] = np.nan
    data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )

    estimate_auto = SyntheticControl(lambda_aug=0.5, completion_max_iter=250).fit(data).estimate()
    estimate_rscm = RSCM(lambda_aug=0.5, completion_max_iter=250).fit(data).estimate()

    assert estimate_auto.model == "RobustSyntheticControl"
    assert abs(float(estimate_auto.att) - float(estimate_rscm.att)) < 1e-9
    assert abs(float(estimate_auto.att_sc) - float(estimate_rscm.att_sc)) < 1e-9
    assert estimate_auto.diagnostics["selected_model"] == "RobustSyntheticControl"


def test_scm_alias_and_not_fitted_guard():
    model = SCM()
    with pytest.raises(RuntimeError, match="fit"):
        model.estimate()


def test_scm_contract_rejects_missing_treated_post_outcomes():
    df = _make_panel_with_effect(effect=2.0)
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == "2020-06-01"), "y"] = np.nan
    with pytest.raises(ValueError, match="treated_unit must have observed y"):
        PanelDataSCM(
            unit_col="unit_id",
            time_col="time_id",
            treated_time="treated_time",
            y="y",
            df=df,
        )


def test_scm_failed_refit_does_not_return_stale_estimate():
    valid_df = _make_panel_with_effect(effect=2.0)
    valid_data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=valid_df,
    )

    model = SyntheticControl(lambda_aug=0.5, min_pre_observed=4).fit(valid_data)
    assert model.estimate().model == "AugmentedSyntheticControl"

    invalid_df = _make_panel_with_effect(effect=2.0)
    invalid_df.loc[(invalid_df["unit_id"] == "T") & (invalid_df["time_id"] == "2020-01-01"), "y"] = np.nan
    invalid_data = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=invalid_df,
    )

    with pytest.raises(ValueError, match="observed treated pre-treatment outcomes"):
        model.fit(invalid_data)

    with pytest.raises(RuntimeError, match="fit"):
        model.estimate()


def test_contract_rejects_gapped_analysis_time_axis():
    rows = []
    for idx, t in enumerate([pd.Period("2020-01", freq="M"), pd.Period("2020-03", freq="M")], start=1):
        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": 10.0 + idx, "treated_time": int(idx >= 2)},
                {"unit_id": "C1", "time_id": t, "y": 8.0 + idx, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": 9.0 + idx, "treated_time": 0},
            ]
        )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="Analysis time axis has gaps"):
        PanelDataSCM(
            unit_col="unit_id",
            time_col="time_id",
            treated_time="treated_time",
            y="y",
            df=df,
        )


def test_contract_rejects_two_point_datetime_frequency_inference():
    rows = []
    for idx, t in enumerate(["2020-01-01", "2020-02-01"], start=1):
        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": 10.0 + idx, "treated_time": int(idx >= 2)},
                {"unit_id": "C1", "time_id": t, "y": 8.0 + idx, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": 9.0 + idx, "treated_time": 0},
            ]
        )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="fewer than 3 unique datetime values"):
        PanelDataSCM(
            unit_col="unit_id",
            time_col="time_id",
            treated_time="treated_time",
            y="y",
            df=df,
        )


def test_contract_rejects_numeric_time_column():
    rows = []
    for t in range(1, 7):
        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": 10.0 + t, "treated_time": int(t >= 4)},
                {"unit_id": "C1", "time_id": t, "y": 8.0 + t, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": 9.0 + t, "treated_time": 0},
            ]
        )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="explicit calendar time"):
        PanelDataSCM(
            unit_col="unit_id",
            time_col="time_id",
            treated_time="treated_time",
            y="y",
            df=df,
        )


def test_contract_uses_validated_snapshot_when_public_df_is_mutated():
    df = _make_panel_with_effect(effect=2.0)
    panel = PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        y="y",
        df=df,
    )
    original_donors = panel.donor_pool()
    panel.df.loc[:, "unit_id"] = "T"
    assert panel.donor_pool() == original_donors
