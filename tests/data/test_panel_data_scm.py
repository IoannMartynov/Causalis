from typing import Optional

import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelDataSCM


def _base_panel_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["T", "T", "T", "C1", "C1", "C1", "C2", "C2", "C2"],
            "time_id": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ],
            "y": [10.0, 11.0, 13.0, 9.0, 9.5, 10.0, 8.5, 9.0, 9.2],
            "treated_time": [0, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )


def _base_kwargs(df: Optional[pd.DataFrame] = None, **overrides) -> dict:
    kwargs = {
        "unit_col": "unit_id",
        "time_col": "time_id",
        "y": "y",
        "treated_time": "treated_time",
        "df": _base_panel_df() if df is None else df,
    }
    kwargs.update(overrides)
    return kwargs


def test_minimum_contract_and_helpers_work():
    panel = PanelDataSCM(**_base_kwargs())

    assert panel.df.columns.tolist() == ["unit_id", "time_id", "treated_time", "observed", "y"]
    assert set(panel.df["observed"].unique()) == {1}
    assert sorted(panel.donor_pool()) == ["C1", "C2"]
    assert [str(t) for t in panel.pre_times()] == ["2020-01"]
    assert [str(t) for t in panel.post_times()] == ["2020-02", "2020-03"]
    assert [str(t) for t in panel.analysis_times()] == ["2020-01", "2020-02", "2020-03"]
    assert panel.treatment_start_idx() == 1
    assert panel.n_pre_periods == 1
    assert panel.n_post_periods == 2

    analysis = panel.df_analysis()
    assert set(analysis["unit_id"].unique()) == {"T", "C1", "C2"}
    assert {str(t) for t in analysis["time_id"].unique()} == {"2020-01", "2020-02", "2020-03"}


def test_missing_required_columns_raise():
    df = _base_panel_df().drop(columns=["y"])

    with pytest.raises(ValueError, match="Missing required columns"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_treated_time_field_is_required():
    kwargs = _base_kwargs()
    kwargs.pop("treated_time")

    with pytest.raises(ValueError, match="treated_time"):
        PanelDataSCM(**kwargs)


def test_column_role_names_must_be_distinct():
    with pytest.raises(ValueError, match="Column role names must be distinct"):
        PanelDataSCM(**_base_kwargs(y="unit_id"))


def test_unit_and_time_keys_must_be_non_null():
    df_unit_null = _base_panel_df()
    df_unit_null.loc[0, "unit_id"] = np.nan
    with pytest.raises(ValueError, match="unit_id' contains nulls"):
        PanelDataSCM(**_base_kwargs(df=df_unit_null))

    df_time_null = _base_panel_df()
    df_time_null.loc[0, "time_id"] = np.nan
    with pytest.raises(ValueError, match="time_id' contains nulls"):
        PanelDataSCM(**_base_kwargs(df=df_time_null))


def test_duplicate_unit_time_rejected():
    df = _base_panel_df()
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=r"Duplicate \(unit,time\) rows"):
        PanelDataSCM(**_base_kwargs(df=df_dup))


def test_treated_time_must_be_binary_and_non_null():
    df_non_binary = _base_panel_df()
    df_non_binary.loc[0, "treated_time"] = 2
    with pytest.raises(ValueError, match="must be boolean or 0/1"):
        PanelDataSCM(**_base_kwargs(df=df_non_binary))

    df_null = _base_panel_df()
    df_null.loc[0, "treated_time"] = np.nan
    with pytest.raises(ValueError, match="treated_time' contains nulls"):
        PanelDataSCM(**_base_kwargs(df=df_null))


def test_treated_time_must_be_consistent_within_unit_time_cell():
    df = _base_panel_df()
    conflict_row = df.iloc[[0]].copy()
    conflict_row.loc[:, "treated_time"] = 1
    df_conflict = pd.concat([df, conflict_row], ignore_index=True)

    with pytest.raises(ValueError, match="must be consistent within each"):
        PanelDataSCM(**_base_kwargs(df=df_conflict))


def test_treated_time_must_have_at_least_one_treated_row():
    df = _base_panel_df()
    df["treated_time"] = 0

    with pytest.raises(ValueError, match="must have at least one treated row"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_treated_time_must_identify_exactly_one_treated_unit():
    df = _base_panel_df()
    df.loc[(df["unit_id"] == "C1") & (df["time_id"] == "2020-03-01"), "treated_time"] = 1

    with pytest.raises(ValueError, match="must identify exactly one treated unit"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_treated_unit_must_stay_treated_after_start():
    df = _base_panel_df()
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == "2020-03-01"), "treated_time"] = 0

    with pytest.raises(ValueError, match="for treated_unit must be 1 at/after treatment_start"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_implicit_donor_pool_must_have_at_least_two_units():
    df_none = pd.DataFrame(
        {
            "unit_id": ["T", "T", "T"],
            "time_id": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "y": [1.0, 2.0, 3.0],
            "treated_time": [0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        PanelDataSCM(**_base_kwargs(df=df_none))

    df_single = pd.DataFrame(
        {
            "unit_id": ["T", "T", "T", "C1", "C1", "C1"],
            "time_id": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ],
            "y": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
            "treated_time": [0, 1, 1, 0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        PanelDataSCM(**_base_kwargs(df=df_single))


def test_each_donor_must_have_pre_treatment_rows():
    df = _base_panel_df()
    df = df[~((df["unit_id"] == "C2") & (df["time_id"] == "2020-01-01"))].copy()

    with pytest.raises(ValueError, match="Each donor must have at least one pre-treatment row"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_treated_post_outcomes_must_be_observed():
    df = _base_panel_df().copy()
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == "2020-03-01"), "y"] = np.nan

    with pytest.raises(ValueError, match="treated_unit must have observed y"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_df_is_projected_to_estimation_columns_only():
    df = _base_panel_df().copy()
    df["x1"] = np.arange(len(df), dtype=float)
    df["note"] = "meta"

    panel = PanelDataSCM(**_base_kwargs(df=df))

    assert panel.df.columns.tolist() == ["unit_id", "time_id", "treated_time", "observed", "y"]
    assert "x1" not in panel.df.columns
    assert "note" not in panel.df.columns


def test_no_pre_treatment_periods_is_rejected():
    df = _base_panel_df().copy()
    df.loc[df["unit_id"] == "T", "treated_time"] = 1

    with pytest.raises(ValueError, match="No pre-treatment periods available"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_time_is_normalized_to_periods():
    df = _base_panel_df().copy()
    df["time_id"] = df["time_id"].astype(str)
    panel = PanelDataSCM(**_base_kwargs(df=df))

    assert str(panel.df["time_id"].dtype) == "period[M]"
    assert [str(t) for t in panel.pre_times()] == ["2020-01"]
    assert [str(t) for t in panel.post_times()] == ["2020-02", "2020-03"]


def test_time_coercion_rejects_timezone_aware_values():
    df = _base_panel_df().copy()
    df["time_id"] = pd.to_datetime(df["time_id"], utc=True)

    with pytest.raises(ValueError, match="timezone-aware datetimes"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_time_coercion_rejects_numeric_and_two_point_datetime_values():
    rows_numeric = []
    for t in range(1, 4):
        rows_numeric.extend(
            [
                {"unit_id": "T", "time_id": t, "y": 10.0 + t, "treated_time": int(t >= 2)},
                {"unit_id": "C1", "time_id": t, "y": 8.0 + t, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": 9.0 + t, "treated_time": 0},
            ]
        )
    with pytest.raises(ValueError, match="explicit calendar time"):
        PanelDataSCM(**_base_kwargs(df=pd.DataFrame(rows_numeric)))

    rows_two_points = []
    for idx, t in enumerate(["2020-01-01", "2020-02-01"], start=1):
        rows_two_points.extend(
            [
                {"unit_id": "T", "time_id": t, "y": 10.0 + idx, "treated_time": int(idx >= 2)},
                {"unit_id": "C1", "time_id": t, "y": 8.0 + idx, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": 9.0 + idx, "treated_time": 0},
            ]
        )
    with pytest.raises(ValueError, match="fewer than 3 unique datetime values"):
        PanelDataSCM(**_base_kwargs(df=pd.DataFrame(rows_two_points)))


def test_gapped_time_axis_can_be_rejected():
    df = _base_panel_df()
    df = df[df["time_id"] != "2020-02-01"].copy()
    df["time_id"] = pd.PeriodIndex(df["time_id"], freq="M")

    with pytest.raises(ValueError, match="Analysis time axis has gaps"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_outcome_must_be_numeric():
    df = _base_panel_df().copy()
    df["y"] = df["y"].astype(object)
    df.loc[0, "y"] = "bad"

    with pytest.raises(ValueError, match="contains non-numeric values"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_panel_repr_is_compact_and_informative():
    panel = PanelDataSCM(**_base_kwargs())

    repr_str = repr(panel)
    assert repr_str.startswith("PanelDataSCM(df=(9, 5),")
    assert "unit_col='unit_id'" in repr_str
    assert "time_col='time_id'" in repr_str
    assert "treated_time='treated_time'" in repr_str
    assert "time_freq='M'" in repr_str
    assert "y='y'" in repr_str
    assert "treated_unit='T'" in repr_str
    assert "treatment_start=Period('2020-02'" in repr_str
    assert "n_pre_periods=1" in repr_str
    assert "n_post_periods=2" in repr_str
    assert "donor_units=['C1', 'C2']" in repr_str
