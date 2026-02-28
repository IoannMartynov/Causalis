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
            "x1": [1.0, 1.1, 1.2, 0.8, 0.9, 1.0, 0.7, 0.8, 0.9],
            "observed": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "w": [1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
        }
    )


def _base_kwargs(df: Optional[pd.DataFrame] = None, **overrides) -> dict:
    kwargs = {
        "unit_col": "unit_id",
        "time_col": "time_id",
        "y": "y",
        "df": _base_panel_df() if df is None else df,
        "treated_unit": "T",
        "treatment_start": "2020-02-01",
    }
    kwargs.update(overrides)
    return kwargs


def test_minimum_contract_and_helpers_work():
    panel = PanelDataSCM(**_base_kwargs())

    assert sorted(panel.donor_pool()) == ["C1", "C2"]
    assert [str(t) for t in panel.pre_times()] == ["2020-01"]
    assert [str(t) for t in panel.post_times()] == ["2020-02", "2020-03"]
    assert [str(t) for t in panel.analysis_times()] == ["2020-01", "2020-02", "2020-03"]
    assert panel.treatment_start_idx() == 1

    analysis = panel.df_analysis()
    assert set(analysis["unit_id"].unique()) == {"T", "C1", "C2"}
    assert {str(t) for t in analysis["time_id"].unique()} == {"2020-01", "2020-02", "2020-03"}


def test_explicit_donors_and_window_filter_analysis_df():
    panel = PanelDataSCM(
        **_base_kwargs(
            donor_units=["C1", "C2"],
            time_window=("2020-02-01", "2020-03-01"),
            treatment_start="2020-03-01",
        )
    )

    analysis = panel.df_analysis()
    assert set(analysis["unit_id"].unique()) == {"T", "C1", "C2"}
    assert {str(t) for t in analysis["time_id"].unique()} == {"2020-02", "2020-03"}


def test_explicit_periods_override_time_split_rule():
    panel = PanelDataSCM(
        **_base_kwargs(
            pre_periods=["2020-01-01"],
            post_periods=["2020-03-01"],
        )
    )

    assert [str(t) for t in panel.pre_times()] == ["2020-01"]
    assert [str(t) for t in panel.post_times()] == ["2020-03"]


def test_missing_required_columns_raise():
    df = _base_panel_df().drop(columns=["y"])

    with pytest.raises(ValueError, match="Missing required columns"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_duplicate_unit_time_rejected_by_default_and_optional_override():
    df = _base_panel_df()
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=r"Duplicate \(unit,time\) rows"):
        PanelDataSCM(**_base_kwargs(df=df_dup))

    panel = PanelDataSCM(**_base_kwargs(df=df_dup, allow_duplicate_unit_time=True))
    assert panel.allow_duplicate_unit_time is True


@pytest.mark.parametrize(
    ("overrides", "msg"),
    [
        ({"treated_unit": "Z"}, r"treated_unit='Z' not found"),
        ({"donor_units": ["T", "C1"]}, "must not include treated_unit"),
        ({"donor_units": ["C1", "C9"]}, "unknown unit ids"),
        ({"donor_units": ["C1", "C1"]}, "must contain unique unit ids"),
        ({"donor_units": []}, "at least 2 unique units"),
    ],
)
def test_treated_and_donor_validation(overrides, msg):
    with pytest.raises(ValueError, match=msg):
        PanelDataSCM(**_base_kwargs(**overrides))


def test_missing_outcome_is_gateable():
    df = _base_panel_df()
    df.loc[(df["unit_id"] == "C1") & (df["time_id"] == "2020-01-01"), "y"] = np.nan

    panel = PanelDataSCM(**_base_kwargs(df=df))
    assert panel.allow_missing_outcome is True

    with pytest.raises(ValueError, match="allow_missing_outcome=False"):
        PanelDataSCM(**_base_kwargs(df=df, allow_missing_outcome=False))


def test_observed_col_must_be_bool_or_binary_and_mask_strictness_is_configurable():
    df_non_binary = _base_panel_df()
    df_non_binary.loc[0, "observed"] = 2
    with pytest.raises(ValueError, match="must be boolean or 0/1"):
        PanelDataSCM(**_base_kwargs(df=df_non_binary, observed_col="observed"))

    df_null_observed = _base_panel_df()
    df_null_observed.loc[0, "observed"] = np.nan
    with pytest.raises(ValueError, match="strict_observed_mask=True"):
        PanelDataSCM(**_base_kwargs(df=df_null_observed, observed_col="observed"))

    panel = PanelDataSCM(
        **_base_kwargs(
            df=df_null_observed,
            observed_col="observed",
            strict_observed_mask=False,
        )
    )
    assert panel.strict_observed_mask is False


def test_weights_col_must_be_numeric_and_non_negative():
    df_non_numeric = _base_panel_df()
    df_non_numeric["w"] = df_non_numeric["w"].astype(object)
    df_non_numeric.loc[0, "w"] = "bad"

    with pytest.raises(ValueError, match="contains non-numeric values"):
        PanelDataSCM(**_base_kwargs(df=df_non_numeric, weights_col="w"))

    df_negative = _base_panel_df()
    df_negative.loc[0, "w"] = -0.1

    with pytest.raises(ValueError, match="must be non-negative"):
        PanelDataSCM(**_base_kwargs(df=df_negative, weights_col="w"))


def test_optional_column_references_must_exist():
    with pytest.raises(ValueError, match="covariate_cols contains missing column"):
        PanelDataSCM(**_base_kwargs(covariate_cols=["x_missing"]))

    with pytest.raises(ValueError, match="observed_col not found"):
        PanelDataSCM(**_base_kwargs(observed_col="missing_obs"))

    with pytest.raises(ValueError, match="weights_col not found"):
        PanelDataSCM(**_base_kwargs(weights_col="missing_w"))


def test_df_keeps_only_estimation_columns():
    df = _base_panel_df().copy()
    df["oracle"] = np.arange(len(df))

    panel = PanelDataSCM(
        **_base_kwargs(
            df=df,
            covariate_cols=["x1"],
            observed_col="observed",
        )
    )
    assert panel.df.columns.tolist() == ["unit_id", "time_id", "y", "x1", "observed"]
    assert "w" not in panel.df.columns
    assert "oracle" not in panel.df.columns

    panel_with_weights = PanelDataSCM(**_base_kwargs(df=df, weights_col="w"))
    assert panel_with_weights.df.columns.tolist() == ["unit_id", "time_id", "y", "w"]


def test_unit_and_time_keys_must_be_non_null():
    df_unit_null = _base_panel_df()
    df_unit_null.loc[0, "unit_id"] = np.nan
    with pytest.raises(ValueError, match="unit_id' contains nulls"):
        PanelDataSCM(**_base_kwargs(df=df_unit_null))

    df_time_null = _base_panel_df()
    df_time_null.loc[0, "time_id"] = np.nan
    with pytest.raises(ValueError, match="time_id' contains nulls"):
        PanelDataSCM(**_base_kwargs(df=df_time_null))


def test_implicit_donor_pool_must_have_at_least_two_units():
    df = pd.DataFrame(
        {
            "unit_id": ["T", "T", "T"],
            "time_id": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "y": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_implicit_donor_pool_rejects_single_donor():
    df = pd.DataFrame(
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
        }
    )

    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_time_is_normalized_to_periods():
    df = _base_panel_df().copy()
    df["time_id"] = df["time_id"].astype(str)
    panel = PanelDataSCM(**_base_kwargs(df=df, treatment_start="2020-02-01"))

    assert str(panel.df["time_id"].dtype) == "period[M]"
    assert [str(t) for t in panel.pre_times()] == ["2020-01"]
    assert [str(t) for t in panel.post_times()] == ["2020-02", "2020-03"]


def test_time_coercion_rejects_timezone_aware_values():
    df = _base_panel_df().copy()
    df["time_id"] = pd.to_datetime(df["time_id"], utc=True)

    with pytest.raises(ValueError, match="timezone-aware datetimes"):
        PanelDataSCM(**_base_kwargs(df=df))


def test_outcome_must_be_numeric():
    df = _base_panel_df().copy()
    df["y"] = df["y"].astype(object)
    df.loc[0, "y"] = "bad"

    with pytest.raises(ValueError, match="contains non-numeric values"):
        PanelDataSCM(**_base_kwargs(df=df, allow_missing_outcome=True))


def test_treated_post_outcomes_must_be_observed():
    df = _base_panel_df().copy()
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == "2020-03-01"), "y"] = np.nan

    with pytest.raises(ValueError, match="treated_unit must have observed y"):
        PanelDataSCM(**_base_kwargs(df=df, allow_missing_outcome=True))


def test_explicit_periods_must_be_valid_and_in_analysis_data():
    with pytest.raises(ValueError, match="pre_periods must contain only periods < treatment_start"):
        PanelDataSCM(**_base_kwargs(pre_periods=["2020-02-01"]))

    with pytest.raises(ValueError, match="post_periods must contain only periods >= treatment_start"):
        PanelDataSCM(**_base_kwargs(post_periods=["2020-01-01"]))

    with pytest.raises(ValueError, match="not present in analysis data"):
        PanelDataSCM(
            **_base_kwargs(
                time_window=("2020-02-01", "2020-03-01"),
                pre_periods=["2020-01-01"],
                post_periods=["2020-02-01"],
            )
        )


def test_gapped_time_axis_can_be_rejected_or_allowed():
    df = _base_panel_df()
    df = df[df["time_id"] != "2020-02-01"].copy()

    with pytest.raises(ValueError, match="Analysis time axis has gaps"):
        PanelDataSCM(**_base_kwargs(df=df, treatment_start="2020-03-01"))

    panel = PanelDataSCM(
        **_base_kwargs(
            df=df,
            treatment_start="2020-03-01",
            allow_gapped_time_axis=True,
        )
    )
    assert panel.allow_gapped_time_axis is True


def test_panel_repr_is_compact_and_informative():
    panel = PanelDataSCM(
        **_base_kwargs(
            covariate_cols=["x1"],
            observed_col="observed",
        )
    )

    repr_str = repr(panel)
    assert repr_str.startswith("PanelDataSCM(df=(9, 5),")
    assert "unit_col='unit_id'" in repr_str
    assert "time_col='time_id'" in repr_str
    assert "time_freq='M'" in repr_str
    assert "y='y'" in repr_str
    assert "treated_unit='T'" in repr_str
    assert "treatment_start=Period('2020-02'" in repr_str
    assert "donor_units=['C1', 'C2']" in repr_str
    assert "covariate_cols=['x1']" in repr_str
    assert "observed_col='observed'" in repr_str
