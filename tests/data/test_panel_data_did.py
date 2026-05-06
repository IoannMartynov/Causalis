from typing import Optional

import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelDataDID


def _base_panel_df() -> pd.DataFrame:
    rows = []
    periods = pd.period_range("2020-01", periods=3, freq="M")
    for unit in ("T1", "T2", "C1", "C2"):
        for idx, period in enumerate(periods):
            is_treated_unit = unit.startswith("T")
            rows.append(
                {
                    "unit_id": unit,
                    "time_id": period,
                    "y": float(idx + (2.0 if is_treated_unit else 1.0)),
                    "treated_time": int(
                        is_treated_unit and period >= pd.Period("2020-02", freq="M")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _canonical_2x2_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["T1", "T1", "C1", "C1"],
            "time_id": pd.PeriodIndex(
                ["2020-01", "2020-02", "2020-01", "2020-02"], freq="M"
            ),
            "y": [10.0, 12.0, 8.0, 9.0],
            "treated_time": [0, 1, 0, 0],
        }
    )


def _panel_df_with_covariates_cluster() -> pd.DataFrame:
    df = _base_panel_df()
    df["x_num"] = np.arange(len(df), dtype=float)
    df["x_bool"] = df["unit_id"].isin(["T1", "C1"])
    df["cluster_id"] = df["unit_id"].str[0]
    return df


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


def test_simultaneous_adoption_contract_and_helpers_work():
    panel = PanelDataDID(**_base_kwargs())

    assert panel.df.columns.tolist() == ["unit_id", "time_id", "treated_time", "y"]
    assert list(panel.treated_units) == ["T1", "T2"]
    assert list(panel.control_units) == ["C1", "C2"]
    assert list(panel.never_treated_units) == ["C1", "C2"]
    assert [str(c) for c in panel.cohorts] == ["2020-02"]
    assert panel.first_treatment_by_unit["T1"] == pd.Period("2020-02", freq="M")
    assert panel.first_treatment_by_unit["C1"] is None
    assert list(panel.cohort_units("2020-02")) == ["T1", "T2"]
    assert [str(t) for t in panel.pre_times()] == ["2020-01"]
    assert [str(t) for t in panel.post_times()] == ["2020-02", "2020-03"]
    assert panel.treatment_start_idx() == 1
    assert panel.n_pre_periods == 1
    assert panel.n_post_periods == 2
    assert panel.design_type == "simultaneous_adoption"

    did_df = panel.df_for_did()
    assert {"treated_group", "post", "cohort", "event_time"}.issubset(did_df.columns)
    assert did_df.loc[did_df["unit_id"].isin(["T1", "T2"]), "treated_group"].eq(1).all()
    assert did_df.loc[did_df["unit_id"].isin(["C1", "C2"]), "treated_group"].eq(0).all()
    assert (
        did_df.loc[
            did_df["unit_id"].isin(["T1", "T2"])
            & (did_df["time_id"] < panel.treatment_start),
            "post",
        ]
        .eq(0)
        .all()
    )
    assert (
        did_df.loc[
            did_df["unit_id"].isin(["T1", "T2"])
            & (did_df["time_id"] >= panel.treatment_start),
            "post",
        ]
        .eq(1)
        .all()
    )
    assert did_df.loc[did_df["unit_id"].isin(["C1", "C2"]), "post"].eq(0).all()
    assert (
        did_df.loc[
            did_df["unit_id"].isin(["T1", "T2"])
            & (did_df["time_id"] == panel.treatment_start),
            "event_time",
        ]
        .eq(0)
        .all()
    )
    assert did_df.loc[did_df["unit_id"].isin(["C1", "C2"]), "event_time"].isna().all()

    cells = panel.att_gt_cells()
    assert cells["is_supported"].all()
    assert (
        cells[
            ["cohort", "time", "base_time", "event_time", "n_treated", "n_control"]
        ].shape[0]
        == 2
    )


def test_canonical_2x2_design_is_identified():
    panel = PanelDataDID(**_base_kwargs(df=_canonical_2x2_df()))

    assert panel.design_type == "canonical_2x2"
    assert panel.n_pre_periods == 1
    assert panel.n_post_periods == 1

    counts = panel.cell_counts()
    assert set(counts["group"]) == {"ever_treated", "never_treated"}
    assert counts["n"].sum() == 4


def test_covariates_and_cluster_are_validated_and_projected():
    df = _panel_df_with_covariates_cluster()

    panel = PanelDataDID(
        **_base_kwargs(
            df=df, covariates=["x_num", "x_bool", "x_num"], cluster_col="cluster_id"
        )
    )

    assert panel.covariates == ("x_num", "x_bool")
    assert panel.cluster_col == "cluster_id"
    assert panel.has_covariates is True
    assert panel.has_cluster is True
    assert panel.df.columns.tolist() == [
        "unit_id",
        "time_id",
        "treated_time",
        "y",
        "x_num",
        "x_bool",
        "cluster_id",
    ]
    assert panel.covariate_frame().columns.tolist() == ["x_num", "x_bool"]
    assert panel.cluster_series().equals(panel.df["cluster_id"])

    did_df = panel.df_for_did(treated_group_col="did_treated", post_col="did_post")
    assert {"x_num", "x_bool", "cluster_id", "did_treated", "did_post"}.issubset(
        did_df.columns
    )


def test_covariants_and_cluster_aliases_are_accepted():
    df = _panel_df_with_covariates_cluster()

    panel = PanelDataDID(
        **_base_kwargs(df=df, covariants="x_num", cluster="cluster_id")
    )

    assert panel.covariates == ("x_num",)
    assert panel.cluster_col == "cluster_id"


def test_cluster_can_reuse_unit_or_time_column():
    panel_unit = PanelDataDID(**_base_kwargs(cluster_col="unit_id"))
    assert panel_unit.cluster_col == "unit_id"
    assert panel_unit.df.columns.tolist() == ["unit_id", "time_id", "treated_time", "y"]

    panel_time = PanelDataDID(**_base_kwargs(cluster_col="time_id"))
    assert panel_time.cluster_col == "time_id"


def test_missing_required_columns_raise():
    df = _base_panel_df().drop(columns=["y"])

    with pytest.raises(ValueError, match="Missing required columns"):
        PanelDataDID(**_base_kwargs(df=df))


def test_column_role_names_must_be_distinct():
    with pytest.raises(ValueError, match="Column role names must be distinct"):
        PanelDataDID(**_base_kwargs(y="unit_id"))


def test_covariates_and_cluster_must_be_distinct_from_disallowed_roles():
    df = _panel_df_with_covariates_cluster()

    with pytest.raises(ValueError, match="covariates must be distinct"):
        PanelDataDID(**_base_kwargs(df=df, covariates=["y"]))

    with pytest.raises(
        ValueError, match="cluster_col must be distinct from y and treated_time"
    ):
        PanelDataDID(**_base_kwargs(df=df, cluster_col="treated_time"))

    with pytest.raises(
        ValueError, match="cluster_col must be distinct from covariates"
    ):
        PanelDataDID(**_base_kwargs(df=df, covariates=["x_num"], cluster_col="x_num"))


def test_unit_time_outcome_and_treatment_must_be_non_null():
    df_unit_null = _base_panel_df()
    df_unit_null.loc[0, "unit_id"] = np.nan
    with pytest.raises(ValueError, match="unit_id' contains nulls"):
        PanelDataDID(**_base_kwargs(df=df_unit_null))

    df_time_null = _base_panel_df()
    df_time_null.loc[0, "time_id"] = np.nan
    with pytest.raises(ValueError, match="time_id' contains nulls"):
        PanelDataDID(**_base_kwargs(df=df_time_null))

    df_outcome_null = _base_panel_df()
    df_outcome_null.loc[0, "y"] = np.nan
    with pytest.raises(ValueError, match="contains nulls"):
        PanelDataDID(**_base_kwargs(df=df_outcome_null))

    df_treatment_null = _base_panel_df()
    df_treatment_null.loc[0, "treated_time"] = np.nan
    with pytest.raises(ValueError, match="treated_time' contains nulls"):
        PanelDataDID(**_base_kwargs(df=df_treatment_null))


def test_covariates_must_exist_be_numeric_non_null_and_non_constant():
    df_missing = _base_panel_df()
    with pytest.raises(ValueError, match="Missing required columns"):
        PanelDataDID(**_base_kwargs(df=df_missing, covariates=["x_missing"]))

    df_null = _panel_df_with_covariates_cluster()
    df_null.loc[0, "x_num"] = np.nan
    with pytest.raises(ValueError, match="Covariate 'x_num' contains nulls"):
        PanelDataDID(**_base_kwargs(df=df_null, covariates=["x_num"]))

    df_non_numeric = _panel_df_with_covariates_cluster()
    df_non_numeric["x_num"] = df_non_numeric["x_num"].astype(object)
    df_non_numeric.loc[0, "x_num"] = "bad"
    with pytest.raises(
        ValueError, match="Covariate 'x_num' contains non-numeric values"
    ):
        PanelDataDID(**_base_kwargs(df=df_non_numeric, covariates=["x_num"]))

    df_constant = _panel_df_with_covariates_cluster()
    df_constant["x_num"] = 1.0
    with pytest.raises(ValueError, match="Covariate 'x_num' is constant"):
        PanelDataDID(**_base_kwargs(df=df_constant, covariates=["x_num"]))


def test_cluster_must_exist_and_be_non_null_when_provided():
    df_missing = _base_panel_df()
    with pytest.raises(ValueError, match="Missing required columns"):
        PanelDataDID(**_base_kwargs(df=df_missing, cluster_col="cluster_id"))

    df_null = _panel_df_with_covariates_cluster()
    df_null.loc[0, "cluster_id"] = np.nan
    with pytest.raises(ValueError, match="cluster_id' contains nulls"):
        PanelDataDID(**_base_kwargs(df=df_null, cluster_col="cluster_id"))


def test_duplicate_unit_time_rejected():
    df = _base_panel_df()
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=r"Duplicate \(unit,time\) rows"):
        PanelDataDID(**_base_kwargs(df=df_dup))


def test_treated_time_must_be_binary():
    df = _base_panel_df()
    df.loc[0, "treated_time"] = 2

    with pytest.raises(ValueError, match="must be boolean or 0/1"):
        PanelDataDID(**_base_kwargs(df=df))


def test_requires_at_least_one_treated_row_and_one_supported_comparison_cell():
    df_no_treated = _base_panel_df()
    df_no_treated["treated_time"] = 0
    with pytest.raises(ValueError, match="must have at least one treated row"):
        PanelDataDID(**_base_kwargs(df=df_no_treated))

    df_no_control = _base_panel_df()
    df_no_control = df_no_control[df_no_control["unit_id"].isin(["T1", "T2"])].copy()
    with pytest.raises(
        ValueError, match=r"No supported Callaway-Sant'Anna ATT\(g,t\) cells"
    ):
        PanelDataDID(**_base_kwargs(df=df_no_control))


def test_staggered_adoption_contract_and_cohort_helpers_work():
    df = _base_panel_df()
    df.loc[
        (df["unit_id"] == "T2") & (df["time_id"] == pd.Period("2020-02", freq="M")),
        "treated_time",
    ] = 0

    panel = PanelDataDID(**_base_kwargs(df=df))

    assert panel.design_type == "staggered_adoption"
    assert [str(c) for c in panel.cohorts] == ["2020-02", "2020-03"]
    assert list(panel.cohort_units("2020-02")) == ["T1"]
    assert list(panel.cohort_units("2020-03")) == ["T2"]
    assert [str(t) for t in panel.pre_times("2020-03")] == ["2020-01", "2020-02"]
    assert [str(t) for t in panel.post_times("2020-03")] == ["2020-03"]
    assert list(panel.comparison_units("2020-02", "2020-02")) == ["T2", "C1", "C2"]
    assert list(panel.comparison_units("2020-03", "2020-03")) == ["C1", "C2"]

    did_df = panel.df_for_did()
    t2 = did_df[did_df["unit_id"] == "T2"].sort_values("time_id")
    assert t2["post"].tolist() == [0, 0, 1]
    assert t2["event_time"].tolist() == [-2, -1, 0]

    cells = panel.att_gt_cells()
    assert set(cells["event_time"]) == {0, 1}
    assert cells["n_control"].min() >= 2


def test_treated_units_must_stay_treated_after_start():
    df = _base_panel_df()
    df.loc[
        (df["unit_id"] == "T1") & (df["time_id"] == pd.Period("2020-03", freq="M")),
        "treated_time",
    ] = 0

    with pytest.raises(
        ValueError,
        match="ever-treated units must be 1 at/after their first treatment period",
    ):
        PanelDataDID(**_base_kwargs(df=df))


def test_supported_att_gt_cells_are_required():
    df = _base_panel_df()
    df = df[
        ~((df["unit_id"] == "C1") & (df["time_id"] == pd.Period("2020-02", freq="M")))
    ].copy()
    df = df[
        ~((df["unit_id"] == "C2") & (df["time_id"] == pd.Period("2020-02", freq="M")))
    ].copy()
    df = df[
        ~((df["unit_id"] == "C1") & (df["time_id"] == pd.Period("2020-03", freq="M")))
    ].copy()
    df = df[
        ~((df["unit_id"] == "C2") & (df["time_id"] == pd.Period("2020-03", freq="M")))
    ].copy()

    with pytest.raises(
        ValueError, match=r"No supported Callaway-Sant'Anna ATT\(g,t\) cells"
    ):
        PanelDataDID(**_base_kwargs(df=df))


def test_gapped_time_axis_is_rejected():
    df = _base_panel_df()
    df = df[df["time_id"] != pd.Period("2020-02", freq="M")].copy()

    with pytest.raises(ValueError, match="Analysis time axis has gaps"):
        PanelDataDID(**_base_kwargs(df=df))


def test_outcome_must_be_numeric():
    df = _base_panel_df()
    df["y"] = df["y"].astype(object)
    df.loc[0, "y"] = "bad"

    with pytest.raises(ValueError, match="contains non-numeric values"):
        PanelDataDID(**_base_kwargs(df=df))


def test_outcome_must_be_finite():
    df = _base_panel_df()
    df.loc[0, "y"] = np.inf

    with pytest.raises(ValueError, match="must contain only finite numeric values"):
        PanelDataDID(**_base_kwargs(df=df))


def test_two_point_datetime_input_requires_explicit_periods():
    df = _canonical_2x2_df()
    df["time_id"] = df["time_id"].astype(str)

    with pytest.raises(ValueError, match="fewer than 3 unique datetime values"):
        PanelDataDID(**_base_kwargs(df=df))


def test_panel_repr_is_compact_and_informative():
    panel = PanelDataDID(**_base_kwargs())

    repr_str = repr(panel)
    assert repr_str.startswith("PanelDataDID(df=(12, 4),")
    assert "unit_col='unit_id'" in repr_str
    assert "time_col='time_id'" in repr_str
    assert "treated_time='treated_time'" in repr_str
    assert "covariates=[]" in repr_str
    assert "cluster_col=None" in repr_str
    assert "time_freq='M'" in repr_str
    assert "design_type='simultaneous_adoption'" in repr_str
    assert "y='y'" in repr_str
    assert "cohorts=[Period('2020-02'" in repr_str
    assert "treatment_start=Period('2020-02'" in repr_str
    assert "latest_treatment_start=Period('2020-02'" in repr_str
    assert "last_post_period=Period('2020-03'" in repr_str
    assert "n_pre_periods=1" in repr_str
    assert "n_post_periods=2" in repr_str
    assert "treated_units=['T1', 'T2']" in repr_str
    assert "never_treated_units=['C1', 'C2']" in repr_str
