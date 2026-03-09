import pandas as pd
import pytest

from causalis.data_contracts import PanelDataSCM
from causalis.dgp import generate_scm_data
from causalis.scenarios.synthetic_control import (
    ASCM,
    placebo_in_space_table,
    placebo_in_time_table,
    run_placebo_tests,
)


def _fit_estimate_and_panel():
    panel = generate_scm_data(
        n_donors=2,
        n_pre_periods=4,
        n_post_periods=3,
        random_state=21,
    )
    estimate = ASCM().fit(panel).estimate()
    return estimate, panel


def test_placebo_in_space_table_schema_and_rows():
    estimate, panel = _fit_estimate_and_panel()
    table = placebo_in_space_table(estimate, panel)

    expected_cols = [
        "unit_id",
        "pre_rmse",
        "post_rmse",
        "post_pre_rmspe_ratio",
        "average_post_gap",
        "max_abs_post_gap",
        "rank_post_pre_rmspe_ratio",
        "is_actual_treated",
    ]
    assert list(table.columns) == expected_cols
    assert len(table) == int(1 + len(panel.donor_pool()))
    assert table["rank_post_pre_rmspe_ratio"].between(1, len(table)).all()

    actual = table.loc[table["is_actual_treated"]]
    assert len(actual) == 1
    assert actual["unit_id"].iloc[0] == panel.treated_unit


def test_placebo_in_time_table_schema_and_rows():
    estimate, panel = _fit_estimate_and_panel()
    table = placebo_in_time_table(estimate, panel)

    expected_cols = [
        "placebo_treatment_start",
        "n_pre_before_placebo",
        "n_post_after_placebo",
        "average_att_placebo",
        "ci_lower",
        "ci_upper",
        "p_value",
        "rejects_zero",
        "pre_fit_metric",
    ]
    assert list(table.columns) == expected_cols
    assert len(table) == max(0, int(len(panel.pre_times()) - 1))

    if not table.empty:
        assert table["n_pre_before_placebo"].min() >= 1
        assert table["n_post_after_placebo"].min() >= 1
        assert table["rejects_zero"].isin([True, False]).all()
        starts = table["placebo_treatment_start"].tolist()
        assert starts == sorted(starts)


def test_run_placebo_tests_returns_both_tables():
    estimate, panel = _fit_estimate_and_panel()
    out = run_placebo_tests(estimate, panel)

    assert set(out.keys()) == {"placebo_in_space", "placebo_in_time"}
    assert isinstance(out["placebo_in_space"], pd.DataFrame)
    assert isinstance(out["placebo_in_time"], pd.DataFrame)


def test_run_placebo_tests_requires_matching_treatment_start():
    estimate, panel = _fit_estimate_and_panel()
    df = panel.df_analysis().copy()
    treated = panel.treated_unit
    placebo_start = list(panel.pre_times())[1]
    df[panel.treated_time] = (
        (df[panel.unit_col] == treated) & (df[panel.time_col] >= placebo_start)
    ).astype(int)

    mismatched_panel = PanelDataSCM(
        df=df,
        y=panel.y,
        unit_col=panel.unit_col,
        time_col=panel.time_col,
        treated_time=panel.treated_time,
    )

    with pytest.raises(ValueError, match="estimate.treatment_start"):
        run_placebo_tests(estimate, mismatched_panel)
