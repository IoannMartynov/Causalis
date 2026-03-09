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
        n_donors=3,
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
    n_pre = int(len(panel.pre_times()))
    n_post = int(len(panel.post_times()))
    # Placebo-in-time now requires at least 2 pre periods before placebo start.
    assert len(table) == max(0, n_pre - n_post - 1)

    if not table.empty:
        assert table["n_pre_before_placebo"].min() >= 2
        assert table["n_post_after_placebo"].nunique() == 1
        assert int(table["n_post_after_placebo"].iloc[0]) == n_post
        assert table["rejects_zero"].isin([True, False]).all()
        starts = table["placebo_treatment_start"].tolist()
        assert starts == sorted(starts)
        pre_times = list(panel.pre_times())
        for start in starts:
            start_idx = pre_times.index(start)
            assert start_idx + n_post - 1 < len(pre_times)


def test_placebo_in_space_refits_treated_row_with_model_kwargs():
    estimate, panel = _fit_estimate_and_panel()
    table = placebo_in_space_table(
        estimate,
        panel,
        model_kwargs={"lambda_aug": 10.0},
    )
    treated_row = table.loc[table["is_actual_treated"]].iloc[0]

    direct = ASCM(lambda_aug=10.0).fit(panel).estimate()
    direct_gap = direct.observed_outcome - direct.synthetic_outcome
    direct_pre = direct_gap.loc[list(direct.pre_times)]
    direct_post = direct_gap.loc[list(direct.post_times)]

    assert treated_row["pre_rmse"] == pytest.approx(float((direct_pre.pow(2).mean()) ** 0.5))
    assert treated_row["post_rmse"] == pytest.approx(float((direct_post.pow(2).mean()) ** 0.5))


def test_placebo_in_space_excludes_actual_treated_for_donor_placebo():
    estimate, panel = _fit_estimate_and_panel()
    donor_unit = panel.donor_pool()[0]
    table = placebo_in_space_table(estimate, panel)
    donor_row = table.loc[table["unit_id"] == donor_unit].iloc[0]

    df = panel.df_analysis().copy()
    df = df[df[panel.unit_col] != panel.treated_unit].copy()
    df[panel.treated_time] = (
        (df[panel.unit_col] == donor_unit) & (df[panel.time_col] >= panel.treatment_start)
    ).astype(int)
    manual_panel = PanelDataSCM(
        df=df,
        y=panel.y,
        unit_col=panel.unit_col,
        time_col=panel.time_col,
        treated_time=panel.treated_time,
    )
    manual = ASCM().fit(manual_panel).estimate()
    manual_gap = manual.observed_outcome - manual.synthetic_outcome
    manual_pre = manual_gap.loc[list(manual.pre_times)]
    manual_post = manual_gap.loc[list(manual.post_times)]

    assert donor_row["pre_rmse"] == pytest.approx(float((manual_pre.pow(2).mean()) ** 0.5))
    assert donor_row["post_rmse"] == pytest.approx(float((manual_post.pow(2).mean()) ** 0.5))


def test_placebo_in_space_requires_enough_donors_after_exclusion():
    panel = generate_scm_data(
        n_donors=2,
        n_pre_periods=4,
        n_post_periods=3,
        random_state=21,
    )
    estimate = ASCM().fit(panel).estimate()

    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        placebo_in_space_table(estimate, panel)


def test_placebo_requires_matching_time_partitions():
    estimate, panel = _fit_estimate_and_panel()
    df = panel.df_analysis().copy()
    earliest_pre = list(panel.pre_times())[0]
    df = df[df[panel.time_col] != earliest_pre].copy()

    reduced_panel = PanelDataSCM(
        df=df,
        y=panel.y,
        unit_col=panel.unit_col,
        time_col=panel.time_col,
        treated_time=panel.treated_time,
    )

    with pytest.raises(ValueError, match="estimate.pre_times"):
        placebo_in_space_table(estimate, reduced_panel)


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
