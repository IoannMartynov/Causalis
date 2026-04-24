import numpy as np
import pandas as pd

from causalis.data_contracts import PanelDataDID
from causalis.dgp import generate_did_data as generate_did_data_top_level
from causalis.dgp.panel_data_did import (
    generate_did_data,
    generate_did_gamma_data,
    generate_did_poisson_data,
)


ORACLE_COLS = {"y_cf", "tau_realized_true", "mu_cf", "mu_treated", "tau_mean_true"}
COVARIATE_COLS = {
    "exposure",
    "avg_order_value",
    "market_competition",
    "macro_index",
    "seasonality_index",
}


def test_generate_did_data_returns_contract_with_oracle_effects():
    panel = generate_did_data(
        n_treated_units=3,
        n_control_units=5,
        n_pre_periods=6,
        n_post_periods=3,
        random_state=7,
        return_panel_data=True,
        include_oracle=True,
    )

    assert isinstance(panel, PanelDataDID)
    assert panel.design_type == "simultaneous_adoption"
    assert panel.n_pre_periods == 6
    assert panel.n_post_periods == 3
    assert len(panel.treated_units) == 3
    assert len(panel.control_units) == 5
    assert panel.covariates == (
        "exposure",
        "avg_order_value",
        "market_competition",
        "macro_index",
        "seasonality_index",
    )
    assert panel.cluster_col == "region"
    assert ORACLE_COLS.issubset(panel.df.columns)
    assert COVARIATE_COLS.issubset(panel.df.columns)
    assert "tau_rate_true" not in panel.df.columns

    treated_post = panel.df[(panel.df["is_treated_unit"] == 1) & (panel.df["treated_time"] == 1)]
    untreated_or_pre = panel.df[(panel.df["is_treated_unit"] == 0) | (panel.df["treated_time"] == 0)]

    assert not treated_post.empty
    assert treated_post["tau_mean_true"].gt(0).all()
    assert np.allclose(treated_post["y"] - treated_post["y_cf"], treated_post["tau_realized_true"])
    assert untreated_or_pre["tau_mean_true"].eq(0.0).all()
    assert untreated_or_pre["tau_realized_true"].eq(0.0).all()

    did_df = panel.df_for_did()
    assert {"treated_group", "post"}.issubset(did_df.columns)
    assert ORACLE_COLS.issubset(did_df.columns)


def test_generate_did_data_raw_dataframe_can_hide_oracles():
    df = generate_did_data(
        n_treated_units=2,
        n_control_units=3,
        n_pre_periods=4,
        n_post_periods=2,
        random_state=11,
        return_panel_data=False,
        include_oracle=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert ORACLE_COLS.isdisjoint(df.columns)
    assert COVARIATE_COLS.issubset(df.columns)
    assert {"unit_id", "calendar_time", "treated_time", "y", "region", "segment"}.issubset(df.columns)
    assert df["treated_time"].isin([0, 1]).all()
    assert isinstance(df["calendar_time"].iloc[0], pd.Period)


def test_positive_did_wrappers_return_realistic_valid_panels():
    gamma_panel = generate_did_gamma_data(
        n_treated_units=2,
        n_control_units=4,
        n_pre_periods=5,
        n_post_periods=3,
        seed=13,
        return_panel_data=True,
    )
    poisson_df = generate_did_poisson_data(
        n_treated_units=2,
        n_control_units=4,
        n_pre_periods=5,
        n_post_periods=3,
        seed=13,
        return_panel_data=False,
    )

    assert isinstance(gamma_panel, PanelDataDID)
    assert gamma_panel.df["y"].ge(0).all()
    assert ORACLE_COLS.issubset(gamma_panel.df.columns)
    assert gamma_panel.cell_counts()["n"].sum() == gamma_panel.df.shape[0]

    assert poisson_df["y"].ge(0).all()
    assert np.allclose(poisson_df["y"], np.round(poisson_df["y"]))
    assert ORACLE_COLS.issubset(poisson_df.columns)


def test_top_level_did_export_matches_package_function():
    panel = generate_did_data_top_level(
        n_treated_units=1,
        n_control_units=2,
        n_pre_periods=3,
        n_post_periods=2,
        random_state=5,
    )

    assert isinstance(panel, PanelDataDID)
    assert panel.df.shape[0] == 15
