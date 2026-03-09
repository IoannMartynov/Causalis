import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelDataSCM
from causalis.dgp import generate_scm_data
from causalis.dgp.panel_data_scm import generate_scm_gamma_data, generate_scm_poisson_data
from causalis.scenarios.synthetic_control import (
    AugmentedSyntheticControl,
    generate_scm_gamma_26,
    generate_scm_poisson_26,
)


def test_generate_scm_data_returns_panel_contract_by_default():
    panel = generate_scm_data(random_state=123)

    assert isinstance(panel, PanelDataSCM)
    assert panel.treated_unit == "treated"
    assert len(panel.donor_pool()) == 5
    assert len(panel.pre_times()) == 20
    assert len(panel.post_times()) == 10


def test_generate_scm_data_can_return_dataframe():
    df = generate_scm_data(return_panel_data=False, random_state=123)

    assert isinstance(df, pd.DataFrame)
    assert {"unit_id", "calendar_time", "y", "y_cf", "tau_realized_true", "observed"}.issubset(df.columns)


def test_generated_data_is_usable_by_ascm():
    true_effect = 3.5
    panel = generate_scm_data(
        n_donors=6,
        n_pre_periods=24,
        n_post_periods=8,
        treatment_effect=true_effect,
        donor_noise_std=0.10,
        treated_noise_std=0.02,
        random_state=7,
    )

    estimate = AugmentedSyntheticControl(lambda_aug=0.5).fit(panel).estimate()

    assert abs(estimate.att - true_effect) < 1.0
    assert len(estimate.donor_weights_sc) == 6
    assert len(estimate.att_by_time) == 8


def test_generate_scm_data_missingness_mode_returns_valid_contract():
    panel = generate_scm_data(
        n_donors=4,
        missing_outcome_frac=0.05,
        missing_cell_frac=0.10,
        random_state=99,
    )

    assert isinstance(panel, PanelDataSCM)
    assert len(panel.donor_pool()) == 4


def test_generate_scm_data_reality_knobs_with_structured_missingness():
    n_donors = 5
    n_pre = 18
    n_post = 6
    panel = generate_scm_data(
        n_donors=n_donors,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dirichlet_alpha=0.2,
        rho_common=0.6,
        rho_donor=0.4,
        n_latent_factors=2,
        rho_latent=0.5,
        prefit_mismatch_std=0.15,
        rho_prefit_mismatch=0.4,
        missing_block_frac=0.10,
        missing_block_min_len=2,
        missing_block_max_len=4,
        random_state=202,
    )

    assert isinstance(panel, PanelDataSCM)
    assert set(panel.df["observed"].unique()).issubset({0, 1})
    n_full = (n_pre + n_post) * (n_donors + 1)
    assert len(panel.df) == n_full
    assert panel.df["y"].isna().any()
    observed_matches_outcome = (
        ((panel.df["observed"] == 0) & panel.df["y"].isna())
        | ((panel.df["observed"] == 1) & panel.df["y"].notna())
    )
    assert observed_matches_outcome.all()


def test_generate_scm_data_multiplicative_effect_mode_tracks_tau_realized_true():
    n_pre = 12
    n_post = 5
    time_start = 2
    df = generate_scm_data(
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        time_start=time_start,
        treatment_effect=0.08,
        treatment_effect_mode="multiplicative",
        return_panel_data=False,
        random_state=11,
    )

    treated = df[df["unit_id"] == "treated"].copy()
    intervention_time = sorted(pd.Index(df["calendar_time"].unique()).tolist())[n_pre]
    pre = treated[treated["calendar_time"] < intervention_time]
    post = treated[treated["calendar_time"] >= intervention_time]

    assert np.allclose(df["tau_realized_true"].to_numpy(), (df["y"] - df["y_cf"]).to_numpy())
    assert np.allclose(pre["tau_realized_true"].to_numpy(), 0.0)
    assert (post["tau_realized_true"] > 0.0).all()


def test_generate_scm_data_can_protect_treated_post_outcomes():
    n_pre = 10
    n_post = 6
    time_start = 1
    df = generate_scm_data(
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        time_start=time_start,
        missing_outcome_frac=0.30,
        missing_cell_frac=0.20,
        missing_block_frac=0.15,
        protect_treated_post=True,
        return_panel_data=False,
        random_state=101,
    )

    intervention_time = sorted(pd.Index(df["calendar_time"].unique()).tolist())[n_pre]
    treated_post = df[(df["unit_id"] == "treated") & (df["calendar_time"] >= intervention_time)]
    assert not treated_post["y"].isna().any()


def test_generate_scm_gamma_data_emits_mean_oracles():
    df = generate_scm_gamma_data(
        n=360,
        n_donors=7,
        n_pre_periods=16,
        n_post_periods=6,
        seed=19,
        return_panel_data=False,
    )

    assert {"mu_cf", "mu_treated", "tau_mean_true"}.issubset(df.columns)
    assert "tau_rate_true" not in df.columns
    assert np.allclose(df["tau_realized_true"].to_numpy(), (df["y"] - df["y_cf"]).to_numpy())
    assert np.allclose(df["tau_mean_true"].to_numpy(), (df["mu_treated"] - df["mu_cf"]).to_numpy())
    donors = df[df["is_treated_unit"] == 0]
    assert np.allclose(donors["tau_mean_true"].to_numpy(), 0.0)


def test_generate_scm_poisson_data_coupled_outcomes_and_mean_oracles():
    n_pre = 14
    n_post = 6
    time_start = 1
    df = generate_scm_poisson_data(
        n=360,
        n_donors=6,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        treatment_effect_rate=0.18,
        treatment_effect_slope=0.0,
        donor_missing_block_frac=0.0,
        seed=31,
        return_panel_data=False,
    )

    assert {"mu_cf", "mu_treated", "tau_mean_true"}.issubset(df.columns)
    assert "tau_rate_true" not in df.columns
    assert np.allclose(df["tau_realized_true"].to_numpy(), (df["y"] - df["y_cf"]).to_numpy())
    assert np.allclose(df["tau_mean_true"].to_numpy(), (df["mu_treated"] - df["mu_cf"]).to_numpy())

    intervention_time = sorted(pd.Index(df["calendar_time"].unique()).tolist())[n_pre]
    treated = df[df["unit_id"] == "treated"].copy()
    pre = treated[treated["calendar_time"] < intervention_time]
    post = treated[treated["calendar_time"] >= intervention_time]
    assert np.allclose(pre["tau_realized_true"].to_numpy(), 0.0)
    assert (post["tau_realized_true"] >= 0.0).all()


def test_generate_scm_poisson_data_rejects_base_missing_overrides():
    with pytest.raises(ValueError, match="does not allow overriding"):
        generate_scm_poisson_data(
            n=360,
            seed=77,
            n_donors=6,
            return_panel_data=False,
            missing_outcome_frac=0.50,
            missing_cell_frac=0.40,
            missing_block_frac=0.30,
        )


def test_generate_scm_gamma_26_updates_covariate_cols_after_drop():
    panel = generate_scm_gamma_26(return_panel_data=True, include_oracles=False, seed=123)
    assert isinstance(panel, PanelDataSCM)
    assert "treated_time" in panel.df.columns
    assert "treatment_start" not in panel.df.columns
    assert "is_anchor_period" not in panel.df.columns
    assert not {"exposure", "macro_index", "seasonality_index"}.intersection(panel.df.columns)


def test_generate_scm_poisson_26_updates_covariate_cols_after_drop():
    panel = generate_scm_poisson_26(return_panel_data=True, include_oracles=False, seed=123)
    assert isinstance(panel, PanelDataSCM)
    assert "treated_time" in panel.df.columns
    assert "treatment_start" not in panel.df.columns
    assert "is_anchor_period" not in panel.df.columns
    assert not {"exposure", "macro_index", "seasonality_index"}.intersection(panel.df.columns)


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    [
        (generate_scm_gamma_26, {}),
        (generate_scm_poisson_26, {"donor_missing_block_frac": 0.0}),
    ],
)
def test_generate_scm26_dataframe_include_oracles_false_keeps_treated_time(generator, kwargs):
    df = generator(
        return_panel_data=False,
        include_oracles=False,
        seed=123,
        **kwargs,
    )
    assert "treated_time" in df.columns
    assert "is_treated_unit" not in df.columns
    assert "treatment_start" not in df.columns
    assert "is_anchor_period" not in df.columns
    assert set(pd.Index(df["treated_time"].unique()).tolist()).issubset({0, 1})
    assert int(df["treated_time"].sum()) > 0


def test_generate_scm_gamma_data_rejects_conflicting_locked_advanced_params():
    with pytest.raises(ValueError, match="does not allow overriding"):
        generate_scm_gamma_data(seed=42, random_state=999)
    with pytest.raises(ValueError, match="does not allow overriding"):
        generate_scm_gamma_data(outcome_distribution="poisson")


def test_generate_scm_poisson_data_rejects_conflicting_locked_advanced_params():
    with pytest.raises(ValueError, match="does not allow overriding"):
        generate_scm_poisson_data(seed=42, random_state=999)
    with pytest.raises(ValueError, match="does not allow overriding"):
        generate_scm_poisson_data(outcome_distribution="gamma")
    with pytest.raises(ValueError, match="does not allow overriding"):
        generate_scm_poisson_data(missing_outcome_frac=0.5)


def test_generate_scm_gamma_data_first_post_rate_is_ramped():
    n_pre = 12
    df = generate_scm_gamma_data(
        n_pre_periods=n_pre,
        n_post_periods=5,
        treatment_effect_rate=0.12,
        treatment_effect_slope=0.0,
        seed=321,
        return_panel_data=False,
    )
    treated = df[df["unit_id"] == "treated"].sort_values("calendar_time")
    first_post_time = sorted(pd.Index(treated["calendar_time"].unique()).tolist())[n_pre]
    first_post = treated[treated["calendar_time"] == first_post_time].iloc[0]
    observed_rate = float(first_post["tau_mean_true"] / first_post["mu_cf"])
    expected_rate = 0.12 * (1.0 - np.exp(-1.0 / 2.5))
    assert np.isclose(observed_rate, expected_rate)


def test_generate_scm_poisson_data_first_post_rate_is_ramped():
    n_pre = 10
    df = generate_scm_poisson_data(
        n_pre_periods=n_pre,
        n_post_periods=4,
        treatment_effect_rate=0.10,
        treatment_effect_slope=0.0,
        donor_missing_block_frac=0.0,
        seed=456,
        return_panel_data=False,
    )
    treated = df[df["unit_id"] == "treated"].sort_values("calendar_time")
    first_post_time = sorted(pd.Index(treated["calendar_time"].unique()).tolist())[n_pre]
    first_post = treated[treated["calendar_time"] == first_post_time].iloc[0]
    observed_rate = float(first_post["tau_mean_true"] / first_post["mu_cf"])
    expected_rate = 0.10 * (1.0 - np.exp(-1.0 / 2.5))
    assert np.isclose(observed_rate, expected_rate)


def test_generate_scm_gamma_data_exposure_varies_over_time():
    df = generate_scm_gamma_data(
        n_pre_periods=16,
        n_post_periods=6,
        n_donors=6,
        seed=91,
        return_panel_data=False,
    )
    donor = df[df["unit_id"] == "donor_1"].sort_values("calendar_time")
    treated = df[df["unit_id"] == "treated"].sort_values("calendar_time")
    assert donor["exposure"].nunique() > 1
    assert treated["exposure"].nunique() > 1


def test_generate_scm_poisson_data_exposure_varies_over_time():
    df = generate_scm_poisson_data(
        n_pre_periods=16,
        n_post_periods=6,
        n_donors=6,
        donor_missing_block_frac=0.0,
        seed=92,
        return_panel_data=False,
    )
    donor = df[df["unit_id"] == "donor_1"].sort_values("calendar_time")
    treated = df[df["unit_id"] == "treated"].sort_values("calendar_time")
    assert donor["exposure"].nunique() > 1
    assert treated["exposure"].nunique() > 1


def test_generate_scm_gamma_data_rejects_partial_pre_post_spec():
    with pytest.raises(ValueError, match="Provide both n_pre_periods and n_post_periods"):
        generate_scm_gamma_data(n_pre_periods=12)
    with pytest.raises(ValueError, match="Provide both n_pre_periods and n_post_periods"):
        generate_scm_gamma_data(n_post_periods=6)


def test_generate_scm_poisson_data_rejects_partial_pre_post_spec():
    with pytest.raises(ValueError, match="Provide both n_pre_periods and n_post_periods"):
        generate_scm_poisson_data(n_pre_periods=12)
    with pytest.raises(ValueError, match="Provide both n_pre_periods and n_post_periods"):
        generate_scm_poisson_data(n_post_periods=6)


def test_generate_scm_gamma_26_accepts_explicit_pre_post_periods():
    n_pre = 15
    n_post = 7
    panel = generate_scm_gamma_26(
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=True,
        seed=42,
    )
    assert len(panel.pre_times()) == n_pre + 1  # includes anchor
    assert len(panel.post_times()) == n_post


def test_generate_scm_poisson_26_accepts_explicit_pre_post_periods():
    n_pre = 14
    n_post = 5
    panel = generate_scm_poisson_26(
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=True,
        donor_missing_block_frac=0.0,
        seed=42,
    )
    assert len(panel.pre_times()) == n_pre + 1  # includes anchor
    assert len(panel.post_times()) == n_post


def test_generate_scm_gamma_26_uses_default_pre_post():
    panel = generate_scm_gamma_26(return_panel_data=True, include_oracles=True, seed=42)
    assert len(panel.pre_times()) == 37  # default 36 + 1 anchor
    assert len(panel.post_times()) == 12


def test_generate_scm_poisson_26_uses_default_pre_post():
    panel = generate_scm_poisson_26(
        return_panel_data=True,
        include_oracles=True,
        donor_missing_block_frac=0.0,
        seed=42,
    )
    assert len(panel.pre_times()) == 37  # default 36 + 1 anchor
    assert len(panel.post_times()) == 12


def test_generate_scm_gamma_26_dataframe_marks_treated_time():
    n_pre = 6
    n_post = 3
    df = generate_scm_gamma_26(
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=False,
        seed=404,
    )

    assert "treated_time" in df.columns
    assert "observed" not in df.columns
    assert "treatment_start" not in df.columns
    assert "is_anchor_period" not in df.columns

    treated_mask = df["unit_id"] == "treated"
    time_values = sorted(pd.Index(df["calendar_time"].unique()).tolist())
    treated_start = time_values[n_pre + 1]
    expected = (treated_mask & (df["calendar_time"] >= treated_start)).astype(int)
    assert np.array_equal(df["treated_time"].to_numpy(), expected.to_numpy())
    assert int(df.loc[treated_mask, "treated_time"].sum()) == n_post
    assert int(df.loc[~treated_mask, "treated_time"].sum()) == 0


def test_generate_scm_poisson_26_panel_marks_treated_time_and_excludes_anchor_from_windows():
    n_pre = 7
    n_post = 4
    panel = generate_scm_poisson_26(
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=True,
        donor_missing_block_frac=0.0,
        seed=505,
    )

    assert "treated_time" in panel.df.columns
    assert "treatment_start" not in panel.df.columns
    assert "is_anchor_period" not in panel.df.columns

    time_values = sorted(pd.Index(panel.df[panel.time_col].unique()).tolist())
    anchor = time_values[n_pre]
    treated_start = time_values[n_pre + 1]
    treated_mask = panel.df[panel.unit_col] == panel.treated_unit
    expected = (treated_mask & (panel.df[panel.time_col] >= treated_start)).astype(int)
    assert np.array_equal(panel.df["treated_time"].to_numpy(), expected.to_numpy())
    assert int(panel.df.loc[~treated_mask, "treated_time"].sum()) == 0

    assert anchor == panel.treatment_start - 1
    assert anchor in panel.pre_times()
    assert anchor not in panel.post_times()


def test_generate_scm_gamma_26_rows_per_unit_include_intervention_anchor():
    n_pre = 5
    n_post = 3
    n_donors = 2
    expected_rows_per_unit = n_pre + 1 + n_post
    df = generate_scm_gamma_26(
        n_donors=n_donors,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=False,
        seed=123,
    )
    donor = df[df["unit_id"] == "donor_1"]
    assert len(donor) == expected_rows_per_unit

    panel = generate_scm_gamma_26(
        n_donors=n_donors,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=True,
        seed=123,
    )
    assert len(panel.pre_times()) == n_pre + 1  # includes anchor
    assert len(panel.post_times()) == n_post


def test_generate_scm_poisson_26_rows_per_unit_include_intervention_anchor():
    n_pre = 6
    n_post = 2
    n_donors = 2
    expected_rows_per_unit = n_pre + 1 + n_post
    df = generate_scm_poisson_26(
        n_donors=n_donors,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=False,
        donor_missing_block_frac=0.0,
        seed=123,
    )
    donor = df[df["unit_id"] == "donor_1"]
    assert len(donor) == expected_rows_per_unit

    panel = generate_scm_poisson_26(
        n_donors=n_donors,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        include_oracles=True,
        return_panel_data=True,
        donor_missing_block_frac=0.0,
        seed=123,
    )
    assert len(panel.pre_times()) == n_pre + 1  # includes anchor
    assert len(panel.post_times()) == n_post
