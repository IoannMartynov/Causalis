import numpy as np

from causalis.data_contracts import IVCausalData
from causalis.scenarios.iv import generate_offer_iv_26


def test_generate_offer_iv_26_raw_has_realistic_columns_and_positive_effect():
    df = generate_offer_iv_26(n=2_500, seed=123, return_causal_data=False)

    expected = {
        "user_id",
        "net_revenue_90d",
        "accepted_offer",
        "offer_eligible",
        "age",
        "annual_income",
        "credit_score",
        "prior_spend_30d",
        "m",
        "iv_first_stage",
        "iv_reduced_form",
        "late",
        "cate",
    }
    assert expected.issubset(df.columns)
    assert df["user_id"].is_unique
    assert df["accepted_offer"].isin([0.0, 1.0]).all()
    assert df["offer_eligible"].isin([0.0, 1.0]).all()
    assert 0.35 < df["offer_eligible"].mean() < 0.57
    assert 0.25 < df["accepted_offer"].mean() < 0.45
    assert df["iv_first_stage"].mean() > 0.15
    assert df["iv_reduced_form"].mean() > 2.0
    assert np.isfinite(df["late"]).all()
    assert 8.0 < df["late"].iloc[0] < 32.0
    assert df["cate"].mean() > 10.0


def test_generate_offer_iv_26_returns_iv_contract():
    data = generate_offer_iv_26(n=1_000, seed=321, return_causal_data=True)

    assert isinstance(data, IVCausalData)
    assert data.outcome_name == "net_revenue_90d"
    assert data.treatment_name == "accepted_offer"
    assert data.instruments == ["offer_eligible"]
    assert "annual_income" in data.confounders
    assert "credit_score" in data.confounders
    assert list(data.Z.columns) == ["offer_eligible"]
    assert data.df.shape[0] == 1_000
