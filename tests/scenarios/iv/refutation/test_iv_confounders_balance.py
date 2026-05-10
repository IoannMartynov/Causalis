import pandas as pd
import pytest

from causalis.data_contracts.iv_causal_data import IVCausalData
from causalis.scenarios.iv.refutation import iv_confounders_balance


def test_iv_confounders_balance_compares_confounders_by_instrument():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "d": [0, 1, 0, 1],
            "z": [0, 0, 1, 1],
            "x1": [1.0, 3.0, 5.0, 7.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
        }
    )
    data = IVCausalData.from_df(
        df,
        treatment="d",
        outcome="y",
        instruments="z",
        confounders=["x1", "x2"],
    )

    balance = iv_confounders_balance(data)

    assert set(balance.columns) == {
        "confounders",
        "mean_z_0",
        "mean_z_1",
        "abs_diff",
        "smd",
        "ks_pvalue",
    }

    row_x1 = balance.loc[balance["confounders"] == "x1"].iloc[0]
    assert row_x1["mean_z_0"] == pytest.approx(2.0)
    assert row_x1["mean_z_1"] == pytest.approx(6.0)
    assert row_x1["abs_diff"] == pytest.approx(4.0)

    row_x2 = balance.loc[balance["confounders"] == "x2"].iloc[0]
    assert row_x2["mean_z_0"] == pytest.approx(15.0)
    assert row_x2["mean_z_1"] == pytest.approx(35.0)


def test_iv_confounders_balance_requires_iv_causal_data():
    with pytest.raises(TypeError, match="IVCausalData"):
        iv_confounders_balance(object())
