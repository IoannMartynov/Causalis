import numpy as np
import pandas as pd
import pytest


from causalis.dgp.causaldata import CausalData


def test_causaldata_rejects_overlapping_roles():
    df = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0],
            "d": [0, 1, 0],
            "x1": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(Exception, match="cannot be both"):
        CausalData(df=df, treatment="y", outcome="y", confounders=["x1"])

    with pytest.raises(Exception, match="disjoint"):
        CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "d"])


def test_causaldata_nan_validation_only_applies_to_used_columns():
    df = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0],
            "d": [0, 1, 0, 1],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "unused": [np.nan, 0.0, 0.0, 0.0],
        }
    )

    # NaN in an unused column should not block construction
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])
    assert list(cd.df.columns) == ["y", "d", "x1"]

    # But NaN in a used column should raise
    df_bad = df.copy()
    df_bad.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        CausalData(df=df_bad, treatment="d", outcome="y", confounders=["x1"])


def test_causaldata_x_property_exposes_confounder_matrix():
    df = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0],
            "d": [0, 1, 0, 1],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 11.0, 12.0, 13.0],
        }
    )

    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])
    X = cd.X
    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == ["x1", "x2"]
    assert np.allclose(X.to_numpy(dtype=float), df[["x1", "x2"]].to_numpy(dtype=float))


def test_causaldata_requires_binary_treatment_and_casts_to_int8():
    df = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0],
            "d": [0.0, 1.0, 0.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])
    assert str(cd.df["d"].dtype) == "int8"
    assert cd.df["d"].tolist() == [0, 1, 0, 1]

    df_bad = df.copy()
    df_bad["d"] = [0.0, 1.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="binary encoded"):
        CausalData(df=df_bad, treatment="d", outcome="y", confounders=["x1"])
