import pandas as pd
import pytest
from pydantic import ValidationError

from causalis.data_contracts import IVCausalData


def test_iv_causal_data_basic_contract():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "d": [0, 1, 0, 1],
            "z": [1, 1, 0, 0],
            "x": [0.1, 0.2, 0.3, 0.4],
            "unused": [None, None, None, None],
        }
    )

    data = IVCausalData.from_df(
        df, treatment="d", outcome="y", instruments="z", confounders=["x"]
    )

    assert data.instruments == ["z"]
    assert list(data.Z.columns) == ["z"]
    assert list(data.df.columns) == ["y", "d", "z", "x"]
    assert list(data.get_df().columns) == ["y", "x", "d", "z"]


def test_iv_causal_data_rejects_multiple_instruments():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "d": [0, 1, 0, 1],
            "z1": [1, 1, 0, 0],
            "z2": [0, 1, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="exactly one instrument"):
        IVCausalData(df=df, treatment="d", outcome="y", instruments=["z1", "z2"])


def test_iv_causal_data_requires_instruments():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "d": [0, 1, 0],
        }
    )

    with pytest.raises(ValidationError, match="instruments"):
        IVCausalData(df=df, treatment="d", outcome="y")

    with pytest.raises(
        TypeError, match="instruments must be a string or a list of strings"
    ):
        IVCausalData.from_df(df, treatment="d", outcome="y", instruments=None)


def test_iv_causal_data_rejects_overlapping_instrument_roles():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "d": [0, 1, 0],
            "z": [1, 1, 0],
            "x": [0.1, 0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="cannot be both treatment and instrument"):
        IVCausalData(df=df, treatment="d", outcome="y", instruments="d")

    with pytest.raises(ValueError, match="confounder columns must be disjoint"):
        IVCausalData(
            df=df, treatment="d", outcome="y", instruments="z", confounders=["z"]
        )


def test_iv_causal_data_rejects_invalid_instruments():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "d": [0, 1, 0],
            "z_constant": [1, 1, 1],
            "z_text": ["a", "b", "c"],
            "z_same_as_d": [0.0, 1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="specified as instrument is constant"):
        IVCausalData(df=df, treatment="d", outcome="y", instruments="z_constant")

    with pytest.raises(
        ValueError,
        match="specified as instruments must contain only int, float, or bool",
    ):
        IVCausalData(df=df, treatment="d", outcome="y", instruments="z_text")

    with pytest.raises(ValueError, match="have identical values"):
        IVCausalData(df=df, treatment="d", outcome="y", instruments="z_same_as_d")


def test_iv_causal_data_requires_binary_treatment_and_instrument():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "d": [0, 1, 2, 1],
            "z": [1, 1, 0, 0],
            "z_nonbinary": [0, 1, 2, 0],
        }
    )

    with pytest.raises(ValueError, match="treatment must be binary encoded"):
        IVCausalData(df=df, treatment="d", outcome="y", instruments="z")

    df_binary_d = df.copy()
    df_binary_d["d"] = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="instrument must be binary encoded"):
        IVCausalData(
            df=df_binary_d,
            treatment="d",
            outcome="y",
            instruments="z_nonbinary",
        )
