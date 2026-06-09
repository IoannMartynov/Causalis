import pytest
import pandas as pd
import numpy as np
from causalis.data_contracts.causaldata import CausalData
from causalis.data_contracts.multicausaldata import MultiCausalData
from pydantic import ValidationError

def test_causal_data_extra_forbid():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d": [0, 1, 0],
        "x": [5, 6, 7]
    })
    # Should work
    CausalData(df=df, outcome="y", treatment="d", confounders=["x"])
    
    # Should fail due to extra="forbid"
    with pytest.raises(ValidationError):
        CausalData(df=df, outcome="y", treatment="d", confounders=["x"], extra_arg="oops")

def test_causal_data_dtype_agnostic_duplicates():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "d": [1, 0, 1],
        "x": [1.0, 0.0, 1.0]  # Same values as t, but different dtype (float vs int)
    })
    # This should now raise ValueError because t and x have identical values
    with pytest.raises(ValueError, match="have identical values"):
        CausalData(df=df, outcome="y", treatment="d", confounders=["x"])


def test_multi_causal_data_dtype_agnostic_duplicates_with_many_columns():
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(6)],
        "y": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "d0": [1, 0, 0, 1, 0, 0],
        "d1": [0, 1, 0, 0, 1, 0],
        "d2": [0, 0, 1, 0, 0, 1],
        "x0": [0, 1, 2, 3, 4, 5],
        "x1": [2, 3, 5, 7, 11, 13],
        "x2": [1, 4, 9, 16, 25, 36],
        "x3": [10, 20, 30, 40, 50, 60],
        "x_dup": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })

    with pytest.raises(ValueError, match="Columns 'x0' and 'x_dup' have identical values"):
        MultiCausalData(
            df=df,
            outcome="y",
            treatment_names=["d0", "d1", "d2"],
            confounders=["x0", "x1", "x2", "x3", "x_dup"],
            user_id="user_id",
            control_treatment="d0",
        )


def test_multi_causal_data_allows_string_user_id_after_optimized_duplicate_check():
    df = pd.DataFrame({
        "user_id": ["u0", "u1", "u2", "u3"],
        "y": [0.5, 1.0, 1.5, 2.0],
        "d0": [1, 0, 1, 0],
        "d1": [0, 1, 0, 1],
        "x": [10, 20, 30, 40],
    })

    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1"],
        confounders=["x"],
        user_id="user_id",
        control_treatment="d0",
    )

    assert list(data.df["user_id"]) == ["u0", "u1", "u2", "u3"]


def test_causal_data_get_df_empty():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d": [0, 1, 0],
        "x": [5, 6, 7]
    })
    cd = CausalData(df=df, outcome="y", treatment="d", confounders=["x"])
    
    empty_df = cd.get_df(include_outcome=False, include_treatment=False, include_confounders=False)
    assert empty_df.shape == (3, 0)
    assert isinstance(empty_df, pd.DataFrame)

def test_multi_causal_data_basic():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d1": [0, 1, 0],
        "d2": [1, 0, 1],
        "x": [5, 6, 7]
    })
    mcd = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d1", "d2"],
        confounders=["x"],
        control_treatment="d1",
    )
    assert mcd.treatment_names == ["d1", "d2"]
    assert mcd.df.shape == (3, 4)
    # Check canonicalization to int8
    assert mcd.df["d1"].dtype == "int8"
    assert mcd.df["d2"].dtype == "int8"


def test_multi_causal_data_binary_validation():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d1": [0, 1, 2], # Not binary
        "d2": [1, 0, 0],
    })
    with pytest.raises(ValueError, match="must be binary encoded"):
        MultiCausalData(df=df, outcome="y", treatment_names=["d1", "d2"], control_treatment="d1")

    df_float = pd.DataFrame({
        "y": [1, 2, 3],
        "d1": [0.0, 1.0, 1e-13], # Close enough to 0
        "d2": [1.0, 0.0, 1.0],
    })
    mcd = MultiCausalData(df=df_float, outcome="y", treatment_names=["d1", "d2"], control_treatment="d1")
    assert mcd.df["d1"].dtype == "int8"
    assert (mcd.df["d1"] == [0, 1, 0]).all()


def test_multi_causal_data_requires_control_treatment():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d0": [1, 0, 1],
        "d1": [0, 1, 0],
    })
    with pytest.raises(ValidationError, match="control_treatment"):
        MultiCausalData(df=df, outcome="y", treatment_names=["d0", "d1"])


def test_multi_causal_data_requires_one_hot_rows():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d0": [1, 0, 0],
        "d1": [0, 1, 1],
        "d2": [0, 1, 0],  # second row has two active treatments
        "x": [5, 6, 7]
    })
    with pytest.raises(ValueError, match="one-hot encoded per row"):
        MultiCausalData(
            df=df,
            outcome="y",
            treatment_names=["d0", "d1", "d2"],
            confounders=["x"],
            control_treatment="d0",
        )


def test_multi_causal_data_control_treatment_reordering():
    df = pd.DataFrame({
        "y": [1, 2, 3, 4],
        "d0": [1, 0, 1, 0],
        "d1": [0, 1, 0, 0],
        "d2": [0, 0, 0, 1],
        "x": [5, 6, 7, 8]
    })
    mcd = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d1", "d2", "d0"],
        control_treatment="d0",
        confounders=["x"],
    )
    assert mcd.treatment_names[0] == "d0"
    assert mcd.control_treatment == "d0"


def test_multi_causal_data_requires_at_least_two_treatments():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d1": [0, 1, 0],
    })
    with pytest.raises(ValueError, match="at least 2 treatment columns"):
        MultiCausalData(df=df, outcome="y", treatment_names=["d1"], control_treatment="d1")



def test_multi_causal_data_outcome_and_user_id_are_stripped():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d0": [1, 0, 1],
        "d1": [0, 1, 0],
        "uid": [10, 11, 12],
    })
    mcd = MultiCausalData(
        df=df,
        outcome="  y  ",
        treatment_names=["d0", "d1"],
        user_id="  uid  ",
        control_treatment="d0",
    )
    assert mcd.outcome == "y"
    assert mcd.user_id == "uid"

def test_multi_causal_data_outcome_and_user_id_cannot_be_blank():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d0": [1, 0, 1],
        "d1": [0, 1, 0],
        "uid": [10, 11, 12],
    })
    with pytest.raises(ValidationError, match="outcome must be a non-empty string"):
        MultiCausalData(df=df, outcome="   ", treatment_names=["d0", "d1"], control_treatment="d0")
    with pytest.raises(ValidationError, match="user_id must be a non-empty string"):
        MultiCausalData(df=df, outcome="y", treatment_names=["d0", "d1"], user_id="   ", control_treatment="d0")

def test_multi_causal_data_finite_checks_for_used_columns():
    df_outcome_inf = pd.DataFrame({
        "y": [1.0, np.inf, 3.0],
        "d0": [1, 0, 1],
        "d1": [0, 1, 0],
    })
    with pytest.raises(ValueError, match="outcome must contain only finite values"):
        MultiCausalData(df=df_outcome_inf, outcome="y", treatment_names=["d0", "d1"], control_treatment="d0")

    df_confounder_inf = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "d0": [1, 0, 1],
        "d1": [0, 1, 0],
        "x": [1.0, -np.inf, 2.0],
    })
    with pytest.raises(ValueError, match="confounder must contain only finite values"):
        MultiCausalData(
            df=df_confounder_inf,
            outcome="y",
            treatment_names=["d0", "d1"],
            confounders=["x"],
            control_treatment="d0",
        )

    df_treatment_inf = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "d0": [1.0, 0.0, np.inf],
        "d1": [0.0, 1.0, 0.0],
    })
    with pytest.raises(ValueError, match="treatment must contain only finite values"):
        MultiCausalData(df=df_treatment_inf, outcome="y", treatment_names=["d0", "d1"], control_treatment="d0")

def test_multi_causal_data_get_df():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d1": [0, 1, 0],
        "d2": [1, 0, 1],
        "x": [5, 6, 7]
    })
    mcd = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d1", "d2"],
        confounders=["x"],
        control_treatment="d1",
    )
    
    df_only_t = mcd.get_df(include_outcome=False, include_confounders=False)
    assert list(df_only_t.columns) == ["d1", "d2"]
    
    df_empty = mcd.get_df(include_outcome=False, include_confounders=False, include_treatments=False)
    assert df_empty.shape == (3, 0)


