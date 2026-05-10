import numpy as np
import pandas as pd
import pytest

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.cuped.model import CUPEDModel
from causalis.scenarios.cuped.refutation import (
    regression_assumptions_table_from_data,
    regression_assumptions_table_from_estimate,
)


def _make_data(n: int = 280, seed: int = 17) -> CausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.4 * x1 + rng.normal(scale=0.8, size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 0.5 + 1.3 * d + 0.7 * x1 - 0.3 * x2 + rng.normal(scale=1.1, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])


def test_regression_assumptions_table_from_estimate_has_expected_columns():
    data = _make_data()
    estimate = CUPEDModel().fit(data, covariates=["x1", "x2"]).estimate()
    table = regression_assumptions_table_from_estimate(
        data,
        estimate,
        style_regression_assumptions_table=lambda t: t,
    )

    expected_cols = {"test_id", "test", "flag", "value", "threshold", "message"}
    assert expected_cols.issubset(set(table.columns))
    assert len(table) >= 8
    assert set(table["flag"].unique()).issubset({"GREEN", "YELLOW", "RED"})


def test_regression_assumptions_table_from_estimate_keeps_estimate_only_call():
    data = _make_data()
    estimate = CUPEDModel().fit(data, covariates=["x1", "x2"]).estimate()
    table = regression_assumptions_table_from_estimate(
        estimate,
        style_regression_assumptions_table=lambda t: t,
    )

    expected_cols = {"test_id", "test", "flag", "value", "threshold", "message"}
    assert expected_cols.issubset(set(table.columns))
    assert len(table) >= 8
    assert set(table["flag"].unique()).issubset({"GREEN", "YELLOW", "RED"})


def test_regression_assumptions_table_from_data_runs_end_to_end():
    data = _make_data()
    table = regression_assumptions_table_from_data(data=data, covariates=["x1", "x2"])
    assert len(table) >= 8
    assert "Design rank" in set(table["test"].tolist())


def test_regression_assumptions_table_from_estimate_validates_data_alignment():
    data = _make_data()
    estimate = CUPEDModel().fit(data, covariates=["x1", "x2"]).estimate()

    bad_df = data.df.copy()
    bad_df["d_other"] = bad_df["d"]
    bad_data = CausalData(
        df=bad_df,
        treatment="d_other",
        outcome="y",
        confounders=["x1", "x2"],
    )

    with pytest.raises(ValueError, match="estimate.treatment must match"):
        regression_assumptions_table_from_estimate(
            bad_data,
            estimate,
            style_regression_assumptions_table=lambda t: t,
        )


def test_regression_assumptions_rank_deficiency_raises():
    data = _make_data()
    df = data.df.copy()
    df["x1_dup"] = df["x1"]
    with pytest.raises(ValueError, match="identical values"):
        CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x1_dup", "x2"])
