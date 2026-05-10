import numpy as np
import pandas as pd
import pytest

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.cuped.model import CUPEDModel
from causalis.scenarios.cuped.refutation import CUPEDRefutationConfig


def _make_data(n: int = 350, seed: int = 123) -> CausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.35 * x1 + rng.normal(scale=0.9, size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 1.0 + 1.5 * d + 0.8 * x1 - 0.2 * x2 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])


def _make_rank_deficient_df(n: int = 220, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x_dup = x1.copy()
    d = rng.binomial(1, 0.5, size=n)
    y = 0.5 + 1.2 * d + 0.7 * x1 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x_dup": x_dup})
    return df


def test_cuped_regression_checks_attached_by_default():
    data = _make_data()
    model = CUPEDModel().fit(data, covariates=["x1", "x2"])
    estimate = model.estimate()

    checks = estimate.diagnostic_data.regression_checks
    assert checks is not None
    assert checks.full_rank is True
    assert checks.k == model._result.model.exog.shape[1]
    assert checks.rank == checks.k
    assert np.isclose(checks.ate_adj, estimate.value, atol=1e-12, rtol=1e-12)
    assert checks.leverage_cutoff > 0.0
    assert checks.cooks_cutoff > 0.0


def test_cuped_regression_checks_can_be_disabled_or_enabled_per_fit():
    data = _make_data()

    model = CUPEDModel(
        refutation_config=CUPEDRefutationConfig(run_regression_checks=False)
    ).fit(data, covariates=["x1"])
    estimate = model.estimate()
    assert model._regression_checks is None
    assert estimate.diagnostic_data.regression_checks is None

    model = CUPEDModel(
        refutation_config=CUPEDRefutationConfig(run_regression_checks=False)
    ).fit(
        data,
        covariates=["x1"],
        run_checks=True,
    )
    estimate = model.estimate()
    assert model._regression_checks is not None
    assert estimate.diagnostic_data.regression_checks is not None


def test_cuped_regression_checks_raise_on_rank_deficiency():
    df = _make_rank_deficient_df()
    with pytest.raises(ValueError, match="identical values"):
        CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x_dup"])


def test_cuped_regression_checks_raise_on_rank_deficiency_even_with_checks_disabled():
    df = _make_rank_deficient_df()
    with pytest.raises(ValueError, match="identical values"):
        CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x_dup"])


def test_cuped_raise_on_yellow_option():
    data = _make_data()
    model = CUPEDModel(
        refutation_config=CUPEDRefutationConfig(check_action="raise")
    ).fit(data, covariates=["x1", "x2"])
    table = pd.DataFrame(
        [
            {
                "test_id": "dummy",
                "test": "Dummy",
                "flag": "YELLOW",
                "value": "n/a",
                "threshold": "n/a",
                "message": "yellow flag",
            }
        ]
    )

    # Default behavior keeps YELLOW silent.
    model._signal_assumption_flags(table)

    strict_model = CUPEDModel(
        refutation_config=CUPEDRefutationConfig(check_action="raise", raise_on_yellow=True)
    ).fit(
        data,
        covariates=["x1", "x2"],
    )
    with pytest.raises(ValueError, match="yellow flag"):
        strict_model._signal_assumption_flags(table)
