import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.data_contracts import IVCausalData, IVCausalEstimate
from causalis.scenarios.iv import IIVM


def _make_iv_data(
    n: int = 600, seed: int = 123, include_x: bool = True
) -> IVCausalData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p_z = 1.0 / (1.0 + np.exp(-0.4 * x))
    z = rng.binomial(1, p_z)
    p_d = 1.0 / (1.0 + np.exp(-(-0.2 + 1.2 * z + 0.5 * x)))
    d = rng.binomial(1, p_d)
    y = 2.0 * d + 0.8 * x + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "z": z, "x": x})
    confounders = ["x"] if include_x else None
    return IVCausalData.from_df(
        df,
        treatment="d",
        outcome="y",
        instruments="z",
        confounders=confounders,
    )


def _make_iivm(data: IVCausalData) -> IIVM:
    return IIVM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        ml_r=LogisticRegression(max_iter=1000),
        n_folds=3,
        trimming_threshold=1e-3,
        random_state=3,
    )


def test_iivm_late_runs_and_satisfies_score_identities():
    data = _make_iv_data()
    model = _make_iivm(data)

    result = model.fit().estimate(score="LATE")

    assert isinstance(result, IVCausalEstimate)
    assert np.isfinite(result.value)
    assert np.isfinite(result.std_error)
    assert result.ci_lower_absolute < result.ci_upper_absolute
    assert model.coef.shape == (1,)
    assert model.se.shape == (1,)
    assert model.confint().shape == (1, 2)
    summary = result.summary()
    assert "estimand" in summary.index
    assert summary.loc["estimand", "value"] == "LATE"
    assert summary.loc["value_relative", "value"] is None

    diag = result.diagnostic_data
    assert diag is not None
    assert diag.y.shape == (data.df.shape[0],)
    assert diag.z.shape == (data.df.shape[0],)
    assert diag.g0_hat.shape == (data.df.shape[0],)
    assert diag.instrument_overlap is not None
    assert diag.first_stage is not None
    assert diag.reduced_form is not None
    assert "instrument_auc" in diag.instrument_overlap
    assert "weak_iv_flag" in diag.first_stage
    assert "reduced_form_effect" in diag.reduced_form
    assert {"instrument_overlap", "first_stage", "reduced_form"}.issubset(
        diag.diagnostics
    )
    np.testing.assert_allclose(diag.psi_a, -diag.phi_d)
    np.testing.assert_allclose(diag.psi_b, diag.phi_y)
    assert np.isclose(np.mean(diag.psi), 0.0, atol=1e-10)
    assert np.isclose(
        result.value,
        np.mean(diag.phi_y) / np.mean(diag.phi_d),
        atol=1e-12,
    )

    j_hat = -np.mean(diag.phi_d)
    expected_se = np.sqrt(np.mean(diag.psi**2) / (len(diag.psi) * j_hat**2))
    assert np.isclose(result.std_error, expected_se)


def test_iivm_supports_no_confounders():
    data = _make_iv_data(include_x=False)
    result = _make_iivm(data).fit().estimate()

    assert np.isfinite(result.value)
    assert result.diagnostic_data.m_hat.shape == (data.df.shape[0],)


def test_iivm_rejects_unknown_score():
    data = _make_iv_data(n=240)
    model = _make_iivm(data).fit()

    with pytest.raises(ValueError, match="score='LATE'"):
        model.estimate(score="ATE")
