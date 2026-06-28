import numpy as np
import pytest
from pydantic import ValidationError

from causalis.data_contracts.causal_diagnostic_data import (
    DiagnosticData,
    IVDiagnosticData,
    UnconfoundednessDiagnosticData,
)
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.iv_causal_estimate import IVCausalEstimate
from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate


def test_diagnostic_data_instantiation():
    diag = DiagnosticData()
    assert isinstance(diag, DiagnosticData)


def test_unconfoundedness_diagnostic_data_instantiation():
    m_hat = np.array([0.1, 0.2, 0.3])
    d = np.array([0, 1, 0])
    diag = UnconfoundednessDiagnosticData(m_hat=m_hat, d=d)
    
    assert np.array_equal(diag.m_hat, m_hat)
    assert np.array_equal(diag.d, d)
    assert diag.y is None
    assert diag.x is None
    assert diag.trimming_threshold == 0.0


def test_unconfoundedness_diagnostic_data_full_instantiation():
    m_hat = np.array([0.1, 0.2, 0.3])
    d = np.array([0, 1, 0])
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([[1, 2], [3, 4], [5, 6]])
    diag = UnconfoundednessDiagnosticData(
        m_hat=m_hat, d=d, y=y, x=x, trimming_threshold=0.1
    )
    
    assert np.array_equal(diag.m_hat, m_hat)
    assert np.array_equal(diag.d, d)
    assert np.array_equal(diag.y, y)
    assert np.array_equal(diag.x, x)
    assert diag.trimming_threshold == 0.1


def test_unconfoundedness_diagnostic_data_missing_fields():
    with pytest.raises(ValidationError):
        UnconfoundednessDiagnosticData(m_hat=np.array([0.1]))


def test_causal_estimate_with_diagnostic_data():
    m_hat = np.array([0.1, 0.2, 0.3])
    d = np.array([0, 1, 0])
    diag = UnconfoundednessDiagnosticData(m_hat=m_hat, d=d)
    
    estimate = CausalEstimate(
        estimand="ATE",
        model="test_model",
        value=1.0,
        ci_upper_absolute=1.5,
        ci_lower_absolute=0.5,
        alpha=0.05,
        is_significant=True,
        n_treated=10,
        n_control=10,
        treatment_mean=1.2,
        control_mean=0.8,
        outcome="y",
        treatment="d",
        diagnostic_data=diag
    )
    
    assert estimate.diagnostic_data == diag
    assert isinstance(estimate.diagnostic_data, UnconfoundednessDiagnosticData)
    assert np.array_equal(estimate.diagnostic_data.m_hat, m_hat)


def test_causal_estimate_without_diagnostic_data():
    estimate = CausalEstimate(
        estimand="ATE",
        model="test_model",
        value=1.0,
        ci_upper_absolute=1.5,
        ci_lower_absolute=0.5,
        alpha=0.05,
        is_significant=True,
        n_treated=10,
        n_control=10,
        treatment_mean=1.2,
        control_mean=0.8,
        outcome="y",
        treatment="d"
    )
    
    assert estimate.diagnostic_data is None


def test_causal_estimate_summary_starts_with_outcome_column_name():
    estimate = CausalEstimate(
        estimand="ATE",
        model="test_model",
        value=1.0,
        ci_upper_absolute=1.5,
        ci_lower_absolute=0.5,
        alpha=0.05,
        is_significant=True,
        n_treated=10,
        n_control=10,
        treatment_mean=1.2,
        control_mean=0.8,
        outcome="y",
        treatment="d",
    )

    summary = estimate.summary()

    assert summary.index[0] == "outcome"
    assert summary.iloc[0]["value"] == "y"


def test_causal_estimate_with_empty_dict_diagnostic_data():
    estimate = CausalEstimate(
        estimand="ATE",
        model="test_model",
        value=1.0,
        ci_upper_absolute=1.5,
        ci_lower_absolute=0.5,
        alpha=0.05,
        is_significant=True,
        n_treated=10,
        n_control=10,
        treatment_mean=1.2,
        control_mean=0.8,
        outcome="y",
        treatment="d",
        diagnostic_data={}
    )
    
    assert isinstance(estimate.diagnostic_data, DiagnosticData)
    assert not isinstance(estimate.diagnostic_data, UnconfoundednessDiagnosticData)


def test_iv_causal_estimate_with_diagnostic_data():
    y = np.array([1.0, 2.0, 3.0])
    d = np.array([0, 1, 1])
    z = np.array([0, 1, 0])
    x = np.array([[0.1], [0.2], [0.3]])
    signal = np.array([0.1, -0.2, 0.1])
    diag = IVDiagnosticData(
        y=y,
        d=d,
        z=z,
        x=x,
        x_names=["x"],
        g0_hat=np.array([0.8, 0.9, 1.0]),
        g1_hat=np.array([1.0, 1.2, 1.4]),
        m_hat=np.array([0.4, 0.5, 0.6]),
        m_hat_raw=np.array([0.4, 0.5, 0.6]),
        r0_hat=np.array([0.2, 0.3, 0.4]),
        r1_hat=np.array([0.7, 0.8, 0.9]),
        folds=np.array([0, 1, 0]),
        psi=signal,
        psi_a=-signal,
        psi_b=signal,
        phi_y=signal,
        phi_d=-signal,
        instrument_overlap={"instrument_auc": 0.5},
        first_stage={"weak_iv_flag": "GREEN"},
        reduced_form={"reduced_form_effect": 0.1},
        diagnostics={"n_obs": 3},
    )

    estimate = IVCausalEstimate(
        estimand="LATE",
        model="IIVM",
        value=1.0,
        std_error=0.1,
        t_stat=10.0,
        p_value=0.01,
        ci_lower_absolute=0.8,
        ci_upper_absolute=1.2,
        alpha=0.05,
        is_significant=True,
        outcome="y",
        treatment="d",
        instrument="z",
        confounders=["x"],
        diagnostic_data=diag,
    )

    assert estimate.diagnostic_data == diag
    assert estimate.diagnostic_data.instrument_overlap["instrument_auc"] == 0.5
    assert estimate.diagnostic_data.first_stage["weak_iv_flag"] == "GREEN"
    assert estimate.diagnostic_data.reduced_form["reduced_form_effect"] == 0.1
    summary = estimate.summary()
    assert summary.index[0] == "outcome"
    assert summary.iloc[0]["value"] == "y"
    assert summary.loc["estimand", "value"] == "LATE"
    assert summary.loc["value", "value"] == "1.0000 (ci_abs: 0.8000, 1.2000)"
    assert summary.loc["value_relative", "value"] is None


def test_iv_causal_estimate_summary_formats_relative_ci():
    estimate = IVCausalEstimate(
        estimand="LATE",
        model="IIVM",
        value=1.0,
        std_error=0.1,
        t_stat=10.0,
        p_value=0.01,
        ci_lower_absolute=0.8,
        ci_upper_absolute=1.2,
        value_relative=0.2,
        ci_lower_relative=0.1,
        ci_upper_relative=0.3,
        alpha=0.05,
        is_significant=True,
        outcome="y",
        treatment="d",
        instrument="z",
    )

    assert (
        estimate.summary().loc["value_relative", "value"]
        == "0.2000 (ci_rel: 0.1000, 0.3000)"
    )


def test_multi_causal_estimate_summary_starts_with_outcome_column_name():
    estimate = MultiCausalEstimate(
        estimand="ATE",
        model="test_model",
        value=np.array([1.0, 2.0]),
        ci_upper_absolute=np.array([1.5, 2.5]),
        ci_lower_absolute=np.array([0.5, 1.5]),
        alpha=0.05,
        p_value=np.array([0.01, 0.02]),
        is_significant=[True, True],
        n_treated=20,
        n_control=10,
        outcome="y",
        treatment=["d0", "d1", "d2"],
    )

    summary = estimate.summary()

    assert summary.index[0] == "outcome"
    assert summary.iloc[0].tolist() == ["y", "y"]
