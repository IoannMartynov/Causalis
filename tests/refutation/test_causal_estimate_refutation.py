import pytest
import numpy as np
import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness import IRM
from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)

@pytest.fixture
def sample_causal_data():
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, (n, 2))
    d = np.random.binomial(1, 0.5, n)
    y = 0.5 * d + x[:, 0] + np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x[:, 0],
        'x2': x[:, 1]
    })
    return CausalData(df=df, treatment='d', outcome='y', confounders=['x1', 'x2'])

def test_overlap_diagnostics_with_causal_estimate(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = run_overlap_diagnostics(sample_causal_data, result)
    assert "summary" in report
    assert "ks" in report
    assert "auc" in report

def test_score_diagnostics_with_causal_estimate(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = run_score_diagnostics(sample_causal_data, result)
    assert "summary" in report
    assert "orthogonality_derivatives" in report

def test_unconfoundedness_diagnostics_with_causal_estimate(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = run_unconfoundedness_diagnostics(sample_causal_data, result)
    assert "summary" in report
    assert "balance" in report
    assert "overall_flag" in report

def test_unconfoundedness_diagnostics_returns_balance_outputs(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = run_unconfoundedness_diagnostics(sample_causal_data, result)
    assert "smd" in report["balance"]
    assert "smd_unweighted" in report["balance"]
    assert "pass" in report["balance"]

def test_score_diagnostics_with_missing_y_d_falls_back_to_causal_data(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')

    diag_without_yd = result.diagnostic_data.model_copy(update={"y": None, "d": None})
    estimate_without_yd = result.model_copy(update={"diagnostic_data": diag_without_yd})

    report = run_score_diagnostics(sample_causal_data, estimate_without_yd)
    assert "summary" in report
    assert report["meta"]["n"] == int(sample_causal_data.get_df().shape[0])
