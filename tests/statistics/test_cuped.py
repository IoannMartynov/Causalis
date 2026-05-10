
import pytest
import numpy as np
import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.preperiod import corr_on_scale
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.scenarios.cuped.dgp import generate_cuped_tweedie_26, _resolve_second_pre_target
from causalis.scenarios.cuped.model import CUPEDModel
from causalis.scenarios.cuped.refutation import CUPEDRefutationConfig

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = np.random.normal(10, 1, size=(n, 1))
    D = np.random.binomial(1, 0.5, size=(n, 1))
    # Y = 1 + 5*D + 2*X + noise
    Y = 1 + 5*D.flatten() + 2*X.flatten() + np.random.normal(0, 0.1, size=n)
    
    df = pd.DataFrame({
        'y': Y,
        'd': D.flatten(),
        'x1': X.flatten()
    })
    return CausalData(df=df, treatment='d', outcome='y', confounders=['x1'])

def test_cuped_init():
    # Valid initializations
    CUPEDModel(cov_type='HC0', alpha=0.01)
    model = CUPEDModel()
    assert model.cov_type == 'HC2'
    assert model.center_covariates is True
    assert model.centering_scope == "global"
    assert model.adjustment == 'lin'

def test_cuped_fit_estimate(sample_data):
    # Standard Lin adjustment
    model = CUPEDModel()
    model.fit(sample_data, covariates=['x1'])
    results = model.estimate()
    
    assert isinstance(results, CausalEstimate)
    assert np.isclose(results.value, 5.0, atol=0.1)
    assert results.outcome == "y"
    assert results.treatment == "d"
    assert results.model_options.get("centering_scope") == "global"
    assert results.diagnostic_data.adj_type == 'lin'
    assert len(results.diagnostic_data.beta_covariates) == 1
    # In Lin model with 1 covariate, we should have 1 main effect and 1 interaction
    assert len(results.diagnostic_data.gamma_interactions) == 1


def test_cuped_diagnostics_se_reduction_and_r2(sample_data):
    model = CUPEDModel(cov_type="HC2")
    model.fit(sample_data, covariates=["x1"])
    results = model.estimate()
    diag = results.diagnostic_data

    se_adj = float(np.asarray(model._result.bse, dtype=float)[1])
    se_naive = float(np.asarray(model._result_naive.bse, dtype=float)[1])
    expected_variance = 100.0 * (1.0 - (se_adj ** 2) / (se_naive ** 2)) if se_naive > 0 else np.nan
    expected_se = 100.0 * (1.0 - se_adj / se_naive) if se_naive > 0 else np.nan

    assert np.isclose(diag.variance_reduction_pct_same_cov, expected_variance, rtol=1e-12, atol=1e-12)
    assert np.isclose(diag.standard_error_reduction_pct_same_cov, expected_se, rtol=1e-12, atol=1e-12)
    assert np.isclose(diag.r2_naive, float(model._result_naive.rsquared), rtol=1e-12, atol=1e-12)
    assert np.isclose(diag.r2_adj, float(model._result.rsquared), rtol=1e-12, atol=1e-12)
    assert len(diag.covariate_outcome_corr) == 1
    assert np.isfinite(diag.covariate_outcome_corr[0])

    summary = model.summary_dict()
    assert "variance_reduction_pct_same_cov" in summary
    assert "standard_error_reduction_pct_same_cov" in summary
    assert "r2_naive" in summary
    assert "r2_adj" in summary
    assert "covariate_outcome_corr" in summary
    assert summary["centering_scope"] == "global"
    assert summary["relative_ci_method"] == "delta"
    assert summary["relative_denominator"] == "adjusted_control"
    assert "run_regression_checks" not in summary
    assert "check_action" not in summary
    assert "dropped_covariates" in summary
    assert summary["dropped_covariates"] == []


def test_cuped_beta_gamma_extraction_matches_design_names():
    rng = np.random.default_rng(99)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 1.0 + 0.7 * d + 0.5 * x1 - 0.4 * x2 + 1.2 * d * x1 - 0.9 * d * x2 + rng.normal(scale=0.3, size=n)
    data = CausalData(
        df=pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2}),
        treatment="d",
        outcome="y",
        confounders=["x1", "x2"],
    )

    model = CUPEDModel(cov_type="HC2").fit(data, covariates=["x1", "x2"])
    results = model.estimate()
    diag = results.diagnostic_data

    params = np.asarray(model._result.params, dtype=float)
    exog_names = list(model._result.model.exog_names)
    beta_idx = [i for i, n in enumerate(exog_names) if n.endswith("__centered")]
    gamma_idx = [i for i, n in enumerate(exog_names) if n.startswith("d:")]
    expected_beta = np.asarray(params[beta_idx], dtype=float)
    expected_gamma = np.asarray(params[gamma_idx], dtype=float)

    assert np.allclose(diag.beta_covariates, expected_beta, rtol=1e-12, atol=1e-12)
    assert np.allclose(diag.gamma_interactions, expected_gamma, rtol=1e-12, atol=1e-12)
    assert len(diag.beta_covariates) == 2
    assert len(diag.gamma_interactions) == 2


def test_cuped_tau_invariant_to_constant_covariate_shift(sample_data):
    base = sample_data.df.copy()
    shifted = base.copy()
    shifted["x1"] = shifted["x1"] + 1234.5

    data_base = CausalData(df=base, treatment="d", outcome="y", confounders=["x1"])
    data_shifted = CausalData(df=shifted, treatment="d", outcome="y", confounders=["x1"])

    model_base = CUPEDModel(cov_type="HC2").fit(data_base, covariates=["x1"])
    model_shifted = CUPEDModel(cov_type="HC2").fit(data_shifted, covariates=["x1"])

    est_base = model_base.estimate()
    est_shifted = model_shifted.estimate()

    assert np.isclose(est_base.value, est_shifted.value, rtol=1e-12, atol=1e-12)


def test_cuped_tau_equals_average_fitted_contrast_under_global_centering():
    rng = np.random.default_rng(101)
    n = 600
    x1 = rng.normal(loc=3.0, scale=1.2, size=n)
    x2 = rng.normal(loc=-2.0, scale=0.8, size=n)
    d = rng.binomial(1, 0.45, size=n)
    y = 1.5 + 0.8 * d + 0.4 * x1 - 0.2 * x2 + 1.1 * d * x1 - 0.7 * d * x2 + rng.normal(scale=0.3, size=n)
    data = CausalData(
        df=pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2}),
        treatment="d",
        outcome="y",
        confounders=["x1", "x2"],
    )

    model = CUPEDModel(cov_type="HC2").fit(data, covariates=["x1", "x2"])
    estimate = model.estimate()

    params = np.asarray(model._result.params, dtype=float)
    design = pd.DataFrame(
        np.asarray(model._result.model.exog, dtype=float),
        columns=list(model._result.model.exog_names),
    )
    design_control = design.copy()
    design_treated = design.copy()
    design_control["d"] = 0.0
    design_treated["d"] = 1.0
    for name in design.columns:
        if name.startswith("d:"):
            raw_name = name.split(":", 1)[1]
            design_control[name] = 0.0
            design_treated[name] = design[f"{raw_name}__centered"]

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        fitted_contrast = design_treated.to_numpy() @ params - design_control.to_numpy() @ params
    assert np.isclose(float(np.mean(fitted_contrast)), estimate.value, rtol=1e-12, atol=1e-12)

def test_cuped_lin_with_interactions():
    # Data with true interaction
    np.random.seed(42)
    n = 1000
    X = np.random.normal(10, 1, size=(n, 1))
    D = np.random.binomial(1, 0.5, size=(n, 1))
    # Y = 1 + 2*D + 3*X + 5*D*X + noise
    # ATE = 2 + 5*E[X] = 2 + 5*10 = 52
    Y = 1 + 2*D.flatten() + 3*X.flatten() + 5*D.flatten()*X.flatten() + np.random.normal(0, 0.1, size=n)
    
    df = pd.DataFrame({'y': Y, 'd': D.flatten(), 'x': X.flatten()})
    data = CausalData(df=df, treatment='d', outcome='y', confounders=['x'])
    
    model = CUPEDModel()
    model.fit(data, covariates=['x'])
    results = model.estimate()
    
    # ATE should be correctly estimated around 52
    assert np.isclose(results.value, 52.0, atol=0.5)
    # gamma interaction coefficient should be around 5
    assert np.isclose(results.diagnostic_data.gamma_interactions[0], 5.0, atol=0.5)

def test_cuped_no_covariates():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'y': np.random.normal(0, 1, n),
        'd': np.random.binomial(1, 0.5, n)
    })
    data = CausalData(df=df, treatment='d', outcome='y', confounders=[])
    
    model = CUPEDModel()
    model.fit(data, covariates=[])
    results = model.estimate()
    assert results.diagnostic_data.ate_naive == results.value
    assert len(results.diagnostic_data.beta_covariates) == 0
    assert len(results.diagnostic_data.covariate_outcome_corr) == 0


def test_cuped_requires_explicit_covariates(sample_data):
    model = CUPEDModel()
    with pytest.raises(ValueError, match="covariates must be provided explicitly"):
        model.fit(sample_data)


def test_cuped_accepts_sequence_covariates(sample_data):
    model_tuple = CUPEDModel().fit(sample_data, covariates=("x1",))
    model_list = CUPEDModel().fit(sample_data, covariates=["x1"])

    assert model_tuple._covariate_names == ["x1"]
    assert model_list._covariate_names == ["x1"]


def test_cuped_rejects_ndarray_covariates(sample_data):
    with pytest.raises(ValueError, match="Sequence\\[str\\]"):
        CUPEDModel().fit(sample_data, covariates=np.array(["x1"], dtype=object))


def test_cuped_rejects_string_covariates(sample_data):
    model = CUPEDModel()
    with pytest.raises(ValueError, match="sequence of column names"):
        model.fit(sample_data, covariates="x1")


def test_cuped_rejects_duplicate_covariates(sample_data):
    model = CUPEDModel()
    with pytest.raises(ValueError, match="must not contain duplicates"):
        model.fit(sample_data, covariates=["x1", "x1"])


def test_cuped_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        CUPEDModel(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        CUPEDModel(alpha=1.0)


def test_cuped_rejects_invalid_relative_ci_method():
    with pytest.raises(ValueError, match="relative_ci_method"):
        CUPEDModel(relative_ci_method="unknown")  # type: ignore[arg-type]


def test_cuped_rejects_invalid_relative_denominator():
    with pytest.raises(ValueError, match="relative_denominator"):
        CUPEDModel(relative_denominator="unknown")  # type: ignore[arg-type]


def test_cuped_accepts_grouped_refutation_config(sample_data):
    config = CUPEDRefutationConfig(
        run_regression_checks=False,
        check_action="ignore",
        condition_number_warn_threshold=123.0,
        vif_warn_threshold=7.0,
        winsor_q=None,
    )
    model = CUPEDModel(refutation_config=config).fit(sample_data, covariates=["x1"])
    estimate = model.estimate()

    assert model.refutation_config == config
    assert estimate.model_options["refutation_config"]["run_regression_checks"] is False
    assert estimate.model_options["refutation_config"]["condition_number_warn_threshold"] == 123.0
    assert "run_regression_checks" not in estimate.model_options
    assert "check_action" not in estimate.model_options
    assert estimate.diagnostic_data.regression_checks is None


def test_cuped_rejects_invalid_refutation_config_type():
    with pytest.raises(TypeError, match="refutation_config"):
        CUPEDModel(refutation_config={"check_action": "ignore"})  # type: ignore[arg-type]


def test_cuped_raises_when_condition_number_huge():
    rng = np.random.default_rng(13)
    n = 250
    d = rng.binomial(1, 0.5, size=n)
    x1 = rng.normal(size=n)
    y = 0.7 + 1.1 * d + 0.4 * x1 + rng.normal(scale=0.6, size=n)
    data = CausalData(
        df=pd.DataFrame({"y": y, "d": d, "x1": x1}),
        treatment="d",
        outcome="y",
        confounders=["x1"],
    )
    with pytest.raises(ValueError, match="ill-conditioned"):
        CUPEDModel(
            refutation_config=CUPEDRefutationConfig(
                condition_number_warn_threshold=1e-3,
                check_action="raise",
            ),
        ).fit(data, covariates=["x1"])


def test_make_cuped_tweedie_26_has_two_pre_covariates():
    data = generate_cuped_tweedie_26(
        n=2000,
        seed=123,
        add_pre=True,
        include_oracle=False,
        return_causal_data=True
    )

    assert "y_pre" in data.df.columns
    assert "y_pre_2" in data.df.columns
    assert "y_pre" in data.confounders_names
    assert "y_pre_2" in data.confounders_names

    corr = np.corrcoef(data.df["y_pre"], data.df["y_pre_2"])[0, 1]
    assert np.isfinite(corr)
    assert abs(corr) > 0.99


def test_make_cuped_tweedie_26_hits_requested_control_correlations():
    target_1 = 0.70
    target_2 = 0.55
    df = generate_cuped_tweedie_26(
        n=6000,
        seed=42,
        add_pre=True,
        pre_target_corr=target_1,
        pre_target_corr_2=target_2,
        include_oracle=False,
        return_causal_data=False,
    )
    ctrl = (df["d"].to_numpy() == 0)
    corr_1 = corr_on_scale(
        df.loc[ctrl, "y_pre"].to_numpy(dtype=float),
        df.loc[ctrl, "y"].to_numpy(dtype=float),
    )
    corr_2 = corr_on_scale(
        df.loc[ctrl, "y_pre_2"].to_numpy(dtype=float),
        df.loc[ctrl, "y"].to_numpy(dtype=float),
    )
    assert 0.0 < corr_2 <= corr_1 < 1.0
    assert np.isclose(corr_2, target_2, atol=0.02)


def test_resolve_second_pre_target_default_formula():
    assert _resolve_second_pre_target(0.82, None) == pytest.approx(0.72)
    assert _resolve_second_pre_target(0.60, None) == pytest.approx(0.50)
    assert _resolve_second_pre_target(0.40, None) == pytest.approx(0.30)
    assert _resolve_second_pre_target(0.05, None) == pytest.approx(0.0)
    assert _resolve_second_pre_target(0.82, 0.64) == pytest.approx(0.64)


def test_make_cuped_tweedie_26_rejects_reserved_pre_column_names():
    with pytest.raises(ValueError, match="collides with an existing generated column"):
        generate_cuped_tweedie_26(
            n=500,
            seed=7,
            pre_name="y",
            return_causal_data=False
        )


def test_cuped_use_t_auto_hc_small_n_uses_t():
    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 1.0 + 0.5 * d + 0.8 * x + rng.normal(scale=1.0, size=n)
    data = CausalData(df=pd.DataFrame({"y": y, "d": d, "x": x}), treatment="d", outcome="y", confounders=["x"])

    model = CUPEDModel(cov_type="HC3", use_t=None).fit(data, covariates=["x"])

    assert model._use_t_effective is True


def test_cuped_use_t_auto_hc_large_n_uses_normal():
    rng = np.random.default_rng(12)
    n = 5200
    x = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 1.0 + 0.5 * d + 0.8 * x + rng.normal(scale=1.0, size=n)
    data = CausalData(df=pd.DataFrame({"y": y, "d": d, "x": x}), treatment="d", outcome="y", confounders=["x"])

    model = CUPEDModel(cov_type="HC3", use_t=None).fit(data, covariates=["x"])

    assert model._use_t_effective is False


def test_cuped_use_t_explicit_override():
    rng = np.random.default_rng(13)
    n = 5200
    x = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 1.0 + 0.5 * d + 0.8 * x + rng.normal(scale=1.0, size=n)
    data = CausalData(df=pd.DataFrame({"y": y, "d": d, "x": x}), treatment="d", outcome="y", confounders=["x"])

    model_true = CUPEDModel(cov_type="HC3", use_t=True).fit(data, covariates=["x"])
    model_false = CUPEDModel(cov_type="HC3", use_t=False).fit(data, covariates=["x"])

    assert model_true._use_t_effective is True
    assert model_false._use_t_effective is False
