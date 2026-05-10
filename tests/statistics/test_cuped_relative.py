import numpy as np
import pandas as pd

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.cuped.model import CUPEDModel


def _make_cuped_data(n: int = 220, seed: int = 7) -> CausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = 0.8 + 0.9 * d + 1.1 * x1 - 0.4 * x2 + rng.normal(scale=1.8, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])


def _manual_relative_ci_delta_raw_control(model: CUPEDModel, alpha: float) -> tuple[float, float, float]:
    result = model._result
    y = np.asarray(result.model.endog, dtype=float)
    design = np.asarray(result.model.exog, dtype=float)
    d = np.asarray(design[:, 1], dtype=float)

    tau = float(np.asarray(result.params, dtype=float)[1])
    se_tau = float(np.asarray(result.bse, dtype=float)[1])
    ci_abs = np.asarray(result.conf_int(alpha=alpha), dtype=float)
    ci_low_abs = float(ci_abs[1, 0])
    ci_high_abs = float(ci_abs[1, 1])
    mu_c = float(np.mean(y[d == 0]))

    tau_rel = 100.0 * tau / mu_c

    control_mask = d == 0.0
    n_control = int(np.sum(control_mask))
    var_tau = float(np.asarray(result.cov_params(), dtype=float)[1, 1])
    var_mu = float(np.var(y[control_mask], ddof=1) / n_control) if n_control > 1 else np.nan
    xtx_inv = np.linalg.pinv(design.T @ design)
    resid = np.asarray(result.resid, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        tau_if = (design @ xtx_inv[:, 1]) * resid
    var_tau_if = float(np.sum(tau_if ** 2))
    if var_tau_if > 0.0:
        tau_if = tau_if * np.sqrt(var_tau / var_tau_if)
    mu_if = np.zeros_like(y, dtype=float)
    mu_if[control_mask] = (y[control_mask] - mu_c) / n_control
    cov_tau_mu = float(np.sum(tau_if * mu_if))

    d_tau = 100.0 / mu_c
    d_mu = -100.0 * tau / (mu_c ** 2)
    var_rel = (d_tau ** 2) * var_tau + (d_mu ** 2) * var_mu + 2.0 * d_tau * d_mu * cov_tau_mu
    se_rel = float(np.sqrt(max(var_rel, 0.0)))

    crit = max(abs(ci_high_abs - tau), abs(tau - ci_low_abs)) / se_tau if se_tau > 0 else np.nan
    ci_low_rel = float(tau_rel - crit * se_rel)
    ci_high_rel = float(tau_rel + crit * se_rel)
    if ci_low_rel > ci_high_rel:
        ci_low_rel, ci_high_rel = ci_high_rel, ci_low_rel

    return tau_rel, ci_low_rel, ci_high_rel


def _manual_relative_ci_delta_adjusted_control(model: CUPEDModel, alpha: float) -> tuple[float, float, float]:
    result = model._result
    params = np.asarray(result.params, dtype=float)
    cov = np.asarray(result.cov_params(), dtype=float)

    tau = float(params[1])
    denom = float(params[0])
    se_tau = float(np.asarray(result.bse, dtype=float)[1])
    ci_abs = np.asarray(result.conf_int(alpha=alpha), dtype=float)
    ci_low_abs = float(ci_abs[1, 0])
    ci_high_abs = float(ci_abs[1, 1])

    tau_rel = 100.0 * tau / denom
    grad = np.zeros((cov.shape[0],), dtype=float)
    grad[0] = -100.0 * tau / (denom ** 2)
    grad[1] = 100.0 / denom
    var_rel = float(grad @ cov @ grad)
    se_rel = float(np.sqrt(max(var_rel, 0.0)))

    crit = max(abs(ci_high_abs - tau), abs(tau - ci_low_abs)) / se_tau if se_tau > 0 else np.nan
    ci_low_rel = float(tau_rel - crit * se_rel)
    ci_high_rel = float(tau_rel + crit * se_rel)
    if ci_low_rel > ci_high_rel:
        ci_low_rel, ci_high_rel = ci_high_rel, ci_low_rel

    return tau_rel, ci_low_rel, ci_high_rel


def test_cuped_relative_ci_delta_method_matches_default_adjusted_control_formula():
    data = _make_cuped_data(n=280, seed=42)
    model = CUPEDModel(cov_type="HC0", alpha=0.1, use_t=False).fit(
        data, covariates=["x1", "x2"]
    )
    estimate = model.estimate(alpha=0.1)

    exp_rel, exp_low, exp_high = _manual_relative_ci_delta_adjusted_control(model, alpha=0.1)

    assert estimate.model_options["relative_denominator"] == "adjusted_control"
    assert np.isclose(estimate.value_relative, exp_rel, rtol=1e-10, atol=1e-10)
    assert np.isclose(estimate.ci_lower_relative, exp_low, rtol=1e-10, atol=1e-10)
    assert np.isclose(estimate.ci_upper_relative, exp_high, rtol=1e-10, atol=1e-10)


def test_cuped_relative_ci_is_not_legacy_rescaled_absolute_ci():
    data = _make_cuped_data(n=90, seed=123)
    model = CUPEDModel(cov_type="HC0", alpha=0.05, use_t=False, relative_ci_method="delta").fit(
        data, covariates=["x1", "x2"]
    )
    estimate = model.estimate(alpha=0.05)

    denom = float(np.asarray(model._result.params, dtype=float)[0])

    ci_abs = np.asarray(model._result.conf_int(alpha=0.05), dtype=float)
    legacy_low = float(ci_abs[1, 0] / denom * 100.0)
    legacy_high = float(ci_abs[1, 1] / denom * 100.0)

    gap = abs(float(estimate.ci_lower_relative) - legacy_low) + abs(float(estimate.ci_upper_relative) - legacy_high)
    assert gap > 1e-4


def test_cuped_relative_ci_nan_when_adjusted_control_mean_zero():
    d = np.array([0] * 30 + [1] * 30, dtype=int)
    x = np.linspace(-1.0, 1.0, 60)
    y = np.where(d == 0, 0.0, 2.0)
    df = pd.DataFrame({"y": y, "d": d, "x1": x})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])

    model = CUPEDModel(cov_type="HC0", alpha=0.05, use_t=False).fit(
        data, covariates=[]
    )
    estimate = model.estimate()

    assert np.isnan(estimate.value_relative)
    assert np.isnan(estimate.ci_lower_relative)
    assert np.isnan(estimate.ci_upper_relative)


def test_cuped_relative_adjusted_control_denominator_uses_intercept_and_covariance():
    data = _make_cuped_data(n=260, seed=99)
    model = CUPEDModel(
        cov_type="HC0",
        alpha=0.1,
        use_t=False,
        relative_ci_method="delta",
        relative_denominator="adjusted_control",
    ).fit(data, covariates=["x1", "x2"])
    estimate = model.estimate(alpha=0.1)

    exp_rel, exp_low, exp_high = _manual_relative_ci_delta_adjusted_control(model, alpha=0.1)

    assert estimate.model_options["relative_denominator"] == "adjusted_control"
    assert np.isclose(estimate.value_relative, exp_rel, rtol=1e-10, atol=1e-10)
    assert np.isclose(estimate.ci_lower_relative, exp_low, rtol=1e-10, atol=1e-10)
    assert np.isclose(estimate.ci_upper_relative, exp_high, rtol=1e-10, atol=1e-10)


def test_cuped_relative_raw_control_denominator_uses_raw_control_mean():
    data = _make_cuped_data(n=260, seed=101)
    model = CUPEDModel(
        cov_type="HC0",
        alpha=0.1,
        use_t=False,
        relative_denominator="raw_control",
    ).fit(data, covariates=["x1", "x2"])
    estimate = model.estimate(alpha=0.1)

    exp_rel, exp_low, exp_high = _manual_relative_ci_delta_raw_control(model, alpha=0.1)

    assert estimate.model_options["relative_denominator"] == "raw_control"
    assert np.isclose(estimate.value_relative, exp_rel, rtol=1e-10, atol=1e-10)
    assert np.isclose(estimate.ci_lower_relative, exp_low, rtol=1e-10, atol=1e-10)
    assert np.isclose(estimate.ci_upper_relative, exp_high, rtol=1e-10, atol=1e-10)


def test_cuped_relative_ci_bootstrap_produces_finite_interval():
    data = _make_cuped_data(n=220, seed=1234)
    model = CUPEDModel(
        cov_type="HC2",
        alpha=0.1,
        use_t=False,
        relative_ci_method="bootstrap",
        relative_ci_bootstrap_draws=200,
        relative_ci_bootstrap_seed=7,
    ).fit(data, covariates=["x1", "x2"])
    estimate = model.estimate(alpha=0.1)

    assert np.isfinite(estimate.value_relative)
    assert np.isfinite(estimate.ci_lower_relative)
    assert np.isfinite(estimate.ci_upper_relative)
    assert estimate.ci_lower_relative <= estimate.ci_upper_relative
