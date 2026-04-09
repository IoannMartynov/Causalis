import numpy as np
import pandas as pd
import pytest
import warnings
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.regression.linear_model import OLS

from causalis.data_contracts.gate_contrast_estimate import GateContrastEstimate
from causalis.data_contracts.gate_estimate import GateEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.gate.model import (
    _coerce_groups_to_basis,
    _compute_gate_signal_from_irm,
    estimate_gate_from_irm,
)
from causalis.scenarios.unconfoundedness.model import IRM


def _make_synthetic_data(n: int = 500, seed: int = 7) -> tuple[CausalData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    d = rng.binomial(1, p)
    tau = 1.2 + 0.5 * (x1 > 0.0).astype(float)
    y = 1.0 + 0.4 * x1 - 0.2 * x2 + tau * d + rng.normal(scale=1.0, size=n)
    user_id = pd.Index([f"u_{i:04d}" for i in range(n)], name="user_id")

    df = pd.DataFrame({"user_id": user_id, "y": y, "d": d, "x1": x1, "x2": x2})
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"], user_id="user_id")
    return cd, df


def _fit_irm(
    cd: CausalData,
    *,
    normalize_ipw: bool = False,
    random_state: int = 17,
    store_diagnostics: bool = True,
) -> IRM:
    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000, solver="lbfgs"),
        n_folds=4,
        normalize_ipw=normalize_ipw,
        trimming_threshold=1e-3,
        random_state=random_state,
    )
    irm.fit(store_diagnostics=store_diagnostics)
    return irm


def _make_mock_gate_irm(n: int = 20000, k: int = 64, seed: int = 77):
    rng = np.random.default_rng(seed)
    user_id = pd.Index([f"u_{i:06d}" for i in range(n)], name="user_id")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    m_hat = 1.0 / (1.0 + np.exp(-(0.4 * x1 - 0.3 * x2)))
    d = rng.binomial(1, m_hat)
    tau = 0.7 + 0.2 * (x1 > 0.0).astype(float)
    g0_hat = 0.3 + 0.4 * x1 - 0.2 * x2
    g1_hat = g0_hat + tau
    y = g0_hat + tau * d + rng.normal(scale=1.0, size=n)

    df = pd.DataFrame({"user_id": user_id, "y": y, "d": d, "x1": x1, "x2": x2})
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"], user_id="user_id")
    groups = pd.Series(np.arange(n) % k, index=user_id, name="segment")

    class _MockIRM(BaseEstimator):
        def __init__(self):
            self.data = cd
            self._y = np.asarray(y, dtype=float)
            self._d = np.asarray(d, dtype=int)
            self.g0_hat_ = np.asarray(g0_hat, dtype=float)
            self.g1_hat_ = np.asarray(g1_hat, dtype=float)
            self.m_hat_ = np.asarray(m_hat, dtype=float)
            self._fit_index_ = pd.Index(cd.user_id.copy(), name=cd.user_id_name)
            self._fit_row_index_ = cd.df.index.copy()
            self.store_diagnostics = True
            self.trimming_threshold = 1e-3
            self.n_folds = 2
            self.random_state = seed

        def fit(self):
            return self

        def _use_normalized_ipw(self, score: str = "ATE", warn: bool = False) -> bool:
            return False

        def _resolve_estimation_targets(self):
            return self._y, self._d

    return _MockIRM(), groups


def _groups_from_df(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(df["x1"] >= 0.0, "high_x1", "low_x1"),
        index=pd.Index(df["user_id"], name="user_id"),
        name="segment",
    )


def _groups_from_row_index(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["x1"] >= 0.0, "high_x1", "low_x1"), index=df.index, name="segment")


def _quantile_groups_from_df(df: pd.DataFrame, q: int = 4) -> pd.Series:
    groups = pd.qcut(df["x1"], q=q, duplicates="drop").rename("x1_bin")
    groups.index = pd.Index(df["user_id"], name="user_id")
    return groups


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float).reshape(-1)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    valid_p = p_values[valid]
    if valid_p.size == 0:
        return adjusted

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    m = sorted_p.size
    holm_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    holm_sorted = np.minimum(holm_sorted, 1.0)
    holm_adjusted = np.empty_like(sorted_p)
    holm_adjusted[order] = holm_sorted
    adjusted[valid] = holm_adjusted
    return adjusted


def test_irm_estimate_gate_dispatches_to_new_module(monkeypatch):
    cd, df = _make_synthetic_data(n=180, seed=1)
    irm = _fit_irm(cd, store_diagnostics=False)
    groups = _groups_from_df(df)
    marker = object()
    captured = {}

    def _fake_gate_estimator(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr("causalis.scenarios.unconfoundedness.model.estimate_gate_from_irm", _fake_gate_estimator)

    res = irm.estimate(
        score="GATE",
        groups=groups,
        alpha=0.1,
        cov_type="HC2",
        cov_kwds={"use_correction": True},
    )
    assert res is marker
    assert captured["irm_model"] is irm
    assert captured["groups"] is groups
    assert captured["alpha"] == 0.1
    assert captured["cov_type"] == "HC2"
    assert captured["cov_kwds"] == {"use_correction": True}
    assert captured["irm_model"].store_diagnostics is False


def test_gate_basis_handling_inputs():
    cd, df = _make_synthetic_data(n=260, seed=2)
    irm = _fit_irm(cd)
    groups_series = _groups_from_df(df)
    groups_row = _groups_from_row_index(df)
    groups_df_1col = groups_series.to_frame()
    groups_dummy = pd.DataFrame(
        {
            "g_high": (groups_series == "high_x1").astype(int),
            "g_low": (groups_series == "low_x1").astype(int),
        }
    )

    res_series = irm.estimate(score="GATE", groups=groups_series)
    res_row = irm.estimate(score="GATE", groups=groups_row)
    res_df = irm.estimate(score="GATE", groups=groups_df_1col)
    res_dummy = irm.estimate(score="GATE", groups=groups_dummy)

    assert isinstance(res_series, GateEstimate)
    assert isinstance(res_row, GateEstimate)
    assert isinstance(res_df, GateEstimate)
    assert isinstance(res_dummy, GateEstimate)
    assert sorted(res_series.group_names) == ["segment=high_x1", "segment=low_x1"]
    np.testing.assert_allclose(np.sort(res_series.values), np.sort(res_row.values), atol=1e-10)
    np.testing.assert_allclose(np.sort(res_series.values), np.sort(res_df.values), atol=1e-10)

    overlap = groups_dummy.copy()
    overlap.iloc[:20, :] = 1
    with pytest.raises(ValueError, match="mutually exclusive and exhaustive"):
        irm.estimate(score="GATE", groups=overlap)

    non_exhaustive = groups_dummy.copy()
    non_exhaustive.iloc[:20, :] = 0
    with pytest.raises(ValueError, match="mutually exclusive and exhaustive"):
        irm.estimate(score="GATE", groups=non_exhaustive)


def test_gate_matches_manual_group_means_of_phi():
    cd, df = _make_synthetic_data(n=320, seed=3)
    irm = _fit_irm(cd, normalize_ipw=True)
    groups = _groups_from_df(df)

    with pytest.warns(RuntimeWarning, match="ignored for GATE"):
        res = irm.estimate(score="GATE", groups=groups, cov_type="HC3")

    assert res.model_options["normalize_ipw_requested"] is True
    assert res.model_options["normalize_ipw_effective"] is False
    assert res.model_options["se_approx_hajek"] is False

    phi, _, _ = _compute_gate_signal_from_irm(irm)
    basis = _coerce_groups_to_basis(groups, n_obs=phi.shape[0])
    basis_np = basis.to_numpy(dtype=float)
    manual = (basis_np.T @ phi) / basis_np.sum(axis=0)

    np.testing.assert_allclose(res.values, manual, atol=1e-10)


def test_gate_inference_covariance_options_and_se_validity():
    cd, df = _make_synthetic_data(n=340, seed=4)
    irm = _fit_irm(cd)
    groups = _groups_from_df(df)

    res_default = irm.estimate(score="GATE", groups=groups)
    assert res_default.model_options["cov_type"] == "HC3"
    assert np.all(np.isfinite(res_default.std_errors))
    assert np.all(res_default.std_errors >= 0.0)

    res_hc1 = irm.estimate(score="GATE", groups=groups, cov_type="HC1")
    assert res_hc1.model_options["cov_type"] == "HC1"
    assert np.all(np.isfinite(res_hc1.std_errors))
    assert np.all(res_hc1.std_errors >= 0.0)


def test_gate_matches_statsmodels_no_intercept_ols_for_hcx_covariances():
    cd, df = _make_synthetic_data(n=320, seed=4)
    irm = _fit_irm(cd, random_state=19)
    groups = _quantile_groups_from_df(df, q=4)

    phi, _, _ = _compute_gate_signal_from_irm(irm)
    basis = _coerce_groups_to_basis(groups, n_obs=phi.shape[0])
    design = basis.to_numpy(dtype=float)

    for cov_type in ("HC0", "HC1", "HC2", "HC3"):
        res = irm.estimate(score="GATE", groups=groups, cov_type=cov_type)
        fit = OLS(phi, design).fit(cov_type=cov_type)

        np.testing.assert_allclose(res.values, np.asarray(fit.params, dtype=float), atol=1e-10)
        np.testing.assert_allclose(
            res.covariance.to_numpy(dtype=float),
            np.asarray(fit.cov_params(), dtype=float),
            atol=1e-10,
        )


def test_gate_summary_includes_is_significant_column():
    cd, df = _make_synthetic_data(n=300, seed=40)
    irm = _fit_irm(cd)
    groups = _groups_from_df(df)

    res = irm.estimate(score="GATE", groups=groups)
    summary = res.summary()

    assert list(summary.columns[:9]) == [
        "group",
        "value",
        "is_significant",
        "ci_lower",
        "ci_upper",
        "n_group",
        "n_treated",
        "n_control",
        "share_treated",
    ]
    assert "wald_stat" not in summary.columns
    np.testing.assert_array_equal(summary["is_significant"].to_numpy(dtype=bool), res.p_values < res.alpha)
    np.testing.assert_array_equal(res.summary_table["is_significant"].to_numpy(dtype=bool), res.p_values < res.alpha)


def test_gate_cov_kwds_are_recorded_as_ignored_request():
    cd, df = _make_synthetic_data(n=240, seed=24)
    irm = _fit_irm(cd)
    groups = _groups_from_df(df)

    with pytest.warns(RuntimeWarning, match="cov_kwds are ignored for GATE"):
        res = irm.estimate(score="GATE", groups=groups, cov_kwds={"use_correction": True})

    assert res.model_options["cov_kwds"] == {}
    assert res.model_options["cov_kwds_requested"] == {"use_correction": True}


def test_gate_payload_follows_fit_store_diagnostics_setting():
    cd, df = _make_synthetic_data(n=220, seed=14)
    groups = _groups_from_df(df)

    irm_diag = _fit_irm(cd, store_diagnostics=True)
    irm_light = _fit_irm(cd, store_diagnostics=False)

    diagnostic_payload = irm_diag.estimate(score="GATE", groups=groups).diagnostic_data
    assert diagnostic_payload is not None
    assert "basis" not in diagnostic_payload
    assert "group_codes" in diagnostic_payload
    assert irm_light.estimate(score="GATE", groups=groups).diagnostic_data is None


def test_gate_large_label_path_uses_compact_diagnostics():
    irm, groups = _make_mock_gate_irm(n=20000, k=64, seed=81)

    res = estimate_gate_from_irm(irm, groups=groups, cov_type="HC2")

    assert res.diagnostic_data is not None
    assert "basis" not in res.diagnostic_data
    assert res.diagnostic_data["group_codes"].shape == (20000,)
    assert np.issubdtype(res.diagnostic_data["group_codes"].dtype, np.integer)
    assert res.diagnostic_data["group_names"] == res.group_names
    assert len(np.unique(res.diagnostic_data["group_codes"])) == 64


def test_gate_lightweight_mode_uses_cached_targets_without_reloading_dataframe(monkeypatch):
    cd, df = _make_synthetic_data(n=200, seed=12)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, store_diagnostics=False)

    def _unexpected_get_df(*args, **kwargs):
        raise AssertionError("GATE unexpectedly reloaded the dataframe")

    monkeypatch.setattr(CausalData, "get_df", _unexpected_get_df)
    res = irm.estimate(score="GATE", groups=groups)

    assert isinstance(res, GateEstimate)


def test_gate_lightweight_mode_matches_full_mode_on_unchanged_data():
    cd, df = _make_synthetic_data(n=260, seed=18)
    groups = _groups_from_df(df)

    res_full = _fit_irm(cd, store_diagnostics=True).estimate(score="GATE", groups=groups)
    res_light = _fit_irm(cd, store_diagnostics=False).estimate(score="GATE", groups=groups)

    np.testing.assert_allclose(res_light.values, res_full.values, atol=1e-12)


def test_gate_lightweight_mode_is_stable_after_data_reordering():
    cd, df = _make_synthetic_data(n=240, seed=22)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, store_diagnostics=False)

    first = irm.estimate(score="GATE", groups=groups)
    cd.df = cd.df.sample(frac=1.0, random_state=31).reset_index(drop=True)
    second = irm.estimate(score="GATE", groups=groups)

    np.testing.assert_allclose(second.values, first.values, atol=1e-12)


def test_gate_reorders_groups_to_fit_time_user_ids():
    cd, df = _make_synthetic_data(n=260, seed=26)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, store_diagnostics=False)

    first = irm.estimate(score="GATE", groups=groups)
    shuffled_groups = groups.sample(frac=1.0, random_state=41)
    second = irm.estimate(score="GATE", groups=shuffled_groups)

    np.testing.assert_allclose(second.values, first.values, atol=1e-12)
    np.testing.assert_allclose(second.std_errors, first.std_errors, atol=1e-12)


def test_gate_accepts_explicit_sequential_integer_user_ids():
    rng = np.random.default_rng(52)
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    d = rng.binomial(1, p)
    tau = 1.2 + 0.5 * (x1 > 0.0).astype(float)
    y = 1.0 + 0.4 * x1 - 0.2 * x2 + tau * d + rng.normal(scale=1.0, size=n)
    user_id = np.arange(n)

    df = pd.DataFrame({"user_id": user_id, "y": y, "d": d, "x1": x1, "x2": x2})
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"], user_id="user_id")
    irm = _fit_irm(cd, store_diagnostics=False)
    groups = pd.Series(
        np.where(df["x1"] >= 0.0, "high_x1", "low_x1"),
        index=pd.Index(df["user_id"], name="user_id"),
        name="segment",
    )

    res = irm.estimate(score="GATE", groups=groups)

    assert isinstance(res, GateEstimate)
    assert sorted(res.group_names) == ["segment=high_x1", "segment=low_x1"]
    assert np.all(np.isfinite(res.values))


def test_gate_row_index_groups_match_user_id_groups_for_qcut_notebook_pattern():
    cd, df = _make_synthetic_data(n=220, seed=27)
    irm = _fit_irm(cd, store_diagnostics=False)
    groups_row = pd.qcut(df["x1"], q=5, duplicates="drop")
    groups_user_id = groups_row.copy()
    groups_user_id.index = pd.Index(df["user_id"], name="user_id")

    res_row = irm.estimate(score="GATE", groups=groups_row)
    res_user_id = irm.estimate(score="GATE", groups=groups_user_id)

    np.testing.assert_allclose(res_row.values, res_user_id.values, atol=1e-12)
    np.testing.assert_allclose(res_row.std_errors, res_user_id.std_errors, atol=1e-12)


def test_gate_rejects_row_index_groups_after_row_to_user_id_mapping_changes():
    cd, df = _make_synthetic_data(n=220, seed=27)
    irm = _fit_irm(cd, store_diagnostics=False)
    cd.df = cd.df.sample(frac=1.0, random_state=43).reset_index(drop=True)
    groups_row = pd.qcut(cd.df["x1"], q=5, duplicates="drop")

    with pytest.raises(ValueError, match="row-to-id mapping remains unchanged"):
        irm.estimate(score="GATE", groups=groups_row)


def test_gate_requires_fitted_model():
    cd, df = _make_synthetic_data(n=120, seed=5)
    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000, solver="lbfgs"),
        n_folds=3,
        random_state=11,
    )
    groups = _groups_from_df(df)

    with pytest.raises(NotFittedError):
        irm.estimate(score="GATE", groups=groups)


def test_gate_groups_fallback_uses_causaldata_attribute():
    cd, df = _make_synthetic_data(n=260, seed=6)
    groups = _groups_from_df(df)
    object.__setattr__(cd, "gate_groups", groups)
    irm = _fit_irm(cd)

    res = irm.estimate(score="GATE")
    assert isinstance(res, GateEstimate)
    assert len(res.group_names) == 2


def test_gate_requires_user_id_on_causaldata():
    rng = np.random.default_rng(28)
    n = 180
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    d = rng.binomial(1, p)
    tau = 1.2 + 0.5 * (x1 > 0.0).astype(float)
    y = 1.0 + 0.4 * x1 - 0.2 * x2 + tau * d + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2}, index=pd.Index([f"row_{i}" for i in range(n)]))
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])
    irm = _fit_irm(cd, store_diagnostics=False)
    groups = pd.Series(np.where(df["x1"] >= 0.0, "high_x1", "low_x1"), name="segment")

    with pytest.raises(ValueError, match="requires CausalData.user_id"):
        irm.estimate(score="GATE", groups=groups)


def test_gate_raises_when_groups_missing_and_no_fallback():
    cd, _ = _make_synthetic_data(n=180, seed=8)
    irm = _fit_irm(cd)

    with pytest.raises(ValueError, match="GATE requires pre-defined groups"):
        irm.estimate(score="GATE")


def test_gate_method_is_thin_wrapper_over_estimate():
    cd, df = _make_synthetic_data(n=220, seed=10)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, random_state=101)

    res_gate = irm.gate(groups=groups, alpha=0.1, cov_type="HC2")
    res_estimate = irm.estimate(score="GATE", groups=groups, alpha=0.1, cov_type="HC2")

    np.testing.assert_allclose(res_gate.values, res_estimate.values, atol=1e-12)
    np.testing.assert_allclose(res_gate.std_errors, res_estimate.std_errors, atol=1e-12)
    assert res_gate.model_options["cov_type"] == "HC2"


def test_gate_contrast_matches_manual_formula():
    cd, df = _make_synthetic_data(n=320, seed=41)
    irm = _fit_irm(cd, random_state=44)
    groups = _groups_from_df(df)
    res = irm.estimate(score="GATE", groups=groups, cov_type="HC2")

    contrast = res.contrast("segment=high_x1", "segment=low_x1")

    assert isinstance(contrast, GateContrastEstimate)
    left_idx = res.group_names.index("segment=high_x1")
    right_idx = res.group_names.index("segment=low_x1")
    contrast_vector = np.zeros(len(res.group_names), dtype=float)
    contrast_vector[left_idx] = 1.0
    contrast_vector[right_idx] = -1.0
    covariance = res.covariance.loc[res.group_names, res.group_names].to_numpy(dtype=float)

    manual_value = float(res.values[left_idx] - res.values[right_idx])
    manual_se = float(np.sqrt(contrast_vector @ covariance @ contrast_vector))
    manual_test = float(manual_value / manual_se)
    manual_p = float(2.0 * norm.sf(abs(manual_test)))
    z_crit = float(norm.ppf(1.0 - (contrast.alpha / 2.0)))

    assert contrast.contrast_label == "segment=high_x1 - segment=low_x1"
    assert contrast.left_value == pytest.approx(res.values[left_idx])
    assert contrast.right_value == pytest.approx(res.values[right_idx])
    assert contrast.n_left == int(res.n_group[left_idx])
    assert contrast.n_right == int(res.n_group[right_idx])
    assert contrast.value == pytest.approx(manual_value)
    assert contrast.std_error == pytest.approx(manual_se)
    assert contrast.test_stat == pytest.approx(manual_test)
    assert contrast.p_value == pytest.approx(manual_p)
    assert contrast.ci_lower == pytest.approx(manual_value - z_crit * manual_se)
    assert contrast.ci_upper == pytest.approx(manual_value + z_crit * manual_se)
    assert bool(contrast.summary().loc["is_significant", "value"]) == (contrast.p_value < contrast.alpha)


def test_gate_contrast_one_sided_alternatives_behave_correctly():
    cd, df = _make_synthetic_data(n=320, seed=42)
    irm = _fit_irm(cd, random_state=45)
    groups = _groups_from_df(df)
    res = irm.estimate(score="GATE", groups=groups, cov_type="HC3")

    two_sided = res.contrast("segment=high_x1", "segment=low_x1", alternative="two-sided")
    greater = res.contrast("segment=high_x1", "segment=low_x1", alternative="greater")
    less = res.contrast("segment=high_x1", "segment=low_x1", alternative="less")

    assert two_sided.value > 0.0
    assert greater.ci_lower is None
    assert greater.ci_upper is None
    assert less.ci_lower is None
    assert less.ci_upper is None
    assert greater.p_value == pytest.approx(norm.sf(two_sided.test_stat))
    assert less.p_value == pytest.approx(norm.cdf(two_sided.test_stat))
    assert greater.p_value < two_sided.p_value
    assert less.p_value > 0.5


def test_gate_pairwise_summary_supports_reference_and_p_adjustments():
    cd, df = _make_synthetic_data(n=360, seed=43)
    irm = _fit_irm(cd, random_state=46)
    groups = _quantile_groups_from_df(df, q=4)
    res = irm.estimate(score="GATE", groups=groups, cov_type="HC2")
    k = len(res.group_names)

    pairwise_none = res.pairwise_summary()
    assert pairwise_none.shape[0] == k * (k - 1) // 2
    assert list(pairwise_none.columns) == [
        "left_group",
        "right_group",
        "contrast_label",
        "left_value",
        "right_value",
        "estimate_diff",
        "std_error",
        "test_stat",
        "p_value",
        "p_value_adj",
        "ci_lower",
        "ci_upper",
        "is_significant",
        "is_significant_adj",
        "n_left",
        "n_right",
        "alpha",
        "p_adjust",
    ]
    np.testing.assert_allclose(
        pairwise_none["p_value_adj"].to_numpy(dtype=float),
        pairwise_none["p_value"].to_numpy(dtype=float),
        equal_nan=True,
    )

    reference = res.group_names[-1]
    pairwise_ref = res.pairwise_summary(reference=reference)
    assert pairwise_ref.shape[0] == k - 1
    assert set(pairwise_ref["right_group"]) == {reference}
    for _, row in pairwise_ref.iterrows():
        assert row["contrast_label"] == f"{row['left_group']} - {row['right_group']}"
        assert row["estimate_diff"] == pytest.approx(row["left_value"] - row["right_value"])

    pairwise_bonferroni = res.pairwise_summary(p_adjust="bonferroni")
    m = pairwise_bonferroni.shape[0]
    np.testing.assert_allclose(
        pairwise_bonferroni["p_value_adj"].to_numpy(dtype=float),
        np.minimum(pairwise_none["p_value"].to_numpy(dtype=float) * m, 1.0),
        equal_nan=True,
    )

    pairwise_holm = res.pairwise_summary(p_adjust="holm")
    expected_holm = _holm_adjust(pairwise_none["p_value"].to_numpy(dtype=float))
    np.testing.assert_allclose(
        pairwise_holm["p_value_adj"].to_numpy(dtype=float),
        expected_holm,
        equal_nan=True,
    )
    assert np.all(pairwise_holm["p_value_adj"].to_numpy(dtype=float) >= pairwise_none["p_value"].to_numpy(dtype=float))


def test_gate_contrast_and_pairwise_validation_errors():
    cd, df = _make_synthetic_data(n=280, seed=44)
    irm = _fit_irm(cd, random_state=47)
    groups = _groups_from_df(df)
    res = irm.estimate(score="GATE", groups=groups)

    with pytest.raises(ValueError, match="Unknown left_group"):
        res.contrast("missing_group", "segment=low_x1")
    with pytest.raises(ValueError, match="Unknown right_group"):
        res.contrast("segment=high_x1", "missing_group")
    with pytest.raises(ValueError, match="must be different"):
        res.contrast("segment=high_x1", "segment=high_x1")
    with pytest.raises(ValueError, match="alternative must be one of"):
        res.contrast("segment=high_x1", "segment=low_x1", alternative="up")
    with pytest.raises(ValueError, match="Unknown reference"):
        res.pairwise_summary(reference="missing_group")
    with pytest.raises(ValueError, match="p_adjust must be one of"):
        res.pairwise_summary(p_adjust="bh")


def test_gate_pairwise_summary_requires_at_least_two_groups():
    cd, df = _make_synthetic_data(n=260, seed=45)
    irm = _fit_irm(cd, random_state=48)
    groups = pd.Series("all_users", index=pd.Index(df["user_id"], name="user_id"), name="segment")
    res = irm.estimate(score="GATE", groups=groups)

    assert res.group_names == ["segment=all_users"]
    with pytest.raises(ValueError, match="at least two estimable GATE groups"):
        res.pairwise_summary()


def test_gate_deterministic_with_fixed_random_state():
    cd1, df1 = _make_synthetic_data(n=300, seed=9)
    groups1 = _groups_from_df(df1)
    irm1 = _fit_irm(cd1, random_state=123)
    res1 = irm1.estimate(score="GATE", groups=groups1, cov_type="HC2")

    cd2, df2 = _make_synthetic_data(n=300, seed=9)
    groups2 = _groups_from_df(df2)
    irm2 = _fit_irm(cd2, random_state=123)
    res2 = irm2.estimate(score="GATE", groups=groups2, cov_type="HC2")

    np.testing.assert_allclose(res1.values, res2.values, atol=1e-12)
    np.testing.assert_allclose(res1.std_errors, res2.std_errors, atol=1e-12)



def test_gate_raises_for_groups_without_treated_and_control_support():
    cd, df = _make_synthetic_data(n=240, seed=15)
    irm = _fit_irm(cd, random_state=33)
    groups = pd.Series(df["d"].to_numpy(dtype=float), index=pd.Index(df["user_id"], name="user_id"), name="d")

    irm.m_hat_ = np.where(irm._d == 1.0, 0.965, 0.010)

    with pytest.raises(
        ValueError,
        match="Each GATE group must contain at least one treated and one control observation",
    ):
        irm.estimate(score="GATE", groups=groups, cov_type="HC3")


def test_gate_signal_non_finite_values_raise_runtime_error():
    cd, _ = _make_synthetic_data(n=180, seed=14)
    irm = _fit_irm(cd, random_state=91)

    irm.m_hat_ = np.ones_like(irm.m_hat_)
    with pytest.raises(RuntimeError, match="non-finite values"):
        _compute_gate_signal_from_irm(irm)
