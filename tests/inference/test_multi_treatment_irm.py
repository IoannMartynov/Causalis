import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

import causalis.scenarios.multi_unconfoundedness.model as multi_irm_module
from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.model import MultiTreatmentIRM


def _make_multi_causal_data(n: int = 180, seed: int = 42) -> MultiCausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)

    labels = np.tile(np.array([0, 1, 2], dtype=int), int(np.ceil(n / 3)))[:n]
    rng.shuffle(labels)
    d = np.eye(3, dtype=int)[labels]

    effects = np.array([0.0, -0.5, 0.8], dtype=float)
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + effects[labels] + rng.normal(0.0, 0.1, size=n)

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "d0": d[:, 0],
            "d1": d[:, 1],
            "d2": d[:, 2],
        }
    )

    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1", "d2"],
        confounders=["x1", "x2"],
        control_treatment="d0",
    )


def _make_selection_multi_causal_data(n: int = 900, seed: int = 123) -> MultiCausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)

    logits = np.column_stack(
        [
            np.zeros(n, dtype=float),
            1.3 * x1 - 0.2 * x2,
            -1.0 * x1 + 0.4 * x2,
        ]
    )
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    labels = np.array([rng.choice(3, p=p_i) for p_i in probs], dtype=int)
    d = np.eye(3, dtype=int)[labels]

    y0 = 2.0 + 0.7 * x1 - 0.5 * x2 + rng.normal(0.0, 0.25, size=n)
    tau1 = 1.0 + 1.1 * x1
    tau2 = -0.4 + 0.8 * x1
    y = y0 + (labels == 1) * tau1 + (labels == 2) * tau2

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "d0": d[:, 0],
            "d1": d[:, 1],
            "d2": d[:, 2],
        }
    )
    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1", "d2"],
        confounders=["x1", "x2"],
        control_treatment="d0",
    )


def test_multi_treatment_irm_returns_multicausal_estimate():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    )

    result = model.fit().estimate(alpha=0.05)

    assert isinstance(result, MultiCausalEstimate)
    assert result.value.shape == (2,)
    assert result.p_value.shape == (2,)
    assert result.n_control == int(np.sum(data.get_df()["d0"].to_numpy() == 1))
    assert result.n_treated == int(np.sum(data.get_df()[["d1", "d2"]].to_numpy() == 1))
    assert result.contrast_labels == ["d1 vs d0", "d2 vs d0"]
    assert np.array_equal(
        result.n_treated_by_arm,
        np.array(
            [
                int(np.sum(data.get_df()["d1"].to_numpy() == 1)),
                int(np.sum(data.get_df()["d2"].to_numpy() == 1)),
            ]
        ),
    )
    assert result.diagnostic_data is not None
    assert result.diagnostic_data.m_hat_raw is not None
    assert result.diagnostic_data.m_hat_raw.shape == result.diagnostic_data.m_hat.shape


def test_multi_treatment_irm_summary_uses_causalestimate_style_with_one_column_per_contrast():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    )
    result = model.fit().estimate(alpha=0.05)
    summary = result.summary()

    assert summary.index.name == "field"
    assert summary.columns.tolist() == ["d1 vs d0", "d2 vs d0"]
    assert summary.index.tolist() == [
        "estimand",
        "model",
        "value",
        "value_relative",
        "alpha",
        "p_value",
        "is_significant",
        "n_treated",
        "n_control",
        "treatment_mean",
        "control_mean",
        "time",
    ]
    assert "ci_abs:" in summary.loc["value", "d1 vs d0"]
    assert "ci_abs:" in summary.loc["value", "d2 vs d0"]
    assert summary.loc["n_treated", "d1 vs d0"] == int(np.sum(data.get_df()["d1"].to_numpy() == 1))
    assert summary.loc["n_treated", "d2 vs d0"] == int(np.sum(data.get_df()["d2"].to_numpy() == 1))
    assert summary.loc["n_control", "d1 vs d0"] == int(np.sum(data.get_df()["d0"].to_numpy() == 1))
    assert summary.loc["n_control", "d2 vs d0"] == int(np.sum(data.get_df()["d0"].to_numpy() == 1))


def test_multi_treatment_irm_fit_requires_multicausaldata():
    model = MultiTreatmentIRM(
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=2,
    )
    with pytest.raises(TypeError, match="MultiCausalData"):
        model.fit(data="not_multicausaldata")


def test_multi_treatment_irm_enforces_probabilistic_classifier_for_ml_m():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=DummyRegressor(strategy="mean"),
        n_folds=2,
    )
    with pytest.raises(ValueError, match="ml_m must be a classifier"):
        model.fit()


def test_multi_treatment_irm_validates_alpha():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    ).fit()

    with pytest.raises(ValueError, match="alpha must be in"):
        model.estimate(alpha=1.0)


def test_multi_treatment_irm_rejects_too_many_folds():
    data = _make_multi_causal_data(n=9, seed=7)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=4,
        random_state=1,
    )
    with pytest.raises(ValueError, match="minimum treatment class count"):
        model.fit()


def test_multi_treatment_irm_handles_single_class_binary_outcome_arm_fold():
    n = 90
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(1.0, -1.0, n)
    labels = np.tile(np.array([0, 1, 2], dtype=int), n // 3)
    d = np.eye(3, dtype=int)[labels]

    # Binary outcome with one treatment arm always zero -> per-arm folds can be single-class.
    y = np.zeros(n, dtype=int)
    y[labels == 2] = 1
    # Break exact equality with treatment columns while keeping arm-level degeneracy.
    y[0] = 1

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "d0": d[:, 0],
            "d1": d[:, 1],
            "d2": d[:, 2],
        }
    )
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1", "d2"],
        confounders=["x1", "x2"],
        control_treatment="d0",
    )

    model = MultiTreatmentIRM(
        data=data,
        ml_g=LogisticRegression(max_iter=1000),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=0,
    )
    result = model.fit().estimate()
    assert isinstance(result, MultiCausalEstimate)
    assert np.all(np.isfinite(model.g_hat_))


def test_multi_treatment_irm_trimmed_propensity_rows_sum_to_one():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    ).fit()

    row_sums = model.m_hat_.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10, rtol=0.0)
    assert np.all(model.m_hat_ >= model.trimming_threshold - 1e-12)


def test_multi_treatment_irm_atte_score_matches_closed_form_and_disables_hajek():
    model = MultiTreatmentIRM(normalize_ipw=True)
    model.score = "ATTE"

    y = np.array([1.2, 2.1, 0.9, 1.7], dtype=float)
    d = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )
    g_hat = np.array(
        [
            [1.0, 1.5, 0.8],
            [1.1, 2.0, 1.4],
            [0.7, 1.2, 1.0],
            [1.3, 1.8, 1.1],
        ],
        dtype=float,
    )
    m_hat = np.array(
        [
            [0.55, 0.25, 0.20],
            [0.30, 0.45, 0.25],
            [0.25, 0.20, 0.55],
            [0.28, 0.52, 0.20],
        ],
        dtype=float,
    )

    y_col, _, h, psi_a, psi_b = model._compute_score_terms(
        y=y,
        d=d,
        g_hat=g_hat,
        m_hat=m_hat,
        score="ATTE",
    )

    g0_hat = g_hat[:, [0]]
    residual0 = y_col - g0_hat
    d0 = d[:, [0]].astype(float)
    dk = d[:, 1:].astype(float)
    pk = dk.mean(axis=0)
    ratio = m_hat[:, 1:] / m_hat[:, [0]]
    expected_psi_b = (dk / pk[None, :]) * residual0 - (d0 / pk[None, :]) * ratio * residual0

    np.testing.assert_allclose(psi_b, expected_psi_b, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(h, d / m_hat, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(psi_a, -np.ones(y.shape[0], dtype=float), atol=0.0, rtol=0.0)


def test_multi_treatment_irm_atte_api_uses_single_score():
    data = _make_selection_multi_causal_data(seed=202)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        random_state=23,
    ).fit()

    result = model.estimate(score="ATTE", diagnostic_data=False)

    assert result.estimand == "ATTE"
    assert np.all(np.isfinite(result.value))
    assert np.all(np.isfinite(result.value_relative))


def test_multi_treatment_irm_rejects_legacy_atte_variant_kwarg():
    data = _make_multi_causal_data(seed=88)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=5,
    ).fit()

    with pytest.raises(TypeError, match="atte_variant"):
        model.estimate(score="ATE", atte_variant="dr")


def test_multi_treatment_irm_atte_differs_from_ate_under_selection_on_x():
    data = _make_selection_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        random_state=7,
    ).fit()

    ate = model.estimate(score="ATE", diagnostic_data=False)
    atte = model.estimate(score="ATTE", diagnostic_data=False)

    assert ate.estimand == "ATE"
    assert atte.estimand == "ATTE"
    assert ate.value.shape == (2,)
    assert atte.value.shape == (2,)
    assert np.all(np.isfinite(ate.value))
    assert np.all(np.isfinite(atte.value))
    assert np.max(np.abs(ate.value - atte.value)) > 0.1


def test_multi_treatment_irm_ate_is_stable_after_atte_calls():
    data = _make_selection_multi_causal_data(seed=222)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        random_state=13,
    ).fit()

    ate_before = model.estimate(score="ATE", diagnostic_data=False)
    _ = model.estimate(score="ATTE", diagnostic_data=False)
    ate_after = model.estimate(score="ATE", diagnostic_data=False)

    np.testing.assert_allclose(ate_before.value, ate_after.value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ate_before.ci_lower_absolute, ate_after.ci_lower_absolute, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ate_before.ci_upper_absolute, ate_after.ci_upper_absolute, rtol=1e-10, atol=1e-12)


def test_multi_treatment_irm_atte_relative_outputs_use_per_arm_control_mean():
    data = _make_selection_multi_causal_data(seed=321)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        random_state=11,
    ).fit()

    result = model.estimate(score="ATTE")

    assert result.control_mean_by_arm is not None
    assert result.control_mean_by_arm.shape == (2,)
    np.testing.assert_allclose(
        result.value_relative,
        100.0 * result.value / result.control_mean_by_arm,
        rtol=1e-10,
        atol=1e-12,
    )

    summary = result.summary()
    for idx, contrast in enumerate(result.contrast_labels):
        assert summary.loc["control_mean", contrast] == f"{result.control_mean_by_arm[idx]:.4f}"


def test_multi_treatment_irm_atte_runs_on_small_synthetic_data():
    data = _make_multi_causal_data(n=210, seed=314)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        random_state=29,
    ).fit()

    atte = model.estimate(score="ATTE", diagnostic_data=False)

    assert atte.estimand == "ATTE"
    assert np.all(np.isfinite(atte.value))
    assert np.all(np.isfinite(atte.value_relative))


def test_multi_treatment_irm_atte_diagnostics_skip_ate_sensitivity_and_report_effective_ipw():
    data = _make_selection_multi_causal_data(seed=777)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        normalize_ipw=True,
        random_state=19,
    ).fit()

    result = model.estimate(score="ATTE", diagnostic_data=True)
    diag = result.diagnostic_data

    assert diag is not None
    assert result.model_options["normalize_ipw_requested"] is True
    assert result.model_options["normalize_ipw_effective"] is False
    assert result.model_options["normalize_ipw"] is False
    assert model.normalize_ipw_effective_ is False
    assert diag.score == "ATTE"
    assert diag.normalize_ipw is False
    assert diag.sigma2 is None
    assert diag.nu2 is None
    assert diag.psi_sigma2 is None
    assert diag.psi_nu2 is None
    assert diag.riesz_rep is None
    assert diag.m_alpha is None
    assert diag.psi is None


def test_multi_treatment_irm_rejects_trimming_threshold_above_one_over_k():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        trimming_threshold=0.34,  # K=3 -> must be < 1/3
    )
    with pytest.raises(ValueError, match="1/K"):
        model.fit()


def test_diagnostics_use_cached_x_without_reloading_dataframe(monkeypatch):
    data = _make_multi_causal_data(seed=55)
    model = MultiTreatmentIRM(
        data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit()

    def _unexpected_get_df(*args, **kwargs):
        raise AssertionError("estimate() unexpectedly reloaded the dataframe")

    monkeypatch.setattr(MultiCausalData, "get_df", _unexpected_get_df)
    result = model.estimate(score="ATE")

    assert result.diagnostic_data is not None
    np.testing.assert_allclose(result.diagnostic_data.x, model._X)


def test_fit_store_diagnostics_controls_payload_and_model_caches():
    data = _make_multi_causal_data(seed=17)
    model = MultiTreatmentIRM(
        data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    assert model._X is None
    assert model._y is not None
    assert model._d is not None
    assert model.m_hat_raw_ is None
    assert model.folds_ is None

    result = model.estimate(score="ATE")
    assert result.diagnostic_data is None
    assert np.all(np.isfinite(result.value))


def test_lightweight_mode_estimate_uses_cached_targets_without_reloading_dataframe(monkeypatch):
    data = _make_multi_causal_data(seed=29)
    model = MultiTreatmentIRM(
        data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    def _unexpected_get_df(*args, **kwargs):
        raise AssertionError("estimate() unexpectedly reloaded the dataframe")

    monkeypatch.setattr(MultiCausalData, "get_df", _unexpected_get_df)
    result = model.estimate(score="ATE")

    assert result.diagnostic_data is None
    assert np.all(np.isfinite(result.value))


def test_lightweight_mode_matches_full_mode_on_unchanged_data():
    data = _make_multi_causal_data(seed=41)
    kwargs = dict(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=13,
    )

    result_full = MultiTreatmentIRM(**kwargs).fit(store_diagnostics=True).estimate(score="ATE")
    result_light = MultiTreatmentIRM(**kwargs).fit(store_diagnostics=False).estimate(score="ATE")

    np.testing.assert_allclose(result_light.value, result_full.value, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        result_light.model_options["std_error"],
        result_full.model_options["std_error"],
        rtol=1e-12,
        atol=0.0,
    )


def test_lightweight_mode_estimate_is_stable_after_data_reordering():
    data = _make_multi_causal_data(seed=1234)
    model = MultiTreatmentIRM(
        data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    first = model.estimate(score="ATE")
    data.df = data.df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = model.estimate(score="ATE")

    np.testing.assert_allclose(second.value, first.value, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        second.model_options["std_error"],
        first.model_options["std_error"],
        rtol=1e-12,
        atol=0.0,
    )


def test_lightweight_mode_legacy_reload_fallback_rejects_changed_data():
    data = _make_multi_causal_data(seed=4321)
    model = MultiTreatmentIRM(
        data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    model._y = None
    model._d = None
    data.df = data.df.sample(frac=1.0, random_state=19).reset_index(drop=True)

    with pytest.raises(RuntimeError, match="changed after fit"):
        model.estimate(score="ATE")


def test_default_catboost_parallelism_is_reduced_only_for_internal_defaults(monkeypatch):
    class _FakeCatBoost(BaseEstimator):
        def __init__(
            self,
            *,
            thread_count: int = -1,
            verbose: bool = False,
            logging_level: str | None = None,
            allow_writing_files: bool = False,
            random_seed: int | None = None,
            loss_function: str | None = None,
        ):
            self.thread_count = thread_count
            self.verbose = verbose
            self.logging_level = logging_level
            self.allow_writing_files = allow_writing_files
            self.random_seed = random_seed
            self.loss_function = loss_function

    data = _make_multi_causal_data(seed=314)

    monkeypatch.setattr(multi_irm_module, "HAS_CATBOOST", True)
    monkeypatch.setattr(multi_irm_module, "CatBoostClassifier", _FakeCatBoost)
    monkeypatch.setattr(multi_irm_module, "CatBoostRegressor", _FakeCatBoost)

    model_defaults = MultiTreatmentIRM(data=data, n_jobs=2, random_state=11)
    model_defaults._configure_default_learner_parallelism()

    assert model_defaults._ml_g_is_default is True
    assert model_defaults._ml_m_is_default is True
    assert model_defaults.ml_g.get_params()["thread_count"] == 1
    assert model_defaults.ml_m.get_params()["thread_count"] == 1

    ml_g = _FakeCatBoost(thread_count=-1)
    ml_m = _FakeCatBoost(thread_count=-1)
    model_custom = MultiTreatmentIRM(
        data=data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_jobs=2,
        random_state=11,
    )
    model_custom._configure_default_learner_parallelism()

    assert model_custom._ml_g_is_default is False
    assert model_custom._ml_m_is_default is False
    assert model_custom.ml_g.get_params()["thread_count"] == -1
    assert model_custom.ml_m.get_params()["thread_count"] == -1
