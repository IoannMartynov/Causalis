import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from causalis.dgp.causaldata import CausalData
from causalis.dgp import generate_rct
from causalis.scenarios.unconfoundedness.model import IRM


class _FeaturePropensityClassifier(BaseEstimator, ClassifierMixin):
    """Classifier that returns the first feature as P(D=1)."""

    _estimator_type = "classifier"

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float)[:, 0]
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def make_causal_data(n=1000, outcome_type="normal", random_state=1):
    df = generate_rct(n=n, split=0.5, random_state=random_state, outcome_type=outcome_type, k=3, add_ancillary=False)
    # map to expected columns: outcome y, treatment t, confounders any x*
    y = "y"; d = "d"
    xcols = [c for c in df.columns if c.startswith("x")]
    cd = CausalData(df=df[[y, d] + xcols], treatment=d, outcome=y, confounders=xcols)
    return cd


def make_overlap_policy_data():
    p = np.array(
        [0.02, 0.08, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92, 0.98] * 8,
        dtype=float,
    )
    n = p.size
    d = np.tile([0, 1], n // 2)
    y = 1.0 + 2.0 * d + 0.5 * p + np.linspace(-0.2, 0.2, n)
    df = pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x_propensity": p,
            "x_trend": np.linspace(-1.0, 1.0, n),
        }
    )
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=["x_propensity", "x_trend"],
    )


def test_irm_ate_runs_and_shapes():
    cd = make_causal_data(n=800, outcome_type="normal", random_state=42)
    ml_g = RandomForestRegressor(n_estimators=50, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=50, random_state=42)

    est = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, random_state=123)
    est.fit().estimate(score="ATE")

    assert est.coef.shape == (1,)
    assert est.se.shape == (1,)
    assert np.isfinite(est.se[0])
    ci = est.confint()
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape == (1, 2)


def test_irm_atte_runs():
    cd = make_causal_data(n=600, outcome_type="normal", random_state=7)
    ml_g = RandomForestRegressor(n_estimators=40, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=40, random_state=0)

    est = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, random_state=1)
    est.fit().estimate(score="ATTE")
    assert np.isfinite(est.coef[0])


def test_irm_binary_outcome_with_classifier():
    cd = make_causal_data(n=800, outcome_type="binary", random_state=21)
    ml_g = RandomForestClassifier(n_estimators=60, random_state=21)
    ml_m = RandomForestClassifier(n_estimators=60, random_state=21)

    est = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, random_state=21)
    est.fit().estimate(score="ATE")
    assert np.isfinite(est.se[0])


def test_irm_raises_on_non_binary_treatment():
    cd = make_causal_data(n=300, outcome_type="normal", random_state=3)
    # Modify treatment to be non-binary
    df = cd.df.copy()
    df[cd.treatment.name] = df[cd.treatment.name].replace({1: 2})

    with pytest.raises(ValueError, match="binary encoded"):
        CausalData(df=df, treatment=cd.treatment.name, outcome=cd.outcome.name, confounders=cd.confounders)


def test_irm_raises_early_when_n_folds_exceeds_minority_class_size():
    cd = make_causal_data(n=24, outcome_type="normal", random_state=9)
    df = cd.df.copy()
    df[cd.treatment.name] = np.array([1] * 3 + [0] * (len(df) - 3))
    cd_small = CausalData(df=df, treatment=cd.treatment.name, outcome=cd.outcome.name, confounders=cd.confounders)

    ml_g = RandomForestRegressor(n_estimators=10, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=10, random_state=0)

    with pytest.raises(ValueError, match="minimum treatment class count=3"):
        IRM(cd_small, ml_g=ml_g, ml_m=ml_m, n_folds=4).fit()


def test_irm_overlap_policy_clip_bounds_propensity_without_dropping():
    cd = make_overlap_policy_data()
    est = IRM(
        cd,
        ml_g=LinearRegression(),
        ml_m=_FeaturePropensityClassifier(),
        n_folds=3,
        overlap_policy="clip",
        overlap_threshold=0.10,
        random_state=12,
    ).fit()

    assert est.overlap_n_dropped_ == 0
    assert np.all(est.overlap_mask_)
    assert len(est.m_hat_) == len(cd.df)
    assert np.min(est.m_hat_) >= 0.10
    assert np.max(est.m_hat_) <= 0.90
    assert np.min(est.m_hat_raw_) < 0.10
    assert np.max(est.m_hat_raw_) > 0.90
    assert len(est._full_sample_folds_) == len(cd.df)
    assert np.array_equal(est._full_sample_folds_, est.folds_)


def test_irm_overlap_policy_drop_filters_estimation_and_diagnostics():
    cd = make_overlap_policy_data()
    est = IRM(
        cd,
        ml_g=LinearRegression(),
        ml_m=_FeaturePropensityClassifier(),
        n_folds=3,
        overlap_policy="drop",
        overlap_threshold=0.10,
        random_state=12,
    ).fit()
    result = est.estimate(score="ATE")

    expected_mask = (cd.df["x_propensity"].to_numpy() > 0.10) & (
        cd.df["x_propensity"].to_numpy() < 0.90
    )
    expected_n = int(np.sum(expected_mask))

    assert np.array_equal(est.overlap_mask_, expected_mask)
    assert est.overlap_n_dropped_ == len(cd.df) - expected_n
    assert len(est.m_hat_) == expected_n
    assert np.all(est.m_hat_ > 0.10)
    assert np.all(est.m_hat_ < 0.90)
    assert len(est.g0_hat_) == expected_n
    assert len(est.g1_hat_) == expected_n
    assert len(est._y) == expected_n
    assert len(est._d) == expected_n
    assert result.n_treated == int(np.sum(cd.df.loc[expected_mask, "d"] == 1))
    assert result.n_control == int(np.sum(cd.df.loc[expected_mask, "d"] == 0))
    assert result.model_options["overlap_policy"] == "drop"
    assert result.model_options["overlap_threshold"] == 0.10
    assert result.diagnostic_data is not None
    assert len(result.diagnostic_data.m_hat) == expected_n
    assert len(result.diagnostic_data.y) == expected_n
    assert len(result.diagnostic_data.d) == expected_n
    assert len(est._full_sample_folds_) == len(cd.df)
    assert len(est.folds_) == expected_n


def test_irm_preserves_full_sample_folds_without_diagnostics():
    cd = make_overlap_policy_data()
    est = IRM(
        cd,
        ml_g=LinearRegression(),
        ml_m=_FeaturePropensityClassifier(),
        n_folds=3,
        overlap_policy="drop",
        overlap_threshold=0.10,
        random_state=12,
        store_diagnostics=False,
    ).fit()

    assert est.folds_ is None
    assert len(est._full_sample_folds_) == len(cd.df)
    assert len(est.m_hat_) < len(est._full_sample_folds_)


def test_irm_rejects_invalid_overlap_config():
    cd = make_overlap_policy_data()
    with pytest.raises(ValueError, match="overlap_policy"):
        IRM(cd, ml_g=LinearRegression(), ml_m=_FeaturePropensityClassifier(), overlap_policy="truncate")
    with pytest.raises(ValueError, match="overlap_threshold"):
        IRM(cd, ml_g=LinearRegression(), ml_m=_FeaturePropensityClassifier(), overlap_threshold=0.5)


def test_irm_old_trimming_constructor_args_are_removed():
    cd = make_overlap_policy_data()
    with pytest.raises(TypeError):
        IRM(
            cd,
            ml_g=LinearRegression(),
            ml_m=_FeaturePropensityClassifier(),
            trimming_threshold=0.1,
        )
    with pytest.raises(TypeError):
        IRM(
            cd,
            ml_g=LinearRegression(),
            ml_m=_FeaturePropensityClassifier(),
            trimming_rule="truncate",
        )
