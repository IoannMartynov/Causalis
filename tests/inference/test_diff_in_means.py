import pytest
import pandas as pd
import numpy as np
from causalis.data_contracts import CausalData
from causalis.scenarios.classic_rct.model import DiffInMeans


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "treatment": np.random.choice([0, 1], size=n),
            "outcome": np.random.normal(0, 1, size=n),
        }
    )
    # Add some effect
    df.loc[df["treatment"] == 1, "outcome"] += 0.5
    return CausalData(df=df, treatment="treatment", outcome="outcome")


@pytest.fixture
def binary_outcome_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "treatment": np.random.choice([0, 1], size=n),
            "outcome": np.random.choice([0, 1], size=n),
        }
    )
    return CausalData(df=df, treatment="treatment", outcome="outcome")


def test_diff_in_means_fit_effect_ttest(sample_data):
    model = DiffInMeans()
    model.fit(sample_data)
    result = model.estimate(method="ttest")
    assert hasattr(result, "p_value")
    assert hasattr(result, "value")
    assert isinstance(result.p_value, float)


def test_diff_in_means_fit_effect_welch_permutation_t_test(sample_data):
    model = DiffInMeans()
    model.fit(sample_data)
    result = model.estimate(method="welch_permutation_t_test", B=100, seed=1)
    assert hasattr(result, "p_value")
    assert hasattr(result, "value")


def test_diff_in_means_fit_effect_conversion(binary_outcome_data):
    model = DiffInMeans()
    model.fit(binary_outcome_data)
    result = model.estimate(method="conversion_ztest")
    assert hasattr(result, "p_value")
    assert hasattr(result, "value")


def test_diff_in_means_not_fitted():
    model = DiffInMeans()
    with pytest.raises(RuntimeError):
        model.estimate()


def test_diff_in_means_invalid_method(sample_data):
    model = DiffInMeans()
    model.fit(sample_data)
    with pytest.raises(ValueError):
        model.estimate(method="invalid_method")


def test_diff_in_means_aliases(sample_data, binary_outcome_data):
    model = DiffInMeans()
    model.fit(sample_data)
    result_permutation = model.estimate(method="welch_permutation_t_test", B=10, seed=1)
    assert hasattr(result_permutation, "p_value")

    with pytest.raises(ValueError):
        model.estimate(method="bootstrap")

    model.fit(binary_outcome_data)
    result_conversion = model.estimate(method="conversion_ztest")
    assert hasattr(result_conversion, "p_value")


def test_diff_in_means_repr(sample_data):
    model = DiffInMeans()
    assert "status='unfitted'" in repr(model)
    model.fit(sample_data)
    assert "status='fitted'" in repr(model)


def test_diff_in_means_alpha(sample_data):
    model = DiffInMeans()
    model.fit(sample_data)
    result = model.estimate(method="ttest", alpha=0.1)
    # We could check if CI matches 0.1, but here we just check if it works
    assert hasattr(result, "ci_lower_absolute")
    assert hasattr(result, "ci_upper_absolute")
    assert isinstance(result.is_significant, bool)
    assert result.alpha == 0.1
