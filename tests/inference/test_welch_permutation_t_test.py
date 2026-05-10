"""
Tests for the welch_permutation_t_test function in the classic RCT inference module.
"""

import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import CausalData
from causalis.scenarios.classic_rct import welch_permutation_t_test


@pytest.fixture
def random_seed():
    return 123


@pytest.fixture
def cont_test_data(random_seed):
    rng = np.random.default_rng(random_seed)
    n = 1200
    control_mean = 5.0
    treatment_effect = 1.5
    sd = 2.0

    treatment = rng.choice([0, 1], size=n)
    target = np.where(
        treatment == 1,
        rng.normal(control_mean + treatment_effect, sd, size=n),
        rng.normal(control_mean, sd, size=n),
    )

    df = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": target,
            "age": rng.integers(18, 70, size=n),
        }
    )

    return {
        "df": df,
        "n": n,
        "control_mean": control_mean,
        "treatment_effect": treatment_effect,
    }


@pytest.fixture
def causal_data(cont_test_data):
    return CausalData(
        df=cont_test_data["df"],
        outcome="outcome",
        treatment="treatment",
        confounders=["age"],
    )


def test_basic_keys_and_types(causal_data):
    res = welch_permutation_t_test(causal_data, B=300, seed=7)
    expected = [
        "p_value",
        "t_obs",
        "absolute_difference",
        "absolute_ci",
        "relative_difference",
        "relative_ci",
        "B",
        "alternative",
    ]

    assert all(k in res for k in expected)
    assert isinstance(res["p_value"], float)
    assert 0 <= res["p_value"] <= 1
    assert isinstance(res["t_obs"], float)
    assert isinstance(res["absolute_ci"], tuple) and len(res["absolute_ci"]) == 2
    assert isinstance(res["relative_ci"], tuple) and len(res["relative_ci"]) == 2
    assert res["B"] == 300
    assert res["alternative"] == "two-sided"


def test_reproducible_with_fixed_seed(causal_data):
    res1 = welch_permutation_t_test(causal_data, B=400, seed=99)
    res2 = welch_permutation_t_test(causal_data, B=400, seed=99)

    assert res1["p_value"] == res2["p_value"]
    assert res1["t_obs"] == res2["t_obs"]
    assert res1["absolute_ci"] == res2["absolute_ci"]


def test_alternatives(causal_data):
    two_sided = welch_permutation_t_test(
        causal_data, B=600, alternative="two-sided", seed=42
    )
    greater = welch_permutation_t_test(
        causal_data, B=600, alternative="greater", seed=42
    )
    less = welch_permutation_t_test(causal_data, B=600, alternative="less", seed=42)

    assert two_sided["alternative"] == "two-sided"
    assert greater["alternative"] == "greater"
    assert less["alternative"] == "less"
    assert greater["p_value"] < less["p_value"]


def test_p_value_uses_plus_one_correction(causal_data):
    B = 25
    res = welch_permutation_t_test(causal_data, B=B, alternative="greater", seed=101)

    assert res["p_value"] >= 1 / (B + 1)


def test_effect_size_and_ci(causal_data, cont_test_data):
    res = welch_permutation_t_test(causal_data, B=300, seed=11)
    expected = cont_test_data["treatment_effect"]
    actual = res["absolute_difference"]

    assert abs(actual - expected) < 0.3
    assert actual > 0
    assert res["t_obs"] > 0

    lo, hi = res["absolute_ci"]
    assert lo <= expected <= hi


def test_errors_non_binary_treatment(cont_test_data):
    df = cont_test_data["df"].copy()
    rng = np.random.default_rng(7)
    df["treatment"] = rng.choice([0, 1, 2], size=cont_test_data["n"])

    with pytest.raises(ValueError, match="binary encoded"):
        CausalData(df=df, outcome="outcome", treatment="treatment", confounders=["age"])


def test_errors_too_few_observations():
    df = pd.DataFrame(
        {
            "treatment": [0, 1, 1],
            "outcome": [1.0, 2.0, 3.0],
        }
    )
    ck = CausalData(df=df, outcome="outcome", treatment="treatment")

    with pytest.raises(ValueError, match="at least 2"):
        welch_permutation_t_test(ck, B=100)


def test_invalid_params(causal_data):
    with pytest.raises(ValueError, match="alpha"):
        welch_permutation_t_test(causal_data, alpha=1.2)
    with pytest.raises(ValueError, match="alpha"):
        welch_permutation_t_test(causal_data, alpha=-0.1)
    with pytest.raises(ValueError, match="B"):
        welch_permutation_t_test(causal_data, B=0)
    with pytest.raises(ValueError, match="B"):
        welch_permutation_t_test(causal_data, B=-10)
    with pytest.raises(ValueError, match="alternative"):
        welch_permutation_t_test(causal_data, alternative="unknown")
