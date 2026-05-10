import importlib


def test_scenarios_rct_exports():
    # These are referenced in docs autosummary under causalis.scenarios.classic_rct
    mod = importlib.import_module("causalis.scenarios.classic_rct")
    for name in ["ttest", "conversion_ztest", "welch_permutation_t_test"]:
        assert hasattr(
            mod, name
        ), f"causalis.scenarios.classic_rct missing expected export: {name}"


def test_scenarios_unconfoundedness_exports():
    mod = importlib.import_module("causalis.scenarios.unconfoundedness")
    assert hasattr(mod, "IRM")


def test_inference_subpackages_functions():
    # ATT/ATE/GATE functions referenced in docs by fully qualified names
    unconf = importlib.import_module("causalis.scenarios.unconfoundedness")
    assert hasattr(unconf, "IRM")
