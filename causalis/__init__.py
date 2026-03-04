"""
Causalis: A Python package for causal inference.
"""

import importlib
import warnings
from typing import TYPE_CHECKING

# Suppress noisy tqdm warning in environments without ipywidgets
try:
    from tqdm import TqdmWarning  # type: ignore
    # Apply more comprehensive filter
    warnings.filterwarnings(
        "ignore",
        message=".*IProgress not found.*",
        category=TqdmWarning,
    )
    # Also filter the exact message from the test
    warnings.filterwarnings(
        "ignore", 
        message="IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html",
        category=TqdmWarning,
    )
except Exception:
    # If tqdm is not installed or any issue arises, do not fail import
    pass

__version__ = "0.1.2"
__all__ = ["data_contracts", "dgp", "scenarios", "shared"]

_LAZY_SUBMODULES = {"data_contracts", "dgp", "scenarios", "shared"}


def __getattr__(name):  # pragma: no cover - behavior tested via subprocess
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module

    # 'design' is optional; keep import non-fatal if missing in editable installs
    if name == "design":
        try:
            module = importlib.import_module(".design", __name__)
        except Exception:
            module = None
        globals()[name] = module
        return module

    # Compatibility mapping
    if name == "data":
        warnings.warn(
            "causalis.data is deprecated, use causalis.data_contracts instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return __getattr__("data_contracts")

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if TYPE_CHECKING:  # Hint for static type checkers without importing at runtime
    from . import scenarios as scenarios  # noqa: F401
    from . import shared as shared  # noqa: F401
    from . import dgp as dgp  # noqa: F401
    from . import data_contracts as data_contracts  # noqa: F401
