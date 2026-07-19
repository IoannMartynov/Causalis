"""
Refutation and robustness utilities for Causalis.

Importing this package exposes the public functions from all refutation
submodules (overlap, score, unconfoundedness) so you can access
commonly used helpers directly via `causalis.refutation`.
"""

from . import overlap, score, unconfoundedness

# Re-export public API from subpackages
# Overlap diagnostics (public API defined in overlap/__init__.py)
from .overlap import *  # noqa: F401,F403

# Score-based refutations and diagnostics
from .score import *  # noqa: F401,F403

# Unconfoundedness sensitivity and balance checks
from .unconfoundedness.love_plot import *  # noqa: F401,F403
from .unconfoundedness.unconfoundedness_validation import *  # noqa: F401,F403
from .unconfoundedness.sensitivity import *  # noqa: F401,F403

# Best-effort __all__: include common high-level entry points for discoverability.
# Note: wildcard imports above already populate the module namespace.
try:
    from .overlap import __all__ as __all_overlap  # type: ignore
except Exception:
    __all_overlap = []

# score_validation and unconfoundedness APIs exposed for discoverability.
__all_score = [
    "run_score_diagnostics",
    "plot_influence_instability",
    "plot_residual_diagnostics",
]

__all_unconf = [
    "love_plot",
    "sensitivity_analysis",
    "get_sensitivity_summary",
    "run_unconfoundedness_diagnostics",
    "sensitivity_benchmark",
    "sensitivity_benchmark_group",
    "run_sensitivity_protocol",
]

__all__ = ["overlap", "score", "unconfoundedness"] + list(dict.fromkeys([*__all_overlap, *__all_score, *__all_unconf]))
