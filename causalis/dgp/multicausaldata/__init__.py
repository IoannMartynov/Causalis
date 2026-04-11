from causalis.data_contracts.multicausaldata import MultiCausalData
from .base import MultiCausalDatasetGenerator
from .functional import generate_multitreatment


def generate_multitreatment_gamma_26(*args, **kwargs):
    # Lazy import to avoid circular dependency at module import time.
    from causalis.scenarios.multi_unconfoundedness.dgp import (
        generate_multitreatment_gamma_26 as _generate_multitreatment_gamma_26,
    )

    return _generate_multitreatment_gamma_26(*args, **kwargs)


def generate_multitreatment_binary_26(*args, **kwargs):
    # Lazy import to avoid circular dependency at module import time.
    from causalis.scenarios.multi_unconfoundedness.dgp import (
        generate_multitreatment_binary_26 as _generate_multitreatment_binary_26,
    )

    return _generate_multitreatment_binary_26(*args, **kwargs)


def generate_multitreatment_irm_26(*args, **kwargs):
    # Lazy import to avoid circular dependency at module import time.
    from causalis.scenarios.multi_unconfoundedness.dgp import (
        generate_multitreatment_irm_26 as _generate_multitreatment_irm_26,
    )

    return _generate_multitreatment_irm_26(*args, **kwargs)


def generate_multi_dml_cx_26(*args, **kwargs):
    # Lazy import to avoid circular dependency at module import time.
    from causalis.scenarios.multi_unconfoundedness.dgp import (
        generate_multi_dml_cx_26 as _generate_multi_dml_cx_26,
    )

    return _generate_multi_dml_cx_26(*args, **kwargs)


__all__ = [
    "MultiCausalData",
    "MultiCausalDatasetGenerator",
    "generate_multitreatment",
    "generate_multitreatment_gamma_26",
    "generate_multitreatment_binary_26",
    "generate_multitreatment_irm_26",
    "generate_multi_dml_cx_26",
]
