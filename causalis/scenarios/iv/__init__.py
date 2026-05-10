"""
Instrumental-variable (IV) scenario and estimators.

This module provides tools for causal inference when there is unobserved
confounding between treatment and outcome, but a valid instrument is available.

The primary estimator is the :class:`IIVM` (Iterated Instrumental Variable
Model), which uses double machine learning to estimate the Local Average
Treatment Effect (LATE).

Examples
--------
>>> from causalis.scenarios.iv import generate_offer_iv_26, IIVM
>>> # data is returned as IVCausalData by default
>>> data = generate_offer_iv_26(n=1000)
>>> model = IIVM()
>>> model.fit(data)
>>> result = model.estimate()
"""

from .dgp import generate_offer_iv_26
from .model import IIVM, IVCausalEstimate

__all__ = ["IIVM", "IVCausalEstimate", "generate_offer_iv_26"]
