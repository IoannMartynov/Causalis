"""Unconfoundedness estimators, data generators, and refutation utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .model import IRM

__all__ = ["gate", "refutation", "dgp", "IRM"]


def __getattr__(name: str) -> Any:
    """Lazily expose heavyweight scenario subpackages."""
    if name == "model":
        module = import_module(f"{__name__}.model")
    elif name == "dgp":
        module = import_module(f"{__name__}.dgp")
    elif name == "refutation":
        module = import_module(f"{__name__}.refutation")
    elif name == "gate":
        module = import_module("causalis.scenarios.gate")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Keep lazy exports discoverable in interactive sessions."""
    return sorted(set(globals()) | set(__all__))
