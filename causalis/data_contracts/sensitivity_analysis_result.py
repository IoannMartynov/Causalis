from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd


class SensitivityAnalysisResult(dict):
    """Dict-compatible sensitivity analysis result with rich summaries."""

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        *,
        summary_builder: Callable[..., pd.DataFrame] | None = None,
        text_summary_builder: Callable[..., str] | None = None,
        summary_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(data or {})
        self._summary_builder = summary_builder
        self._text_summary_builder = text_summary_builder
        self._summary_kwargs = dict(summary_kwargs or {})

    def summary(self) -> pd.DataFrame:
        """Return a tabular summary of the sensitivity result."""
        if self._summary_builder is None:
            return pd.DataFrame([dict(self)])
        return self._summary_builder(self, **self._summary_kwargs)

    def text_summary(self) -> str:
        """Return the human-readable sensitivity report."""
        if self._text_summary_builder is None:
            return self.summary().to_string()
        return self._text_summary_builder(self, **self._summary_kwargs)

    def copy(self) -> "SensitivityAnalysisResult":
        """Copy the result while preserving custom summary builders."""
        return SensitivityAnalysisResult(
            dict(self),
            summary_builder=self._summary_builder,
            text_summary_builder=self._text_summary_builder,
            summary_kwargs=self._summary_kwargs,
        )

    def __str__(self) -> str:
        return self.text_summary()

    def __repr__(self) -> str:
        return self.text_summary()
