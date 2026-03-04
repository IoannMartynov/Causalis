from __future__ import annotations

from datetime import date, datetime
from typing import Hashable, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


TimeLike = Union[str, date, datetime, pd.Timestamp, pd.Period]


class PanelDataSCM(BaseModel):
    """Validated long-format panel contract for Synthetic Control estimators.

    Required fields
    ---------------
    df : pandas.DataFrame
        Long-format panel data.
    y : str
        Outcome column name in ``df``.
    unit_col : str
        Unit identifier column name in ``df``.
    time_col : str
        Calendar time column name in ``df``.
        Preferred input format is ``pandas.Period`` values with a regular
        frequency (for example monthly ``Period['M']``). Datetime/string values
        are accepted only when a regular frequency can be inferred.
    treated_time : str
        Binary treatment-assignment column in ``df`` (0/1 or False/True).

    Notes
    -----
    There are no optional contract fields. Extra keyword arguments are rejected.
    The contract derives ``treated_unit``, ``treatment_start``, and ``time_freq``
    from the input data.
    The model stores a validated internal dataframe snapshot used by all contract
    methods; mutating the public ``df`` attribute after construction does not
    affect validated contract behavior.
    For fiscal quarter/year semantics, pass ``time_col`` explicitly as
    ``pandas.Period`` with the desired fiscal frequency.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    df: pd.DataFrame = Field(..., description="Long-format panel data.")
    y: str = Field(..., description="Outcome column in df.")
    unit_col: str = Field(..., description="Unit identifier column in df.")
    time_col: str = Field(..., description="Calendar time column in df.")
    treated_time: str = Field(..., description="Binary treatment-assignment column in df.")

    _treated_unit: Optional[Hashable] = PrivateAttr(default=None)
    _treatment_start: Optional[pd.Period] = PrivateAttr(default=None)
    _time_freq: Optional[str] = PrivateAttr(default=None)
    _n_pre_periods: Optional[int] = PrivateAttr(default=None)
    _n_post_periods: Optional[int] = PrivateAttr(default=None)
    _df_validated: Optional[pd.DataFrame] = PrivateAttr(default=None)

    @staticmethod
    def _canonical_period_freq(freq: str) -> Optional[str]:
        try:
            return pd.period_range(start="2000-01-01", periods=1, freq=freq).freqstr
        except Exception:
            return None

    def _normalize_inferred_freq(self, freq: Optional[str]) -> Optional[str]:
        if freq is None:
            return None

        raw = str(freq).upper()
        candidates = [raw]
        if raw.startswith("MS") or raw.startswith("ME"):
            candidates.append("M")
        elif raw.startswith("QS") or raw.startswith("Q"):
            candidates.append("Q")
        elif raw.startswith("AS") or raw.startswith("YS") or raw.startswith("A") or raw.startswith("Y"):
            candidates.append("Y")

        for candidate in candidates:
            canonical = self._canonical_period_freq(candidate)
            if canonical is not None:
                return canonical
        return None

    def _infer_period_freq_from_datetimes(self, dt: pd.Series) -> str:
        unique_times = pd.DatetimeIndex(sorted(pd.Index(dt.unique()).tolist()))
        if unique_times.empty:
            raise ValueError(f"{self.time_col!r} must contain at least one time value.")
        if len(unique_times) < 3:
            raise ValueError(
                f"Could not infer a regular period frequency from {self.time_col!r} with fewer than 3 unique "
                "datetime values. Provide pandas.Period values in the time column."
            )

        inferred = pd.infer_freq(unique_times)
        normalized = self._normalize_inferred_freq(inferred)
        if normalized is not None:
            return normalized

        raise ValueError(
            f"Could not infer a regular period frequency from {self.time_col!r}. "
            "Use pandas.Period values with an explicit frequency."
        )

    def _reject_numeric_time_series(self, s: pd.Series) -> None:
        if pd.api.types.is_numeric_dtype(s):
            raise ValueError(
                f"{self.time_col!r} must represent explicit calendar time, not numeric values. "
                "Use pandas.Period or datetime-like values."
            )

        numeric_cast = pd.to_numeric(s, errors="coerce")
        if bool(numeric_cast.notna().all()):
            raise ValueError(
                f"{self.time_col!r} must represent explicit calendar time, not numeric-like values. "
                "Use pandas.Period or datetime-like values."
            )

    def _coerce_period_series(self, s: pd.Series) -> tuple[pd.Series, str]:
        if isinstance(s.dtype, pd.PeriodDtype):
            freq = pd.PeriodIndex(s).freqstr
            if freq is None:
                raise ValueError(f"{self.time_col!r} period frequency is missing.")
            return s, freq

        if isinstance(s.dtype, pd.DatetimeTZDtype):
            raise ValueError(
                f"{self.time_col!r} contains timezone-aware datetimes; normalize to naive/UTC first."
            )

        if bool(s.map(lambda v: isinstance(v, pd.Period)).all()):
            period_index = pd.PeriodIndex(s)
            freq = period_index.freqstr
            if freq is None:
                raise ValueError(f"{self.time_col!r} period frequency is missing.")
            return pd.Series(period_index, index=s.index, name=s.name), freq

        self._reject_numeric_time_series(s)
        dt = pd.to_datetime(s, errors="raise")
        if getattr(dt.dt, "tz", None) is not None:
            raise ValueError(
                f"{self.time_col!r} contains timezone-aware datetimes; normalize to naive/UTC first."
            )

        freq = self._infer_period_freq_from_datetimes(dt)
        try:
            return dt.dt.to_period(freq), freq
        except Exception as exc:
            raise ValueError(
                f"Failed to coerce {self.time_col!r} to Period[{freq}]. "
                "Use pandas.Period values with explicit frequency."
            ) from exc

    @model_validator(mode="after")
    def _validate_schema(self) -> "PanelDataSCM":
        df = self.df
        if df.empty:
            raise ValueError("df must contain at least one row.")

        required = {self.unit_col, self.time_col, self.y, self.treated_time}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        role_columns = [self.unit_col, self.time_col, self.y, self.treated_time]
        if len(set(role_columns)) != len(role_columns):
            raise ValueError("Column role names must be distinct across unit_col, time_col, y, treated_time.")

        if df[self.unit_col].isna().any():
            raise ValueError(f"{self.unit_col!r} contains nulls.")
        if df[self.time_col].isna().any():
            raise ValueError(f"{self.time_col!r} contains nulls.")
        if df[self.treated_time].isna().any():
            raise ValueError(f"{self.treated_time!r} contains nulls.")

        df = df.copy()

        try:
            coerced_time, time_freq = self._coerce_period_series(df[self.time_col])
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to normalize {self.time_col!r}.") from exc
        df[self.time_col] = coerced_time

        y_num = pd.to_numeric(df[self.y], errors="coerce")
        created_nan = y_num.isna() & ~df[self.y].isna()
        if bool(created_nan.any()):
            raise ValueError(f"{self.y!r} contains non-numeric values.")
        df[self.y] = y_num

        treatment_raw = df[self.treated_time]
        treatment_allowed = {0, 1, True, False}
        if not set(treatment_raw.dropna().unique()).issubset(treatment_allowed):
            raise ValueError(f"{self.treated_time!r} must be boolean or 0/1.")
        treatment = treatment_raw.astype(int)
        df[self.treated_time] = treatment

        treatment_conflicts = (
            df.groupby([self.unit_col, self.time_col], sort=False)[self.treated_time]
            .nunique(dropna=False)
            .gt(1)
        )
        if bool(treatment_conflicts.any()):
            raise ValueError(
                f"{self.treated_time!r} must be consistent within each ({self.unit_col}, {self.time_col}) cell."
            )

        if bool(df.duplicated([self.unit_col, self.time_col]).any()):
            raise ValueError(
                f"Duplicate (unit,time) rows found in [{self.unit_col}, {self.time_col}]. "
                "Aggregate duplicated rows before contract construction."
            )

        treated_rows = df[df[self.treated_time] == 1]
        if treated_rows.empty:
            raise ValueError(f"{self.treated_time!r} must have at least one treated row (value 1).")

        treated_units = pd.Index(treated_rows[self.unit_col].unique()).tolist()
        if len(treated_units) != 1:
            raise ValueError(
                f"{self.treated_time!r} must identify exactly one treated unit; got {len(treated_units)}."
            )
        treated_unit = treated_units[0]
        treatment_start = min(pd.Index(treated_rows[self.time_col].unique()).tolist())

        treated_unit_rows = df[df[self.unit_col] == treated_unit]
        if bool(
            (
                treated_unit_rows.loc[treated_unit_rows[self.time_col] < treatment_start, self.treated_time]
                != 0
            ).any()
        ):
            raise ValueError(f"{self.treated_time!r} for treated_unit must be 0 before treatment_start.")
        if bool(
            (
                treated_unit_rows.loc[treated_unit_rows[self.time_col] >= treatment_start, self.treated_time]
                != 1
            ).any()
        ):
            raise ValueError(f"{self.treated_time!r} for treated_unit must be 1 at/after treatment_start.")

        non_treated_rows = df[df[self.unit_col] != treated_unit]
        if bool((non_treated_rows[self.treated_time] != 0).any()):
            raise ValueError(f"{self.treated_time!r} must be 0 for all donor/control units.")

        units = pd.Index(df[self.unit_col].unique()).tolist()
        donors = [u for u in units if u != treated_unit]
        if len(donors) < 2:
            raise ValueError("Need at least 2 donor units.")

        projected = df[[self.unit_col, self.time_col, self.treated_time, self.y]].copy()
        projected["observed"] = projected[self.y].notna().astype(int)
        projected = projected[[self.unit_col, self.time_col, self.treated_time, "observed", self.y]]

        object.__setattr__(self, "_df_validated", projected.copy(deep=True))
        object.__setattr__(self, "df", projected)
        object.__setattr__(self, "_treated_unit", treated_unit)
        object.__setattr__(self, "_treatment_start", treatment_start)
        object.__setattr__(self, "_time_freq", time_freq)

        analysis_df = self.df_analysis()
        analysis_times = list(self.analysis_times())
        expected_times = list(
            pd.period_range(start=min(analysis_times), end=max(analysis_times), freq=self.time_freq)
        )
        if analysis_times != expected_times:
            missing_times = sorted(set(expected_times) - set(analysis_times))
            raise ValueError(
                "Analysis time axis has gaps relative to inferred time_freq. "
                f"Missing periods: {missing_times}"
            )

        pre_times = list(self.pre_times())
        post_times = list(self.post_times())
        if not pre_times:
            raise ValueError("No pre-treatment periods available.")
        if not post_times:
            raise ValueError("No post-treatment periods available.")

        donor_pre_rows = analysis_df[
            (analysis_df[self.unit_col].isin(donors))
            & (analysis_df[self.time_col].isin(pre_times))
        ]
        donor_pre_units = set(donor_pre_rows[self.unit_col].unique().tolist())
        donors_without_pre = sorted(set(donors) - donor_pre_units)
        if donors_without_pre:
            raise ValueError(
                "Each donor must have at least one pre-treatment row in analysis data. "
                f"Donors without pre rows: {donors_without_pre}"
            )

        treated_post = analysis_df[
            (analysis_df[self.unit_col] == treated_unit)
            & (analysis_df[self.time_col].isin(post_times))
        ]

        observed_post_times = set(pd.Index(treated_post[self.time_col].unique()).tolist())
        missing_post_times = sorted(set(post_times) - observed_post_times)
        missing_y_times = sorted(
            pd.Index(treated_post.loc[treated_post[self.y].isna(), self.time_col].unique()).tolist()
        )

        if missing_post_times or missing_y_times:
            bad_times = sorted(set(missing_post_times) | set(missing_y_times))
            raise ValueError(
                "treated_unit must have observed y in all post-treatment periods. "
                f"Missing/unobserved treated post periods: {bad_times}"
            )
        object.__setattr__(self, "_n_pre_periods", len(pre_times))
        object.__setattr__(self, "_n_post_periods", len(post_times))

        return self

    @property
    def treated_unit(self) -> Hashable:
        if self._treated_unit is None:
            raise RuntimeError("treated_unit metadata is not initialized.")
        return self._treated_unit

    @property
    def treatment_start(self) -> pd.Period:
        if self._treatment_start is None:
            raise RuntimeError("treatment_start metadata is not initialized.")
        return self._treatment_start

    @property
    def time_freq(self) -> str:
        if self._time_freq is None:
            raise RuntimeError("time_freq metadata is not initialized.")
        return self._time_freq

    @property
    def n_pre_periods(self) -> int:
        if self._n_pre_periods is None:
            raise RuntimeError("n_pre_periods metadata is not initialized.")
        return self._n_pre_periods

    @property
    def n_post_periods(self) -> int:
        if self._n_post_periods is None:
            raise RuntimeError("n_post_periods metadata is not initialized.")
        return self._n_post_periods

    def _validated_df(self) -> pd.DataFrame:
        if self._df_validated is None:
            raise RuntimeError("validated dataframe snapshot is not initialized.")
        return self._df_validated

    def donor_pool(self) -> Sequence[Hashable]:
        units = pd.Index(self._validated_df()[self.unit_col].unique()).tolist()
        return [u for u in units if u != self.treated_unit]

    def df_analysis(self) -> pd.DataFrame:
        keep_units = set(self.donor_pool()) | {self.treated_unit}
        df = self._validated_df()
        return df[df[self.unit_col].isin(keep_units)].copy()

    def pre_times(self) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted([t for t in times if t < self.treatment_start])

    def post_times(self) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted([t for t in times if t >= self.treatment_start])

    def analysis_times(self) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted(times)

    def time_to_index(self) -> dict[pd.Period, int]:
        times = self.analysis_times()
        return {t: i for i, t in enumerate(times)}

    def treatment_start_idx(self) -> int:
        mapping = self.time_to_index()
        if self.treatment_start not in mapping:
            raise ValueError("treatment_start is not present in analysis time axis.")
        return mapping[self.treatment_start]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"y={self.y!r}, "
            f"unit_col={self.unit_col!r}, "
            f"time_col={self.time_col!r}, "
            f"treated_time={self.treated_time!r}, "
            f"time_freq={self.time_freq!r}, "
            f"treated_unit={self.treated_unit!r}, "
            f"treatment_start={self.treatment_start!r}, "
            f"n_pre_periods={self.n_pre_periods!r}, "
            f"n_post_periods={self.n_post_periods!r}, "
            f"donor_units={list(self.donor_pool())!r})"
        )
