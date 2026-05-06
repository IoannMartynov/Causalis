from __future__ import annotations

from datetime import date, datetime
from typing import Any, Hashable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)


TimeLike = Union[str, date, datetime, pd.Timestamp, pd.Period]


ComparisonGroup = Literal["never_treated", "not_yet_treated", "not_yet_or_never"]


class PanelDataDID(BaseModel):
    """Validated long-format panel contract for staggered-adoption DID.

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
        Binary period-level treatment column in ``df`` (0/1 or False/True).
        Treatment is required to be absorbing: once a unit is treated, every
        later observed row for that unit must remain treated.
    covariates : sequence of str, optional
        Optional numeric covariate column names in ``df``. The input alias
        ``covariants`` is accepted for convenience.
    cluster_col : str, optional
        Optional cluster identifier column name in ``df``. The input alias
        ``cluster`` is accepted for convenience. ``unit_col`` is valid for
        unit-level clustering. Estimators that form unit-level influence
        functions require non-unit cluster columns to be stable within unit;
        ``time_col`` is therefore valid for the contract but not for those
        estimators.

    Notes
    -----
    Extra keyword arguments are rejected.
    The contract derives ``treated_units``, ``control_units`` (never-treated
    units), ``cohorts``, per-unit first treatment dates, and ``time_freq`` from
    the input data. ``treatment_start`` is kept as a compatibility alias for the
    earliest treatment cohort; staggered estimators should use ``cohorts`` and
    ``first_treatment_by_unit`` instead.
    This contract is intended for Callaway & Sant'Anna-style staggered
    difference-in-differences designs. It validates cohort support for at least
    one estimable ``ATT(g,t)`` cell using not-yet-treated or never-treated
    comparison units and exposes a full support table via ``att_gt_cells``.
    The model stores a validated internal dataframe snapshot used by all contract
    methods; mutating the public ``df`` attribute after construction does not
    affect validated contract behavior.
    Outcome ``y`` must not contain null/NaN values. Represent missing panel
    periods by omitting unit-time rows, not by keeping rows with ``NaN`` outcome.
    For fiscal quarter/year semantics, pass ``time_col`` explicitly as
    ``pandas.Period`` with the desired fiscal frequency.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    df: pd.DataFrame = Field(..., description="Long-format panel data.")
    y: str = Field(..., description="Outcome column in df.")
    unit_col: str = Field(..., description="Unit identifier column in df.")
    time_col: str = Field(..., description="Calendar time column in df.")
    treated_time: str = Field(
        ..., description="Binary treatment-assignment column in df."
    )
    covariates: tuple[str, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("covariates", "covariants"),
        serialization_alias="covariates",
        description="Optional numeric covariate columns in df.",
    )
    cluster_col: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("cluster_col", "cluster"),
        serialization_alias="cluster_col",
        description="Optional cluster identifier column in df.",
    )

    _treated_units: Optional[tuple[Hashable, ...]] = PrivateAttr(default=None)
    _control_units: Optional[tuple[Hashable, ...]] = PrivateAttr(default=None)
    _first_treatment_by_unit: Optional[dict[Hashable, Optional[pd.Period]]] = (
        PrivateAttr(default=None)
    )
    _cohort_by_unit: Optional[dict[Hashable, pd.Period]] = PrivateAttr(default=None)
    _cohort_units: Optional[dict[pd.Period, tuple[Hashable, ...]]] = PrivateAttr(
        default=None
    )
    _cohorts: Optional[tuple[pd.Period, ...]] = PrivateAttr(default=None)
    _treatment_start: Optional[pd.Period] = PrivateAttr(default=None)
    _latest_treatment_start: Optional[pd.Period] = PrivateAttr(default=None)
    _time_freq: Optional[str] = PrivateAttr(default=None)
    _n_pre_periods: Optional[int] = PrivateAttr(default=None)
    _n_post_periods: Optional[int] = PrivateAttr(default=None)
    _df_validated: Optional[pd.DataFrame] = PrivateAttr(default=None)

    @staticmethod
    def _normalize_column_name(value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} entries must be strings.")
        out = value.strip()
        if not out:
            raise ValueError(f"{field_name} entries must be non-empty strings.")
        return out

    @field_validator("covariates", mode="before")
    @classmethod
    def _normalize_covariates(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw = [value]
        elif isinstance(value, (list, tuple)):
            raw = list(value)
        else:
            raise TypeError(
                "covariates must be None, a string, or a list/tuple of strings."
            )

        seen = set()
        out = []
        for item in raw:
            covariate = cls._normalize_column_name(item, "covariates")
            if covariate not in seen:
                out.append(covariate)
                seen.add(covariate)
        return tuple(out)

    @field_validator("cluster_col", mode="before")
    @classmethod
    def _normalize_cluster_col(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return cls._normalize_column_name(value, "cluster_col")

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
        elif (
            raw.startswith("AS")
            or raw.startswith("YS")
            or raw.startswith("A")
            or raw.startswith("Y")
        ):
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
    def _validate_schema(self) -> "PanelDataDID":
        df = self.df
        if df.empty:
            raise ValueError("df must contain at least one row.")

        covariates = list(self.covariates)
        cluster_col = self.cluster_col
        primary_roles = {self.unit_col, self.time_col, self.y, self.treated_time}

        required = primary_roles | set(covariates)
        if cluster_col is not None:
            required.add(cluster_col)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        role_columns = [self.unit_col, self.time_col, self.y, self.treated_time]
        if len(set(role_columns)) != len(role_columns):
            raise ValueError(
                "Column role names must be distinct across unit_col, time_col, y, treated_time."
            )

        covariate_role_overlap = sorted(set(covariates).intersection(primary_roles))
        if covariate_role_overlap:
            raise ValueError(
                "covariates must be distinct from unit_col, time_col, y, and treated_time. "
                f"Overlapping columns: {covariate_role_overlap}"
            )
        if cluster_col is not None:
            if cluster_col in {self.y, self.treated_time}:
                raise ValueError(
                    "cluster_col must be distinct from y and treated_time."
                )
            if cluster_col in set(covariates):
                raise ValueError("cluster_col must be distinct from covariates.")

        if df[self.unit_col].isna().any():
            raise ValueError(f"{self.unit_col!r} contains nulls.")
        if df[self.time_col].isna().any():
            raise ValueError(f"{self.time_col!r} contains nulls.")
        if df[self.y].isna().any():
            raise ValueError(
                f"{self.y!r} contains nulls. Represent missing periods by omitting rows, not NaN outcomes."
            )
        if df[self.treated_time].isna().any():
            raise ValueError(f"{self.treated_time!r} contains nulls.")
        for covariate in covariates:
            if df[covariate].isna().any():
                raise ValueError(f"Covariate {covariate!r} contains nulls.")
        if cluster_col is not None and df[cluster_col].isna().any():
            raise ValueError(f"{cluster_col!r} contains nulls.")

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
        if not np.isfinite(y_num.to_numpy(dtype=float)).all():
            raise ValueError(f"{self.y!r} must contain only finite numeric values.")
        df[self.y] = y_num

        for covariate in covariates:
            covariate_num = pd.to_numeric(df[covariate], errors="coerce")
            created_covariate_nan = covariate_num.isna() & ~df[covariate].isna()
            if bool(created_covariate_nan.any()):
                raise ValueError(
                    f"Covariate {covariate!r} contains non-numeric values."
                )
            if not np.isfinite(covariate_num.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Covariate {covariate!r} must contain only finite numeric values."
                )
            if covariate_num.nunique(dropna=False) <= 1:
                raise ValueError(f"Covariate {covariate!r} is constant.")
            df[covariate] = covariate_num

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
            raise ValueError(
                f"{self.treated_time!r} must have at least one treated row (value 1)."
            )

        units = tuple(pd.Index(df[self.unit_col].unique()).tolist())
        starts_by_unit = (
            treated_rows.groupby(self.unit_col, sort=False)[self.time_col]
            .min()
            .to_dict()
        )
        treated_units = tuple(u for u in units if u in starts_by_unit)
        control_units = tuple(u for u in units if u not in starts_by_unit)
        first_treatment_by_unit = {u: starts_by_unit.get(u) for u in units}
        cohort_by_unit = {u: starts_by_unit[u] for u in treated_units}
        cohorts = tuple(sorted(set(starts_by_unit.values())))
        cohort_units = {
            cohort: tuple(u for u in treated_units if starts_by_unit[u] == cohort)
            for cohort in cohorts
        }
        treatment_start = cohorts[0]
        latest_treatment_start = cohorts[-1]

        for unit, start in starts_by_unit.items():
            unit_rows = df[df[self.unit_col] == unit]
            if bool(
                (
                    unit_rows.loc[unit_rows[self.time_col] < start, self.treated_time]
                    != 0
                ).any()
            ):
                raise ValueError(
                    f"{self.treated_time!r} for ever-treated units must be 0 before their first treatment period."
                )
            if bool(
                (
                    unit_rows.loc[unit_rows[self.time_col] >= start, self.treated_time]
                    != 1
                ).any()
            ):
                raise ValueError(
                    f"{self.treated_time!r} for ever-treated units must be 1 at/after their first treatment period."
                )

        control_rows = df[df[self.unit_col].isin(control_units)]
        if bool((control_rows[self.treated_time] != 0).any()):
            raise ValueError(
                f"{self.treated_time!r} must be 0 for all never-treated control units."
            )

        projected_cols = [self.unit_col, self.time_col, self.treated_time, self.y]
        projected_cols.extend(covariates)
        if cluster_col is not None and cluster_col not in projected_cols:
            projected_cols.append(cluster_col)
        projected = df[projected_cols].copy()

        object.__setattr__(self, "_df_validated", projected.copy(deep=True))
        object.__setattr__(self, "df", projected)
        object.__setattr__(self, "_treated_units", treated_units)
        object.__setattr__(self, "_control_units", control_units)
        object.__setattr__(self, "_first_treatment_by_unit", first_treatment_by_unit)
        object.__setattr__(self, "_cohort_by_unit", cohort_by_unit)
        object.__setattr__(self, "_cohort_units", cohort_units)
        object.__setattr__(self, "_cohorts", cohorts)
        object.__setattr__(self, "_treatment_start", treatment_start)
        object.__setattr__(self, "_latest_treatment_start", latest_treatment_start)
        object.__setattr__(self, "_time_freq", time_freq)

        analysis_times = list(self.analysis_times())
        expected_times = list(
            pd.period_range(
                start=min(analysis_times), end=max(analysis_times), freq=self.time_freq
            )
        )
        if analysis_times != expected_times:
            missing_times = sorted(set(expected_times) - set(analysis_times))
            raise ValueError(
                "Analysis time axis has gaps relative to inferred time_freq. "
                f"Missing periods: {missing_times}"
            )

        time_index = self.time_to_index()
        cohorts_without_pre = [cohort for cohort in cohorts if time_index[cohort] == 0]
        if cohorts_without_pre:
            raise ValueError(
                "Each treatment cohort must have at least one pre-treatment analysis period. "
                f"Cohorts at the first analysis period: {cohorts_without_pre}"
            )

        pre_times = list(self.pre_times())
        post_times = list(self.post_times())
        if not pre_times:
            raise ValueError("No pre-treatment periods available.")
        if not post_times:
            raise ValueError("No post-treatment periods available.")

        support = self.att_gt_cells(
            control_group="not_yet_or_never", include_unsupported=True
        )
        if support.empty or not bool(support["is_supported"].any()):
            raise ValueError(
                "No supported Callaway-Sant'Anna ATT(g,t) cells found. "
                "Need at least one cohort-time cell with observed cohort units and not-yet-treated or "
                "never-treated comparison units in both the base and target periods."
            )

        object.__setattr__(self, "_n_pre_periods", len(pre_times))
        object.__setattr__(self, "_n_post_periods", len(post_times))

        return self

    @property
    def treated_units(self) -> Sequence[Hashable]:
        if self._treated_units is None:
            raise RuntimeError("treated_units metadata is not initialized.")
        return self._treated_units

    @property
    def control_units(self) -> Sequence[Hashable]:
        if self._control_units is None:
            raise RuntimeError("control_units metadata is not initialized.")
        return self._control_units

    @property
    def never_treated_units(self) -> Sequence[Hashable]:
        """Units that are never treated in the observed panel."""

        return self.control_units

    @property
    def first_treatment_by_unit(self) -> dict[Hashable, Optional[pd.Period]]:
        """Map every unit to its first treatment period, or ``None`` if never treated."""

        if self._first_treatment_by_unit is None:
            raise RuntimeError("first_treatment_by_unit metadata is not initialized.")
        return dict(self._first_treatment_by_unit)

    @property
    def cohort_by_unit(self) -> dict[Hashable, pd.Period]:
        """Map ever-treated units to their first treatment cohort."""

        if self._cohort_by_unit is None:
            raise RuntimeError("cohort_by_unit metadata is not initialized.")
        return dict(self._cohort_by_unit)

    @property
    def cohorts(self) -> Sequence[pd.Period]:
        """Sorted first-treatment periods among ever-treated units."""

        if self._cohorts is None:
            raise RuntimeError("cohorts metadata is not initialized.")
        return self._cohorts

    @property
    def treatment_start(self) -> pd.Period:
        if self._treatment_start is None:
            raise RuntimeError("treatment_start metadata is not initialized.")
        return self._treatment_start

    @property
    def latest_treatment_start(self) -> pd.Period:
        if self._latest_treatment_start is None:
            raise RuntimeError("latest_treatment_start metadata is not initialized.")
        return self._latest_treatment_start

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

    @property
    def last_post_period(self) -> pd.Period:
        post_times = self.post_times()
        if not post_times:
            raise RuntimeError(
                "last_post_period is not available because there are no post-treatment periods."
            )
        return post_times[-1]

    @property
    def design_type(
        self,
    ) -> Literal["canonical_2x2", "simultaneous_adoption", "staggered_adoption"]:
        if len(self.cohorts) > 1:
            return "staggered_adoption"
        if self.n_pre_periods == 1 and self.n_post_periods == 1:
            return "canonical_2x2"
        return "simultaneous_adoption"

    @property
    def has_covariates(self) -> bool:
        return bool(self.covariates)

    @property
    def has_cluster(self) -> bool:
        return self.cluster_col is not None

    def _validated_df(self) -> pd.DataFrame:
        if self._df_validated is None:
            raise RuntimeError("validated dataframe snapshot is not initialized.")
        return self._df_validated

    def df_analysis(self) -> pd.DataFrame:
        return self._validated_df().copy()

    def covariate_frame(self) -> pd.DataFrame:
        """Return a copy of the validated covariate design columns."""

        return self.df_analysis()[list(self.covariates)].copy()

    def cluster_series(self) -> pd.Series:
        """Return a copy of the validated cluster identifier column."""

        if self.cluster_col is None:
            raise RuntimeError("cluster_col is not set.")
        return self.df_analysis()[self.cluster_col].copy()

    def _coerce_period_value(self, value: TimeLike, field_name: str) -> pd.Period:
        if isinstance(value, pd.Period):
            if value.freqstr != self.time_freq:
                raise ValueError(
                    f"{field_name} must have frequency {self.time_freq!r}; got {value.freqstr!r}."
                )
            return value
        try:
            return pd.Period(value, freq=self.time_freq)
        except Exception as exc:
            raise ValueError(
                f"{field_name} must be coercible to Period[{self.time_freq}]."
            ) from exc

    def cohort_units(self, cohort: TimeLike) -> Sequence[Hashable]:
        """Return units first treated in the requested cohort."""

        cohort_period = self._coerce_period_value(cohort, "cohort")
        if self._cohort_units is None:
            raise RuntimeError("cohort_units metadata is not initialized.")
        if cohort_period not in self._cohort_units:
            raise ValueError(f"Unknown treatment cohort: {cohort_period!r}.")
        return self._cohort_units[cohort_period]

    def not_yet_treated_units(
        self, time: TimeLike, *, include_never: bool = True
    ) -> Sequence[Hashable]:
        """Return units untreated at ``time`` because they adopt later or never adopt."""

        time_period = self._coerce_period_value(time, "time")
        first_by_unit = self.first_treatment_by_unit
        out = []
        for unit in self.df_analysis()[self.unit_col].drop_duplicates().tolist():
            first_treatment = first_by_unit[unit]
            if first_treatment is None:
                if include_never:
                    out.append(unit)
            elif first_treatment > time_period:
                out.append(unit)
        return tuple(out)

    def comparison_units(
        self,
        cohort: TimeLike,
        time: TimeLike,
        *,
        control_group: ComparisonGroup = "not_yet_or_never",
    ) -> Sequence[Hashable]:
        """Return valid comparison units for a Callaway-Sant'Anna ``ATT(g,t)`` cell."""

        cohort_period = self._coerce_period_value(cohort, "cohort")
        time_period = self._coerce_period_value(time, "time")
        if cohort_period not in set(self.cohorts):
            raise ValueError(f"Unknown treatment cohort: {cohort_period!r}.")
        if time_period < cohort_period:
            raise ValueError(
                "ATT(g,t) comparison time must be at or after the cohort treatment period."
            )
        if control_group not in {
            "never_treated",
            "not_yet_treated",
            "not_yet_or_never",
        }:
            raise ValueError(
                "control_group must be one of 'never_treated', 'not_yet_treated', or 'not_yet_or_never'."
            )

        first_by_unit = self.first_treatment_by_unit
        out = []
        for unit in self.df_analysis()[self.unit_col].drop_duplicates().tolist():
            first_treatment = first_by_unit[unit]
            if control_group == "never_treated":
                include = first_treatment is None
            elif control_group == "not_yet_treated":
                include = first_treatment is not None and first_treatment > time_period
            else:
                include = first_treatment is None or first_treatment > time_period
            if include:
                out.append(unit)
        return tuple(out)

    def df_for_did(
        self,
        *,
        treated_group_col: str = "treated_group",
        post_col: str = "post",
        cohort_col: str = "cohort",
        event_time_col: str = "event_time",
    ) -> pd.DataFrame:
        """Return analysis data with derived staggered-DID cohort and event-time columns."""

        reserved = set(self.df_analysis().columns)
        extra_cols = {treated_group_col, post_col, cohort_col, event_time_col}
        if len(extra_cols) != 4:
            raise ValueError("Derived DID column names must be distinct.")
        overlap = reserved.intersection(extra_cols)
        if overlap:
            raise ValueError(
                f"Derived DID column names conflict with contract columns: {sorted(overlap)}"
            )

        df = self.df_analysis()
        first_by_unit = self.first_treatment_by_unit
        time_index = self.time_to_index()
        cohorts = df[self.unit_col].map(first_by_unit)
        df[cohort_col] = cohorts
        df[treated_group_col] = (
            df[self.unit_col].isin(set(self.treated_units)).astype(int)
        )
        df[post_col] = [
            int(not pd.isna(cohort) and time >= cohort)
            for time, cohort in zip(df[self.time_col], cohorts)
        ]
        df[event_time_col] = pd.Series(
            [
                time_index[time] - time_index[cohort] if not pd.isna(cohort) else pd.NA
                for time, cohort in zip(df[self.time_col], cohorts)
            ],
            index=df.index,
            dtype="Int64",
        )
        return df

    def pre_times(self, cohort: Optional[TimeLike] = None) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        cutoff = (
            self.treatment_start
            if cohort is None
            else self._coerce_period_value(cohort, "cohort")
        )
        if cohort is not None and cutoff not in set(self.cohorts):
            raise ValueError(f"Unknown treatment cohort: {cutoff!r}.")
        return sorted([t for t in times if t < cutoff])

    def post_times(self, cohort: Optional[TimeLike] = None) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        cutoff = (
            self.treatment_start
            if cohort is None
            else self._coerce_period_value(cohort, "cohort")
        )
        if cohort is not None and cutoff not in set(self.cohorts):
            raise ValueError(f"Unknown treatment cohort: {cutoff!r}.")
        return sorted([t for t in times if t >= cutoff])

    def analysis_times(self) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted(times)

    def time_to_index(self) -> dict[pd.Period, int]:
        times = self.analysis_times()
        return {t: i for i, t in enumerate(times)}

    def treatment_start_idx(self, cohort: Optional[TimeLike] = None) -> int:
        mapping = self.time_to_index()
        target = (
            self.treatment_start
            if cohort is None
            else self._coerce_period_value(cohort, "cohort")
        )
        if target not in mapping:
            raise ValueError(
                "treatment_start/cohort is not present in analysis time axis."
            )
        return mapping[target]

    def att_gt_cells(
        self,
        *,
        control_group: ComparisonGroup = "not_yet_or_never",
        anticipation: int = 0,
        base_period: Literal["universal", "varying"] = "universal",
        include_pre_periods: bool = False,
        include_unsupported: bool = False,
    ) -> pd.DataFrame:
        """Return Callaway-Sant'Anna ``ATT(g,t)`` support under an explicit policy."""

        if control_group not in {
            "never_treated",
            "not_yet_treated",
            "not_yet_or_never",
        }:
            raise ValueError(
                "control_group must be one of 'never_treated', 'not_yet_treated', or 'not_yet_or_never'."
            )
        anticipation = int(anticipation)
        if anticipation < 0:
            raise ValueError("anticipation must be a non-negative integer.")
        if base_period not in {"universal", "varying"}:
            raise ValueError("base_period must be either 'universal' or 'varying'.")

        df = self.df_analysis()
        times = list(self.analysis_times())
        time_index = self.time_to_index()
        first_by_unit = self.first_treatment_by_unit
        rows = []

        def comparison_at(time: pd.Period) -> set[Hashable]:
            target_idx = time_index[time] + anticipation
            out = set()
            for unit in df[self.unit_col].drop_duplicates().tolist():
                first_treatment = first_by_unit[unit]
                if control_group == "never_treated":
                    include = first_treatment is None
                elif control_group == "not_yet_treated":
                    include = first_treatment is not None and time_index[first_treatment] > target_idx
                else:
                    include = first_treatment is None or time_index[first_treatment] > target_idx
                if include:
                    out.add(unit)
            return out

        for cohort in self.cohorts:
            cohort_idx = time_index[cohort]
            universal_base_idx = cohort_idx - anticipation - 1
            if universal_base_idx < 0:
                if include_unsupported:
                    rows.append(
                        {
                            "cohort": cohort,
                            "time": pd.NaT,
                            "base_time": pd.NaT,
                            "event_time": pd.NA,
                            "control_group": control_group,
                            "anticipation": anticipation,
                            "base_period": base_period,
                            "is_post_treatment": False,
                            "n_treated_available": 0,
                            "n_treated_complete": 0,
                            "n_control_available": 0,
                            "n_control_complete": 0,
                            "n_treated": 0,
                            "n_control": 0,
                            "is_supported": False,
                            "unsupported_reason": "no_valid_base_period",
                        }
                    )
                continue

            cohort_units = set(self.cohort_units(cohort))
            first_target_idx = 1 if include_pre_periods else cohort_idx
            for target_idx in range(first_target_idx, len(times)):
                is_post = target_idx >= cohort_idx
                pre_cutoff = cohort_idx - anticipation
                is_valid_pre = (
                    target_idx < universal_base_idx
                    if base_period == "universal"
                    else target_idx < pre_cutoff
                )
                if not is_post and not is_valid_pre:
                    continue
                if is_post or base_period == "universal":
                    base_idx = universal_base_idx
                else:
                    base_idx = target_idx - 1
                if base_idx < 0 or base_idx == target_idx:
                    continue

                time = times[target_idx]
                base_time = times[base_idx]
                comparison_units = comparison_at(time)
                base_rows = df[df[self.time_col] == base_time]
                target_rows = df[df[self.time_col] == time]

                n_treated_available = int(len(cohort_units))
                n_control_available = int(len(comparison_units))
                cohort_base_units = set(
                    base_rows.loc[
                        base_rows[self.unit_col].isin(cohort_units), self.unit_col
                    ]
                )
                cohort_target_units = set(
                    target_rows.loc[
                        target_rows[self.unit_col].isin(cohort_units), self.unit_col
                    ]
                )
                comparison_base_units = set(
                    base_rows.loc[
                        base_rows[self.unit_col].isin(comparison_units), self.unit_col
                    ]
                )
                comparison_target_units = set(
                    target_rows.loc[
                        target_rows[self.unit_col].isin(comparison_units), self.unit_col
                    ]
                )

                n_treated_complete = len(cohort_base_units.intersection(cohort_target_units))
                n_control_complete = len(
                    comparison_base_units.intersection(comparison_target_units)
                )
                if n_treated_complete <= 0:
                    reason = "no_complete_treated_units"
                elif n_control_complete <= 0:
                    reason = "no_complete_control_units"
                else:
                    reason = ""
                is_supported = reason == ""
                if include_unsupported or is_supported:
                    rows.append(
                        {
                            "cohort": cohort,
                            "time": time,
                            "base_time": base_time,
                            "event_time": target_idx - cohort_idx,
                            "control_group": control_group,
                            "anticipation": anticipation,
                            "base_period": base_period,
                            "is_post_treatment": is_post,
                            "n_treated_available": n_treated_available,
                            "n_treated_complete": n_treated_complete,
                            "n_control_available": n_control_available,
                            "n_control_complete": n_control_complete,
                            "n_treated": n_treated_complete,
                            "n_control": n_control_complete,
                            "is_supported": is_supported,
                            "unsupported_reason": reason,
                        }
                    )

        columns = [
            "cohort",
            "time",
            "base_time",
            "event_time",
            "control_group",
            "anticipation",
            "base_period",
            "is_post_treatment",
            "n_treated_available",
            "n_treated_complete",
            "n_control_available",
            "n_control_complete",
            "n_treated",
            "n_control",
            "is_supported",
            "unsupported_reason",
        ]
        return pd.DataFrame(rows, columns=columns)

    def cell_counts(self) -> pd.DataFrame:
        """Return ever-treated/never-treated row counts by analysis period and own-treatment status."""

        df = self.df_analysis()
        first_by_unit = self.first_treatment_by_unit
        cohorts = df[self.unit_col].map(first_by_unit)
        counts_input = pd.DataFrame(
            {
                "_did_time": df[self.time_col].to_numpy(),
                "_did_post": [
                    int(not pd.isna(cohort) and time >= cohort)
                    for time, cohort in zip(df[self.time_col], cohorts)
                ],
                "_did_ever_treated": df[self.unit_col]
                .isin(set(self.treated_units))
                .astype(int)
                .to_numpy(),
            }
        )
        counts = (
            counts_input.groupby(
                ["_did_time", "_did_post", "_did_ever_treated"], observed=True
            )
            .size()
            .rename("n")
            .reset_index()
        )
        counts = counts.rename(
            columns={"_did_time": self.time_col, "_did_post": "post"}
        )
        counts["group"] = counts["_did_ever_treated"].map(
            {0: "never_treated", 1: "ever_treated"}
        )
        return counts[[self.time_col, "post", "group", "n"]]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"y={self.y!r}, "
            f"unit_col={self.unit_col!r}, "
            f"time_col={self.time_col!r}, "
            f"treated_time={self.treated_time!r}, "
            f"covariates={list(self.covariates)!r}, "
            f"cluster_col={self.cluster_col!r}, "
            f"time_freq={self.time_freq!r}, "
            f"design_type={self.design_type!r}, "
            f"cohorts={list(self.cohorts)!r}, "
            f"treatment_start={self.treatment_start!r}, "
            f"latest_treatment_start={self.latest_treatment_start!r}, "
            f"last_post_period={self.last_post_period!r}, "
            f"n_pre_periods={self.n_pre_periods!r}, "
            f"n_post_periods={self.n_post_periods!r}, "
            f"treated_units={list(self.treated_units)!r}, "
            f"never_treated_units={list(self.never_treated_units)!r})"
        )
