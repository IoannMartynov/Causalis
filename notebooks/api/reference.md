## `causalis`

Causalis: A Python package for causal inference.

**Modules:**

- [**data_contracts**](#causalis.data_contracts) –
- [**dgp**](#causalis.dgp) –
- [**scenarios**](#causalis.scenarios) –
- [**shared**](#causalis.shared) –

### `causalis.data_contracts`

**Modules:**

- [**causal_diagnostic_data**](#causalis.data_contracts.causal_diagnostic_data) –
- [**causal_estimate**](#causalis.data_contracts.causal_estimate) –
- [**causaldata**](#causalis.data_contracts.causaldata) – Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.
- [**causaldata_instrumental**](#causalis.data_contracts.causaldata_instrumental) –
- [**gate_estimate**](#causalis.data_contracts.gate_estimate) –
- [**multicausal_estimate**](#causalis.data_contracts.multicausal_estimate) –
- [**multicausaldata**](#causalis.data_contracts.multicausaldata) – Causalis Dataclass for storing Cross-sectional DataFrame and column metadata
- [**panel_data_scm**](#causalis.data_contracts.panel_data_scm) –
- [**panel_estimate**](#causalis.data_contracts.panel_estimate) –

**Classes:**

- [**CausalData**](#causalis.data_contracts.CausalData) – Container for causal inference datasets.
- [**CausalDataInstrumental**](#causalis.data_contracts.CausalDataInstrumental) – Container for causal inference datasets with causaldata_instrumental variables.
- [**CausalEstimate**](#causalis.data_contracts.CausalEstimate) – Result container for causal effect estimates.
- [**DiagnosticData**](#causalis.data_contracts.DiagnosticData) – Base class for all diagnostic data_contracts.
- [**GateEstimate**](#causalis.data_contracts.GateEstimate) – Result contract for Group Average Treatment Effects (GATE).
- [**MultiCausalData**](#causalis.data_contracts.MultiCausalData) – Data contract for cross-sectional causal data with multi-class one-hot treatments.
- [**PanelDataSCM**](#causalis.data_contracts.PanelDataSCM) – Validated long-format panel contract for Synthetic Control estimators.
- [**PanelEstimate**](#causalis.data_contracts.PanelEstimate) – Result contract for dynamic synthetic-control effect-path estimates.
- [**RegressionChecks**](#causalis.data_contracts.RegressionChecks) – Lightweight OLS/regression health checks for CUPED diagnostics.
- [**UnconfoundednessDiagnosticData**](#causalis.data_contracts.UnconfoundednessDiagnosticData) – Fields common to all models assuming unconfoundedness.

#### `CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data_contracts contract.

**Attributes:**

- [**df**](#causalis.data_contracts.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.data_contracts.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.data_contracts.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.data_contracts.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.data_contracts.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.data_contracts.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.data_contracts.CausalData.get_df) – Get a DataFrame with specified columns.

##### `X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

##### `confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

##### `confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

##### `df`

```python
df: pd.DataFrame
```

##### `from_df`

```python
from_df(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    **kwargs: Any
) -> "CausalData"
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code> – A validated CausalData instance.

##### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_treatment: bool = True,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_user_id: bool = False,
) -> pd.DataFrame
```

Get a DataFrame with specified columns.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

##### `model_config`

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True, populate_by_name=True, extra="forbid"
)

```

##### `outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

##### `outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

##### `treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

##### `treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

##### `user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

##### `user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `CausalDataInstrumental`

Bases: <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code>

Container for causal inference datasets with causaldata_instrumental variables.

**Attributes:**

- [**instrument_name**](#causalis.data_contracts.CausalDataInstrumental.instrument_name) (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.

**Functions:**

- [**from_df**](#causalis.data_contracts.CausalDataInstrumental.from_df) – Friendly constructor for CausalDataInstrumental.
- [**get_df**](#causalis.data_contracts.CausalDataInstrumental.get_df) – Get a DataFrame with specified columns including instrument.

##### `from_df`

```python
from_df(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    instrument: str = None,
    **kwargs: Any
) -> "CausalDataInstrumental"
```

Friendly constructor for CausalDataInstrumental.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- **instrument** (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalDataInstrumental](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental)</code> – A validated CausalDataInstrumental instance.

##### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_treatment: bool = True,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_user_id: bool = False,
    include_instrument: bool = False,
) -> pd.DataFrame
```

Get a DataFrame with specified columns including instrument.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.
- **include_instrument** (<code>[bool](#bool)</code>) – Whether to include the instrument column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

##### `instrument`

```python
instrument: pd.Series
```

instrument column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The instrument column.

##### `instrument_name`

```python
instrument_name: str = Field(alias='instrument')
```

#### `CausalEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result container for causal effect estimates.

**Parameters:**

- **estimand** (<code>[str](#str)</code>) – The estimand being estimated (e.g., 'ATE', 'ATTE', 'CATE').
- **model** (<code>[str](#str)</code>) – The name of the model used for estimation.
- **model_options** (<code>[dict](#dict)</code>) – Options passed to the model.
- **value** (<code>[float](#float)</code>) – The estimated absolute effect.
- **ci_upper_absolute** (<code>[float](#float)</code>) – Upper bound of the absolute confidence interval.
- **ci_lower_absolute** (<code>[float](#float)</code>) – Lower bound of the absolute confidence interval.
- **value_relative** (<code>[float](#float)</code>) – The estimated relative effect.
- **ci_upper_relative** (<code>[float](#float)</code>) – Upper bound of the relative confidence interval.
- **ci_lower_relative** (<code>[float](#float)</code>) – Lower bound of the relative confidence interval.
- **alpha** (<code>[float](#float)</code>) – The significance level (e.g., 0.05).
- **p_value** (<code>[float](#float)</code>) – The p-value from the test.
- **is_significant** (<code>[bool](#bool)</code>) – Whether the result is statistically significant at alpha.
- **n_treated** (<code>[int](#int)</code>) – Number of units in the treatment group.
- **n_control** (<code>[int](#int)</code>) – Number of units in the control group.
- **treatment_mean** (<code>[float](#float)</code>) – Mean outcome in the treatment group.
- **control_mean** (<code>[float](#float)</code>) – Mean outcome in the control group.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome variable.
- **treatment** (<code>[str](#str)</code>) – The name of the treatment variable.
- **confounders** (<code>list of str</code>) – The names of the confounders used in the model.
- **time** (<code>[str](#str)</code>) – The date when the estimate was created (YYYY-MM-DD).
- **diagnostic_data** (<code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>) – Additional diagnostic data_contracts.

**Functions:**

- [**summary**](#causalis.data_contracts.CausalEstimate.summary) – Return a summary DataFrame of the results.

##### `alpha`

```python
alpha: float
```

##### `ci_lower_absolute`

```python
ci_lower_absolute: float
```

##### `ci_lower_relative`

```python
ci_lower_relative: Optional[float] = None
```

##### `ci_upper_absolute`

```python
ci_upper_absolute: float
```

##### `ci_upper_relative`

```python
ci_upper_relative: Optional[float] = None
```

##### `confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

##### `control_mean`

```python
control_mean: float
```

##### `diagnostic_data`

```python
diagnostic_data: Optional[DiagnosticData] = None
```

##### `estimand`

```python
estimand: str
```

##### `is_significant`

```python
is_significant: bool
```

##### `model`

```python
model: str
```

##### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

##### `n_control`

```python
n_control: int
```

##### `n_treated`

```python
n_treated: int
```

##### `outcome`

```python
outcome: str
```

##### `p_value`

```python
p_value: Optional[float] = None
```

##### `summary`

```python
summary() -> pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Summary DataFrame.

##### `time`

```python
time: str = Field(default_factory=(lambda: datetime.now().strftime("%Y-%m-%d")))

```

##### `treatment`

```python
treatment: str
```

##### `treatment_mean`

```python
treatment_mean: float
```

##### `value`

```python
value: float
```

##### `value_relative`

```python
value_relative: Optional[float] = None
```

#### `DiagnosticData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Base class for all diagnostic data_contracts.

##### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

#### `GateEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result contract for Group Average Treatment Effects (GATE).

**Functions:**

- [**summary**](#causalis.data_contracts.GateEstimate.summary) – Return per-group GATE summary table.

##### `alpha`

```python
alpha: float
```

##### `ci_lower`

```python
ci_lower: np.ndarray
```

##### `ci_upper`

```python
ci_upper: np.ndarray
```

##### `covariance`

```python
covariance: pd.DataFrame
```

##### `diagnostic_data`

```python
diagnostic_data: Optional[Dict[str, Any]] = None
```

##### `estimand`

```python
estimand: str = 'GATE'
```

##### `group_names`

```python
group_names: List[str]
```

##### `max_propensity`

```python
max_propensity: np.ndarray
```

##### `mean_phi`

```python
mean_phi: np.ndarray
```

##### `mean_propensity`

```python
mean_propensity: np.ndarray
```

##### `min_propensity`

```python
min_propensity: np.ndarray
```

##### `model`

```python
model: str = 'IRM'
```

##### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

##### `n_control`

```python
n_control: np.ndarray
```

##### `n_group`

```python
n_group: np.ndarray
```

##### `n_treated`

```python
n_treated: np.ndarray
```

##### `p_values`

```python
p_values: np.ndarray
```

##### `share_treated`

```python
share_treated: np.ndarray
```

##### `std_errors`

```python
std_errors: np.ndarray
```

##### `std_phi`

```python
std_phi: np.ndarray
```

##### `summary`

```python
summary() -> pd.DataFrame
```

Return per-group GATE summary table.

##### `summary_table`

```python
summary_table: pd.DataFrame
```

##### `test_stats`

```python
test_stats: np.ndarray
```

##### `time`

```python
time: str = Field(default_factory=(lambda: datetime.now().strftime("%Y-%m-%d")))

```

##### `values`

```python
values: np.ndarray
```

#### `MultiCausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Data contract for cross-sectional causal data with multi-class one-hot treatments.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the causal data.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome column.
- **treatment_names** (<code>[List](#typing.List)\[[str](#str)\]</code>) – The names of the treatment columns.
- **confounders** (<code>[List](#typing.List)\[[str](#str)\]</code>) – The names of the confounder columns, by default [].
- **user_id** (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) – The name of the user ID column, by default None.
- **control_treatment** (<code>[str](#str)</code>) – Name of the control/baseline treatment column.

<details class="note" open markdown="1">
<summary>Notes</summary>

This class enforces several constraints on the data, including:

- Maximum number of treatment_names (default 15).
- No duplicate column names in the input DataFrame.
- Disjoint roles for columns (outcome, treatment_names, confounders, user_id).
- Non-empty normalized names for outcome and user_id (if provided).
- Existence of all specified columns in the DataFrame.
- Numeric or boolean types for outcome and confounders.
- Finite values for outcome, confounders, and treatment_names.
- Non-constant values for outcome, treatment_names, and confounders.
- No NaN values in used columns.
- Binary (0/1) encoding for treatment columns.
- One-hot treatment assignment (exactly one active treatment per row).
- A stable control treatment in position 0.
- No identical values between different columns.
- Unique values for user_id (if specified).

</details>

**Functions:**

- [**from_df**](#causalis.data_contracts.MultiCausalData.from_df) – Create a MultiCausalData instance from a pandas DataFrame.
- [**get_df**](#causalis.data_contracts.MultiCausalData.get_df) – Get a subset of the underlying DataFrame.

##### `FLOAT_TOL`

```python
FLOAT_TOL: float = 1e-12
```

##### `MAX_TREATMENTS`

```python
MAX_TREATMENTS: int = 15
```

##### `X`

```python
X: pd.DataFrame
```

Return the confounder columns as a pandas DataFrame.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The confounder columns.

##### `confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

##### `control_treatment`

```python
control_treatment: str
```

##### `df`

```python
df: pd.DataFrame
```

##### `from_df`

```python
from_df(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment_names: Union[str, List[str]],
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    control_treatment: str,
    **kwargs: Any
) -> "MultiCausalData"
```

Create a MultiCausalData instance from a pandas DataFrame.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The input DataFrame.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome column.
- **treatment_names** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – The name(s) of the treatment column(s).
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – The name(s) of the confounder column(s), by default None.
- **user_id** (<code>[str](#str)</code>) – The name of the user ID column, by default None.
- **control_treatment** (<code>[str](#str)</code>) – Name of the control treatment column.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional keyword arguments passed to the constructor.

**Returns:**

- <code>[MultiCausalData](#causalis.data_contracts.multicausaldata.MultiCausalData)</code> – An instance of MultiCausalData.

##### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_treatments: bool = True,
    include_user_id: bool = False,
) -> pd.DataFrame
```

Get a subset of the underlying DataFrame.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific columns to include, by default None.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column, by default True.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns, by default True.
- **include_treatments** (<code>[bool](#bool)</code>) – Whether to include treatment columns, by default True.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user ID column, by default False.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the requested DataFrame subset.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any of the requested columns do not exist.

##### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
```

##### `outcome`

```python
outcome: str
```

##### `treatment`

```python
treatment: pd.Series
```

Return the single treatment column as a pandas Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

**Raises:**

- <code>[AttributeError](#AttributeError)</code> – If there is more than one treatment column.

##### `treatment_names`

```python
treatment_names: List[str]
```

##### `treatments`

```python
treatments: pd.DataFrame
```

Return the treatment columns as a pandas DataFrame.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The treatment columns.

##### `user_id`

```python
user_id: Optional[str] = None
```

#### `PanelDataSCM`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Validated long-format panel contract for Synthetic Control estimators.

<details class="required-fields" open markdown="1">
<summary>Required fields</summary>

df : pandas.DataFrame
Long-format panel data.
y : str
Outcome column name in `df`.
unit_col : str
Unit identifier column name in `df`.
time_col : str
Calendar time column name in `df`.
Preferred input format is `pandas.Period` values with a regular
frequency (for example monthly `Period['M']`). Datetime/string values
are accepted only when a regular frequency can be inferred.
treated_time : str
Binary treatment-assignment column in `df` (0/1 or False/True).

</details>

<details class="note" open markdown="1">
<summary>Notes</summary>

There are no optional contract fields. Extra keyword arguments are rejected.
The contract derives `treated_unit`, `treatment_start`, and `time_freq`
from the input data.
The model stores a validated internal dataframe snapshot used by all contract
methods; mutating the public `df` attribute after construction does not
affect validated contract behavior.
Outcome `y` must not contain null/NaN values. Represent missing panel
periods by omitting unit-time rows, not by keeping rows with `NaN` outcome.
For fiscal quarter/year semantics, pass `time_col` explicitly as
`pandas.Period` with the desired fiscal frequency.

</details>

**Functions:**

- [**analysis_times**](#causalis.data_contracts.PanelDataSCM.analysis_times) –
- [**df_analysis**](#causalis.data_contracts.PanelDataSCM.df_analysis) –
- [**donor_pool**](#causalis.data_contracts.PanelDataSCM.donor_pool) –
- [**post_times**](#causalis.data_contracts.PanelDataSCM.post_times) –
- [**pre_times**](#causalis.data_contracts.PanelDataSCM.pre_times) –
- [**time_to_index**](#causalis.data_contracts.PanelDataSCM.time_to_index) –
- [**treatment_start_idx**](#causalis.data_contracts.PanelDataSCM.treatment_start_idx) –

##### `analysis_times`

```python
analysis_times() -> Sequence[pd.Period]
```

##### `df`

```python
df: pd.DataFrame = Field(..., description='Long-format panel data.')
```

##### `df_analysis`

```python
df_analysis() -> pd.DataFrame
```

##### `donor_pool`

```python
donor_pool() -> Sequence[Hashable]
```

##### `last_post_period`

```python
last_post_period: pd.Period
```

##### `model_config`

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True, extra="forbid", frozen=True
)

```

##### `n_post_periods`

```python
n_post_periods: int
```

##### `n_pre_periods`

```python
n_pre_periods: int
```

##### `post_times`

```python
post_times() -> Sequence[pd.Period]
```

##### `pre_times`

```python
pre_times() -> Sequence[pd.Period]
```

##### `time_col`

```python
time_col: str = Field(..., description='Calendar time column in df.')
```

##### `time_freq`

```python
time_freq: str
```

##### `time_to_index`

```python
time_to_index() -> dict[pd.Period, int]
```

##### `treated_time`

```python
treated_time: str = Field(
    ..., description="Binary treatment-assignment column in df."
)

```

##### `treated_unit`

```python
treated_unit: Hashable
```

##### `treatment_start`

```python
treatment_start: pd.Period
```

##### `treatment_start_idx`

```python
treatment_start_idx() -> int
```

##### `unit_col`

```python
unit_col: str = Field(..., description='Unit identifier column in df.')
```

##### `y`

```python
y: str = Field(..., description='Outcome column in df.')
```

#### `PanelEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result contract for dynamic synthetic-control effect-path estimates.

**Functions:**

- [**summary**](#causalis.data_contracts.PanelEstimate.summary) – Return a compact CausalEstimate-style summary table.
- [**summary_poinwise**](#causalis.data_contracts.PanelEstimate.summary_poinwise) – Return pointwise post-period estimates as a flat DataFrame.

##### `alpha`

```python
alpha: float
```

##### `ci_lower_by_time`

```python
ci_lower_by_time: pd.Series
```

##### `ci_upper_by_time`

```python
ci_upper_by_time: pd.Series
```

##### `confidence_set_by_time`

```python
confidence_set_by_time: Dict[TimeLike, list[tuple[float, float]]]
```

##### `created_at`

```python
created_at: datetime = Field(
    default_factory=(lambda: datetime.now(timezone.utc))
)

```

##### `diagnostics`

```python
diagnostics: Dict[str, Any] = Field(default_factory=dict)
```

##### `donor_weights_augmented`

```python
donor_weights_augmented: Dict[Hashable, float]
```

##### `effect_by_time`

```python
effect_by_time: pd.Series
```

##### `estimand`

```python
estimand: Literal['dynamic_effect_path'] = 'dynamic_effect_path'
```

##### `is_significant_by_time`

```python
is_significant_by_time: pd.Series
```

##### `model`

```python
model: str
```

##### `model_config`

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
)

```

##### `observed_outcome`

```python
observed_outcome: pd.Series
```

##### `p_value_by_time`

```python
p_value_by_time: pd.Series
```

##### `post_times`

```python
post_times: List[TimeLike]
```

##### `pre_times`

```python
pre_times: List[TimeLike]
```

##### `summary`

```python
summary() -> pd.DataFrame
```

Return a compact CausalEstimate-style summary table.

##### `summary_poinwise`

```python
summary_poinwise() -> pd.DataFrame
```

Return pointwise post-period estimates as a flat DataFrame.

##### `synthetic_outcome`

```python
synthetic_outcome: pd.Series
```

##### `treated_unit`

```python
treated_unit: Hashable
```

##### `treatment_start`

```python
treatment_start: TimeLike
```

#### `RegressionChecks`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Lightweight OLS/regression health checks for CUPED diagnostics.

##### `ate_adj`

```python
ate_adj: float
```

##### `ate_adj_winsor`

```python
ate_adj_winsor: Optional[float] = None
```

##### `ate_adj_winsor_gap`

```python
ate_adj_winsor_gap: Optional[float] = None
```

##### `ate_gap`

```python
ate_gap: float
```

##### `ate_gap_over_se_naive`

```python
ate_gap_over_se_naive: Optional[float] = None
```

##### `ate_naive`

```python
ate_naive: float
```

##### `condition_number`

```python
condition_number: float
```

##### `cooks_cutoff`

```python
cooks_cutoff: float
```

##### `full_rank`

```python
full_rank: bool
```

##### `k`

```python
k: int
```

##### `leverage_cutoff`

```python
leverage_cutoff: float
```

##### `max_abs_std_resid`

```python
max_abs_std_resid: float
```

##### `max_cooks`

```python
max_cooks: float
```

##### `max_leverage`

```python
max_leverage: float
```

##### `min_one_minus_h`

```python
min_one_minus_h: float
```

##### `n_high_cooks`

```python
n_high_cooks: int
```

##### `n_high_leverage`

```python
n_high_leverage: int
```

##### `n_std_resid_gt_3`

```python
n_std_resid_gt_3: int
```

##### `n_std_resid_gt_4`

```python
n_std_resid_gt_4: int
```

##### `n_tiny_one_minus_h`

```python
n_tiny_one_minus_h: int
```

##### `near_duplicate_pairs`

```python
near_duplicate_pairs: list[tuple[str, str, float]] = Field(default_factory=list)

```

##### `p_main_covariates`

```python
p_main_covariates: int
```

##### `rank`

```python
rank: int
```

##### `resid_scale_mad`

```python
resid_scale_mad: float
```

##### `vif`

```python
vif: Optional[Dict[str, float]] = None
```

##### `winsor_q`

```python
winsor_q: Optional[float] = None
```

#### `UnconfoundednessDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Fields common to all models assuming unconfoundedness.

##### `d`

```python
d: np.ndarray
```

##### `folds`

```python
folds: Optional[np.ndarray] = None
```

##### `g0_hat`

```python
g0_hat: Optional[np.ndarray] = None
```

##### `g1_hat`

```python
g1_hat: Optional[np.ndarray] = None
```

##### `m_alpha`

```python
m_alpha: Optional[np.ndarray] = None
```

##### `m_hat`

```python
m_hat: np.ndarray
```

##### `m_hat_raw`

```python
m_hat_raw: Optional[np.ndarray] = None
```

##### `normalize_ipw`

```python
normalize_ipw: Optional[bool] = None
```

##### `nu2`

```python
nu2: Optional[float] = None
```

##### `psi`

```python
psi: Optional[np.ndarray] = None
```

##### `psi_b`

```python
psi_b: Optional[np.ndarray] = None
```

##### `psi_nu2`

```python
psi_nu2: Optional[np.ndarray] = None
```

##### `psi_sigma2`

```python
psi_sigma2: Optional[np.ndarray] = None
```

##### `residual_plot_cache`

```python
residual_plot_cache: Optional[Dict[str, Any]] = None
```

##### `riesz_rep`

```python
riesz_rep: Optional[np.ndarray] = None
```

##### `score`

```python
score: Optional[str] = None
```

##### `score_plot_cache`

```python
score_plot_cache: Optional[Dict[str, Any]] = None
```

##### `sensitivity_analysis`

```python
sensitivity_analysis: Optional[Dict[str, Any]] = None
```

##### `sigma2`

```python
sigma2: Optional[float] = None
```

##### `trimming_threshold`

```python
trimming_threshold: float = 0.0
```

##### `w`

```python
w: Optional[np.ndarray] = None
```

##### `w_bar`

```python
w_bar: Optional[np.ndarray] = None
```

##### `x`

```python
x: Optional[np.ndarray] = None
```

##### `y`

```python
y: Optional[np.ndarray] = None
```

#### `causal_diagnostic_data`

**Classes:**

- [**CUPEDDiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData) – Diagnostic data_contracts for CUPED-style (Lin-interacted OLS) adjustment.
- [**DiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData) – Base class for all diagnostic data_contracts.
- [**DiffInMeansDiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.DiffInMeansDiagnosticData) – Diagnostic data_contracts for Difference-in-Means model.
- [**MultiUnconfoundednessDiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.MultiUnconfoundednessDiagnosticData) – Fields common to all models assuming unconfoundedness with multi_unconfoundedness.
- [**UnconfoundednessDiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData) – Fields common to all models assuming unconfoundedness.

##### `CUPEDDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Diagnostic data_contracts for CUPED-style (Lin-interacted OLS) adjustment.

###### `adj_type`

```python
adj_type: str
```

###### `ate_naive`

```python
ate_naive: float
```

###### `beta_covariates`

```python
beta_covariates: np.ndarray
```

###### `covariate_outcome_corr`

```python
covariate_outcome_corr: Optional[np.ndarray] = None
```

###### `covariates`

```python
covariates: List[str]
```

###### `gamma_interactions`

```python
gamma_interactions: np.ndarray
```

###### `r2_adj`

```python
r2_adj: float
```

###### `r2_naive`

```python
r2_naive: float
```

###### `regression_checks`

```python
regression_checks: Optional[RegressionChecks] = None
```

###### `se_naive`

```python
se_naive: float
```

###### `se_reduction_pct_same_cov`

```python
se_reduction_pct_same_cov: float
```

##### `DiagnosticData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Base class for all diagnostic data_contracts.

###### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `DiffInMeansDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Diagnostic data_contracts for Difference-in-Means model.

##### `MultiUnconfoundednessDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Fields common to all models assuming unconfoundedness with multi_unconfoundedness.

###### `d`

```python
d: np.ndarray
```

###### `folds`

```python
folds: Optional[np.ndarray] = None
```

###### `g_hat`

```python
g_hat: Optional[np.ndarray] = None
```

###### `m_alpha`

```python
m_alpha: Optional[np.ndarray] = None
```

###### `m_hat`

```python
m_hat: np.ndarray
```

###### `m_hat_raw`

```python
m_hat_raw: Optional[np.ndarray] = None
```

###### `normalize_ipw`

```python
normalize_ipw: Optional[bool] = None
```

###### `nu2`

```python
nu2: Optional[np.ndarray] = None
```

###### `psi`

```python
psi: Optional[np.ndarray] = None
```

###### `psi_b`

```python
psi_b: Optional[np.ndarray] = None
```

###### `psi_nu2`

```python
psi_nu2: Optional[np.ndarray] = None
```

###### `psi_sigma2`

```python
psi_sigma2: Optional[np.ndarray] = None
```

###### `residual_plot_cache`

```python
residual_plot_cache: Optional[Dict[str, Any]] = None
```

###### `riesz_rep`

```python
riesz_rep: Optional[np.ndarray] = None
```

###### `score`

```python
score: Optional[str] = None
```

###### `sensitivity_analysis`

```python
sensitivity_analysis: Optional[Dict[str, Any]] = None
```

###### `sigma2`

```python
sigma2: Union[float, np.ndarray] = None
```

###### `trimming_threshold`

```python
trimming_threshold: float = 0.0
```

###### `x`

```python
x: Optional[np.ndarray] = None
```

###### `y`

```python
y: Optional[np.ndarray] = None
```

##### `UnconfoundednessDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Fields common to all models assuming unconfoundedness.

###### `d`

```python
d: np.ndarray
```

###### `folds`

```python
folds: Optional[np.ndarray] = None
```

###### `g0_hat`

```python
g0_hat: Optional[np.ndarray] = None
```

###### `g1_hat`

```python
g1_hat: Optional[np.ndarray] = None
```

###### `m_alpha`

```python
m_alpha: Optional[np.ndarray] = None
```

###### `m_hat`

```python
m_hat: np.ndarray
```

###### `m_hat_raw`

```python
m_hat_raw: Optional[np.ndarray] = None
```

###### `normalize_ipw`

```python
normalize_ipw: Optional[bool] = None
```

###### `nu2`

```python
nu2: Optional[float] = None
```

###### `psi`

```python
psi: Optional[np.ndarray] = None
```

###### `psi_b`

```python
psi_b: Optional[np.ndarray] = None
```

###### `psi_nu2`

```python
psi_nu2: Optional[np.ndarray] = None
```

###### `psi_sigma2`

```python
psi_sigma2: Optional[np.ndarray] = None
```

###### `residual_plot_cache`

```python
residual_plot_cache: Optional[Dict[str, Any]] = None
```

###### `riesz_rep`

```python
riesz_rep: Optional[np.ndarray] = None
```

###### `score`

```python
score: Optional[str] = None
```

###### `score_plot_cache`

```python
score_plot_cache: Optional[Dict[str, Any]] = None
```

###### `sensitivity_analysis`

```python
sensitivity_analysis: Optional[Dict[str, Any]] = None
```

###### `sigma2`

```python
sigma2: Optional[float] = None
```

###### `trimming_threshold`

```python
trimming_threshold: float = 0.0
```

###### `w`

```python
w: Optional[np.ndarray] = None
```

###### `w_bar`

```python
w_bar: Optional[np.ndarray] = None
```

###### `x`

```python
x: Optional[np.ndarray] = None
```

###### `y`

```python
y: Optional[np.ndarray] = None
```

#### `causal_estimate`

**Classes:**

- [**CausalEstimate**](#causalis.data_contracts.causal_estimate.CausalEstimate) – Result container for causal effect estimates.

##### `CausalEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result container for causal effect estimates.

**Parameters:**

- **estimand** (<code>[str](#str)</code>) – The estimand being estimated (e.g., 'ATE', 'ATTE', 'CATE').
- **model** (<code>[str](#str)</code>) – The name of the model used for estimation.
- **model_options** (<code>[dict](#dict)</code>) – Options passed to the model.
- **value** (<code>[float](#float)</code>) – The estimated absolute effect.
- **ci_upper_absolute** (<code>[float](#float)</code>) – Upper bound of the absolute confidence interval.
- **ci_lower_absolute** (<code>[float](#float)</code>) – Lower bound of the absolute confidence interval.
- **value_relative** (<code>[float](#float)</code>) – The estimated relative effect.
- **ci_upper_relative** (<code>[float](#float)</code>) – Upper bound of the relative confidence interval.
- **ci_lower_relative** (<code>[float](#float)</code>) – Lower bound of the relative confidence interval.
- **alpha** (<code>[float](#float)</code>) – The significance level (e.g., 0.05).
- **p_value** (<code>[float](#float)</code>) – The p-value from the test.
- **is_significant** (<code>[bool](#bool)</code>) – Whether the result is statistically significant at alpha.
- **n_treated** (<code>[int](#int)</code>) – Number of units in the treatment group.
- **n_control** (<code>[int](#int)</code>) – Number of units in the control group.
- **treatment_mean** (<code>[float](#float)</code>) – Mean outcome in the treatment group.
- **control_mean** (<code>[float](#float)</code>) – Mean outcome in the control group.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome variable.
- **treatment** (<code>[str](#str)</code>) – The name of the treatment variable.
- **confounders** (<code>list of str</code>) – The names of the confounders used in the model.
- **time** (<code>[str](#str)</code>) – The date when the estimate was created (YYYY-MM-DD).
- **diagnostic_data** (<code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>) – Additional diagnostic data_contracts.

**Functions:**

- [**summary**](#causalis.data_contracts.causal_estimate.CausalEstimate.summary) – Return a summary DataFrame of the results.

###### `alpha`

```python
alpha: float
```

###### `ci_lower_absolute`

```python
ci_lower_absolute: float
```

###### `ci_lower_relative`

```python
ci_lower_relative: Optional[float] = None
```

###### `ci_upper_absolute`

```python
ci_upper_absolute: float
```

###### `ci_upper_relative`

```python
ci_upper_relative: Optional[float] = None
```

###### `confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

###### `control_mean`

```python
control_mean: float
```

###### `diagnostic_data`

```python
diagnostic_data: Optional[DiagnosticData] = None
```

###### `estimand`

```python
estimand: str
```

###### `is_significant`

```python
is_significant: bool
```

###### `model`

```python
model: str
```

###### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

###### `n_control`

```python
n_control: int
```

###### `n_treated`

```python
n_treated: int
```

###### `outcome`

```python
outcome: str
```

###### `p_value`

```python
p_value: Optional[float] = None
```

###### `summary`

```python
summary() -> pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Summary DataFrame.

###### `time`

```python
time: str = Field(default_factory=(lambda: datetime.now().strftime("%Y-%m-%d")))

```

###### `treatment`

```python
treatment: str
```

###### `treatment_mean`

```python
treatment_mean: float
```

###### `value`

```python
value: float
```

###### `value_relative`

```python
value_relative: Optional[float] = None
```

#### `causaldata`

Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.

**Classes:**

- [**CausalData**](#causalis.data_contracts.causaldata.CausalData) – Container for causal inference datasets.

##### `CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data_contracts contract.

**Attributes:**

- [**df**](#causalis.data_contracts.causaldata.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.data_contracts.causaldata.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.data_contracts.causaldata.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.data_contracts.causaldata.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.data_contracts.causaldata.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.data_contracts.causaldata.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.data_contracts.causaldata.CausalData.get_df) – Get a DataFrame with specified columns.

###### `X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

###### `confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

###### `confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

###### `df`

```python
df: pd.DataFrame
```

###### `from_df`

```python
from_df(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    **kwargs: Any
) -> "CausalData"
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code> – A validated CausalData instance.

###### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_treatment: bool = True,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_user_id: bool = False,
) -> pd.DataFrame
```

Get a DataFrame with specified columns.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

###### `model_config`

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True, populate_by_name=True, extra="forbid"
)

```

###### `outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

###### `outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

###### `treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

###### `treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

###### `user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

###### `user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causaldata_instrumental`

**Classes:**

- [**CausalDataInstrumental**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental) – Container for causal inference datasets with causaldata_instrumental variables.

##### `CausalDataInstrumental`

Bases: <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code>

Container for causal inference datasets with causaldata_instrumental variables.

**Attributes:**

- [**instrument_name**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.instrument_name) (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.

**Functions:**

- [**from_df**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.from_df) – Friendly constructor for CausalDataInstrumental.
- [**get_df**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.get_df) – Get a DataFrame with specified columns including instrument.

###### `from_df`

```python
from_df(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    instrument: str = None,
    **kwargs: Any
) -> "CausalDataInstrumental"
```

Friendly constructor for CausalDataInstrumental.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- **instrument** (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalDataInstrumental](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental)</code> – A validated CausalDataInstrumental instance.

###### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_treatment: bool = True,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_user_id: bool = False,
    include_instrument: bool = False,
) -> pd.DataFrame
```

Get a DataFrame with specified columns including instrument.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.
- **include_instrument** (<code>[bool](#bool)</code>) – Whether to include the instrument column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

###### `instrument`

```python
instrument: pd.Series
```

instrument column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The instrument column.

###### `instrument_name`

```python
instrument_name: str = Field(alias='instrument')
```

#### `gate_estimate`

**Classes:**

- [**GateEstimate**](#causalis.data_contracts.gate_estimate.GateEstimate) – Result contract for Group Average Treatment Effects (GATE).

##### `GateEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result contract for Group Average Treatment Effects (GATE).

**Functions:**

- [**summary**](#causalis.data_contracts.gate_estimate.GateEstimate.summary) – Return per-group GATE summary table.

###### `alpha`

```python
alpha: float
```

###### `ci_lower`

```python
ci_lower: np.ndarray
```

###### `ci_upper`

```python
ci_upper: np.ndarray
```

###### `covariance`

```python
covariance: pd.DataFrame
```

###### `diagnostic_data`

```python
diagnostic_data: Optional[Dict[str, Any]] = None
```

###### `estimand`

```python
estimand: str = 'GATE'
```

###### `group_names`

```python
group_names: List[str]
```

###### `max_propensity`

```python
max_propensity: np.ndarray
```

###### `mean_phi`

```python
mean_phi: np.ndarray
```

###### `mean_propensity`

```python
mean_propensity: np.ndarray
```

###### `min_propensity`

```python
min_propensity: np.ndarray
```

###### `model`

```python
model: str = 'IRM'
```

###### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

###### `n_control`

```python
n_control: np.ndarray
```

###### `n_group`

```python
n_group: np.ndarray
```

###### `n_treated`

```python
n_treated: np.ndarray
```

###### `p_values`

```python
p_values: np.ndarray
```

###### `share_treated`

```python
share_treated: np.ndarray
```

###### `std_errors`

```python
std_errors: np.ndarray
```

###### `std_phi`

```python
std_phi: np.ndarray
```

###### `summary`

```python
summary() -> pd.DataFrame
```

Return per-group GATE summary table.

###### `summary_table`

```python
summary_table: pd.DataFrame
```

###### `test_stats`

```python
test_stats: np.ndarray
```

###### `time`

```python
time: str = Field(default_factory=(lambda: datetime.now().strftime("%Y-%m-%d")))

```

###### `values`

```python
values: np.ndarray
```

#### `multicausal_estimate`

**Classes:**

- [**MultiCausalEstimate**](#causalis.data_contracts.multicausal_estimate.MultiCausalEstimate) – Result container for causal effect estimates.

##### `MultiCausalEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result container for causal effect estimates.

**Parameters:**

- **estimand** (<code>[str](#str)</code>) – The estimand being estimated (e.g., 'ATE', 'ATTE', 'CATE').
- **model** (<code>[str](#str)</code>) – The name of the model used for estimation.
- **model_options** (<code>[dict](#dict)</code>) – Options passed to the model.
- **value** (<code>[float](#float)</code>) – The estimated absolute effect.
- **ci_upper_absolute** (<code>[float](#float)</code>) – Upper bound of the absolute confidence interval.
- **ci_lower_absolute** (<code>[float](#float)</code>) – Lower bound of the absolute confidence interval.
- **value_relative** (<code>[float](#float)</code>) – The estimated relative effect.
- **ci_upper_relative** (<code>[float](#float)</code>) – Upper bound of the relative confidence interval.
- **ci_lower_relative** (<code>[float](#float)</code>) – Lower bound of the relative confidence interval.
- **alpha** (<code>[float](#float)</code>) – The significance level (e.g., 0.05).
- **p_value** (<code>[float](#float)</code>) – The p-value from the test.
- **is_significant** (<code>[bool](#bool)</code>) – Whether the result is statistically significant at alpha.
- **n_treated** (<code>[int](#int)</code>) – Number of units in the treatment group.
- **n_control** (<code>[int](#int)</code>) – Number of units in the control group.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome variable.
- **treatment** (<code>[str](#str)</code>) – The name of the treatment variable.
- **confounders** (<code>list of str</code>) – The names of the confounders used in the model.
- **time** (<code>[str](#str)</code>) – The date when the estimate was created (YYYY-MM-DD).
- **diagnostic_data** (<code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>) – Additional diagnostic data_contracts.

**Functions:**

- [**summary**](#causalis.data_contracts.multicausal_estimate.MultiCausalEstimate.summary) – Return a CausalEstimate-like summary for all baseline contrasts.

###### `alpha`

```python
alpha: float
```

###### `ci_lower_absolute`

```python
ci_lower_absolute: np.ndarray
```

###### `ci_lower_relative`

```python
ci_lower_relative: Optional[np.ndarray] = None
```

###### `ci_upper_absolute`

```python
ci_upper_absolute: np.ndarray
```

###### `ci_upper_relative`

```python
ci_upper_relative: Optional[np.ndarray] = None
```

###### `confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

###### `contrast_labels`

```python
contrast_labels: List[str] = Field(default_factory=list)
```

###### `control_mean`

```python
control_mean: Optional[float] = None
```

###### `diagnostic_data`

```python
diagnostic_data: Optional[DiagnosticData] = None
```

###### `estimand`

```python
estimand: str
```

###### `is_significant`

```python
is_significant: List[bool]
```

###### `model`

```python
model: str
```

###### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

###### `n_control`

```python
n_control: int
```

###### `n_treated`

```python
n_treated: int
```

###### `n_treated_by_arm`

```python
n_treated_by_arm: Optional[np.ndarray] = None
```

###### `outcome`

```python
outcome: str
```

###### `p_value`

```python
p_value: Optional[np.ndarray] = None
```

###### `summary`

```python
summary() -> pd.DataFrame
```

Return a CausalEstimate-like summary for all baseline contrasts.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Summary indexed by `field` and with one column per contrast (d_k vs d_0).

###### `time`

```python
time: str = Field(default_factory=(lambda: datetime.now().strftime("%Y-%m-%d")))

```

###### `treatment`

```python
treatment: List[str]
```

###### `treatment_mean`

```python
treatment_mean: Optional[np.ndarray] = None
```

###### `value`

```python
value: np.ndarray
```

###### `value_relative`

```python
value_relative: Optional[np.ndarray] = None
```

#### `multicausaldata`

Causalis Dataclass for storing Cross-sectional DataFrame and column metadata
for causal inference with multiple treatments.

**Classes:**

- [**MultiCausalData**](#causalis.data_contracts.multicausaldata.MultiCausalData) – Data contract for cross-sectional causal data with multi-class one-hot treatments.

##### `MultiCausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Data contract for cross-sectional causal data with multi-class one-hot treatments.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the causal data.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome column.
- **treatment_names** (<code>[List](#typing.List)\[[str](#str)\]</code>) – The names of the treatment columns.
- **confounders** (<code>[List](#typing.List)\[[str](#str)\]</code>) – The names of the confounder columns, by default [].
- **user_id** (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) – The name of the user ID column, by default None.
- **control_treatment** (<code>[str](#str)</code>) – Name of the control/baseline treatment column.

<details class="note" open markdown="1">
<summary>Notes</summary>

This class enforces several constraints on the data, including:

- Maximum number of treatment_names (default 15).
- No duplicate column names in the input DataFrame.
- Disjoint roles for columns (outcome, treatment_names, confounders, user_id).
- Non-empty normalized names for outcome and user_id (if provided).
- Existence of all specified columns in the DataFrame.
- Numeric or boolean types for outcome and confounders.
- Finite values for outcome, confounders, and treatment_names.
- Non-constant values for outcome, treatment_names, and confounders.
- No NaN values in used columns.
- Binary (0/1) encoding for treatment columns.
- One-hot treatment assignment (exactly one active treatment per row).
- A stable control treatment in position 0.
- No identical values between different columns.
- Unique values for user_id (if specified).

</details>

**Functions:**

- [**from_df**](#causalis.data_contracts.multicausaldata.MultiCausalData.from_df) – Create a MultiCausalData instance from a pandas DataFrame.
- [**get_df**](#causalis.data_contracts.multicausaldata.MultiCausalData.get_df) – Get a subset of the underlying DataFrame.

###### `FLOAT_TOL`

```python
FLOAT_TOL: float = 1e-12
```

###### `MAX_TREATMENTS`

```python
MAX_TREATMENTS: int = 15
```

###### `X`

```python
X: pd.DataFrame
```

Return the confounder columns as a pandas DataFrame.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The confounder columns.

###### `confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

###### `control_treatment`

```python
control_treatment: str
```

###### `df`

```python
df: pd.DataFrame
```

###### `from_df`

```python
from_df(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment_names: Union[str, List[str]],
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    control_treatment: str,
    **kwargs: Any
) -> "MultiCausalData"
```

Create a MultiCausalData instance from a pandas DataFrame.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The input DataFrame.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome column.
- **treatment_names** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – The name(s) of the treatment column(s).
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – The name(s) of the confounder column(s), by default None.
- **user_id** (<code>[str](#str)</code>) – The name of the user ID column, by default None.
- **control_treatment** (<code>[str](#str)</code>) – Name of the control treatment column.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional keyword arguments passed to the constructor.

**Returns:**

- <code>[MultiCausalData](#causalis.data_contracts.multicausaldata.MultiCausalData)</code> – An instance of MultiCausalData.

###### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_treatments: bool = True,
    include_user_id: bool = False,
) -> pd.DataFrame
```

Get a subset of the underlying DataFrame.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific columns to include, by default None.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column, by default True.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns, by default True.
- **include_treatments** (<code>[bool](#bool)</code>) – Whether to include treatment columns, by default True.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user ID column, by default False.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the requested DataFrame subset.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any of the requested columns do not exist.

###### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
```

###### `outcome`

```python
outcome: str
```

###### `treatment`

```python
treatment: pd.Series
```

Return the single treatment column as a pandas Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

**Raises:**

- <code>[AttributeError](#AttributeError)</code> – If there is more than one treatment column.

###### `treatment_names`

```python
treatment_names: List[str]
```

###### `treatments`

```python
treatments: pd.DataFrame
```

Return the treatment columns as a pandas DataFrame.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The treatment columns.

###### `user_id`

```python
user_id: Optional[str] = None
```

#### `panel_data_scm`

**Classes:**

- [**PanelDataSCM**](#causalis.data_contracts.panel_data_scm.PanelDataSCM) – Validated long-format panel contract for Synthetic Control estimators.

##### `PanelDataSCM`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Validated long-format panel contract for Synthetic Control estimators.

<details class="required-fields" open markdown="1">
<summary>Required fields</summary>

df : pandas.DataFrame
Long-format panel data.
y : str
Outcome column name in `df`.
unit_col : str
Unit identifier column name in `df`.
time_col : str
Calendar time column name in `df`.
Preferred input format is `pandas.Period` values with a regular
frequency (for example monthly `Period['M']`). Datetime/string values
are accepted only when a regular frequency can be inferred.
treated_time : str
Binary treatment-assignment column in `df` (0/1 or False/True).

</details>

<details class="note" open markdown="1">
<summary>Notes</summary>

There are no optional contract fields. Extra keyword arguments are rejected.
The contract derives `treated_unit`, `treatment_start`, and `time_freq`
from the input data.
The model stores a validated internal dataframe snapshot used by all contract
methods; mutating the public `df` attribute after construction does not
affect validated contract behavior.
Outcome `y` must not contain null/NaN values. Represent missing panel
periods by omitting unit-time rows, not by keeping rows with `NaN` outcome.
For fiscal quarter/year semantics, pass `time_col` explicitly as
`pandas.Period` with the desired fiscal frequency.

</details>

**Functions:**

- [**analysis_times**](#causalis.data_contracts.panel_data_scm.PanelDataSCM.analysis_times) –
- [**df_analysis**](#causalis.data_contracts.panel_data_scm.PanelDataSCM.df_analysis) –
- [**donor_pool**](#causalis.data_contracts.panel_data_scm.PanelDataSCM.donor_pool) –
- [**post_times**](#causalis.data_contracts.panel_data_scm.PanelDataSCM.post_times) –
- [**pre_times**](#causalis.data_contracts.panel_data_scm.PanelDataSCM.pre_times) –
- [**time_to_index**](#causalis.data_contracts.panel_data_scm.PanelDataSCM.time_to_index) –
- [**treatment_start_idx**](#causalis.data_contracts.panel_data_scm.PanelDataSCM.treatment_start_idx) –

###### `analysis_times`

```python
analysis_times() -> Sequence[pd.Period]
```

###### `df`

```python
df: pd.DataFrame = Field(..., description='Long-format panel data.')
```

###### `df_analysis`

```python
df_analysis() -> pd.DataFrame
```

###### `donor_pool`

```python
donor_pool() -> Sequence[Hashable]
```

###### `last_post_period`

```python
last_post_period: pd.Period
```

###### `model_config`

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True, extra="forbid", frozen=True
)

```

###### `n_post_periods`

```python
n_post_periods: int
```

###### `n_pre_periods`

```python
n_pre_periods: int
```

###### `post_times`

```python
post_times() -> Sequence[pd.Period]
```

###### `pre_times`

```python
pre_times() -> Sequence[pd.Period]
```

###### `time_col`

```python
time_col: str = Field(..., description='Calendar time column in df.')
```

###### `time_freq`

```python
time_freq: str
```

###### `time_to_index`

```python
time_to_index() -> dict[pd.Period, int]
```

###### `treated_time`

```python
treated_time: str = Field(
    ..., description="Binary treatment-assignment column in df."
)

```

###### `treated_unit`

```python
treated_unit: Hashable
```

###### `treatment_start`

```python
treatment_start: pd.Period
```

###### `treatment_start_idx`

```python
treatment_start_idx() -> int
```

###### `unit_col`

```python
unit_col: str = Field(..., description='Unit identifier column in df.')
```

###### `y`

```python
y: str = Field(..., description='Outcome column in df.')
```

##### `TimeLike`

```python
TimeLike = Union[str, date, datetime, pd.Timestamp, pd.Period]
```

#### `panel_estimate`

**Classes:**

- [**PanelEstimate**](#causalis.data_contracts.panel_estimate.PanelEstimate) – Result contract for dynamic synthetic-control effect-path estimates.

##### `PanelEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result contract for dynamic synthetic-control effect-path estimates.

**Functions:**

- [**summary**](#causalis.data_contracts.panel_estimate.PanelEstimate.summary) – Return a compact CausalEstimate-style summary table.
- [**summary_poinwise**](#causalis.data_contracts.panel_estimate.PanelEstimate.summary_poinwise) – Return pointwise post-period estimates as a flat DataFrame.

###### `alpha`

```python
alpha: float
```

###### `ci_lower_by_time`

```python
ci_lower_by_time: pd.Series
```

###### `ci_upper_by_time`

```python
ci_upper_by_time: pd.Series
```

###### `confidence_set_by_time`

```python
confidence_set_by_time: Dict[TimeLike, list[tuple[float, float]]]
```

###### `created_at`

```python
created_at: datetime = Field(
    default_factory=(lambda: datetime.now(timezone.utc))
)

```

###### `diagnostics`

```python
diagnostics: Dict[str, Any] = Field(default_factory=dict)
```

###### `donor_weights_augmented`

```python
donor_weights_augmented: Dict[Hashable, float]
```

###### `effect_by_time`

```python
effect_by_time: pd.Series
```

###### `estimand`

```python
estimand: Literal['dynamic_effect_path'] = 'dynamic_effect_path'
```

###### `is_significant_by_time`

```python
is_significant_by_time: pd.Series
```

###### `model`

```python
model: str
```

###### `model_config`

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
)

```

###### `observed_outcome`

```python
observed_outcome: pd.Series
```

###### `p_value_by_time`

```python
p_value_by_time: pd.Series
```

###### `post_times`

```python
post_times: List[TimeLike]
```

###### `pre_times`

```python
pre_times: List[TimeLike]
```

###### `summary`

```python
summary() -> pd.DataFrame
```

Return a compact CausalEstimate-style summary table.

###### `summary_poinwise`

```python
summary_poinwise() -> pd.DataFrame
```

Return pointwise post-period estimates as a flat DataFrame.

###### `synthetic_outcome`

```python
synthetic_outcome: pd.Series
```

###### `treated_unit`

```python
treated_unit: Hashable
```

###### `treatment_start`

```python
treatment_start: TimeLike
```

### `causalis.dgp`

**Modules:**

- [**base**](#causalis.dgp.base) –
- [**causaldata**](#causalis.dgp.causaldata) –
- [**causaldata_instrumental**](#causalis.dgp.causaldata_instrumental) –
- [**multicausaldata**](#causalis.dgp.multicausaldata) –
- [**panel_data_scm**](#causalis.dgp.panel_data_scm) –

**Classes:**

- [**CausalDatasetGenerator**](#causalis.dgp.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,

**Functions:**

- [**classic_rct_gamma**](#causalis.dgp.classic_rct_gamma) – Generate a classic RCT dataset with three binary confounders and a gamma outcome.
- [**classic_rct_gamma_26**](#causalis.dgp.classic_rct_gamma_26) – A pre-configured classic RCT dataset with a gamma outcome.
- [**generate_classic_rct**](#causalis.dgp.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.dgp.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_cuped_binary**](#causalis.dgp.generate_cuped_binary) – Binary CUPED-oriented DGP with richer confounders and structured HTE.
- [**generate_cuped_tweedie_26**](#causalis.dgp.generate_cuped_tweedie_26) – Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
- [**generate_iv_data**](#causalis.dgp.generate_iv_data) – Generate synthetic dataset with instrumental variables.
- [**generate_rct**](#causalis.dgp.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**generate_scm_data**](#causalis.dgp.generate_scm_data) – Medium-level wrapper for Gaussian SCM panel generation.
- [**make_cuped_binary_26**](#causalis.dgp.make_cuped_binary_26) – Binary CUPED benchmark with richer confounders and structured HTE.
- [**make_cuped_tweedie**](#causalis.dgp.make_cuped_tweedie) – Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_gold_linear**](#causalis.dgp.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.dgp.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.dgp.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

#### `CausalDatasetGenerator`

```python
CausalDatasetGenerator(
    theta: float = 1.0,
    tau: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha_y: float = 0.0,
    alpha_d: float = 0.0,
    sigma_y: float = 1.0,
    outcome_type: str = "continuous",
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    use_copula: bool = False,
    copula_corr: Optional[np.ndarray] = None,
    target_d_rate: Optional[float] = None,
    u_strength_d: float = 0.0,
    u_strength_y: float = 0.0,
    propensity_sharpness: float = 1.0,
    score_bounding: Optional[float] = None,
    alpha_zi: float = -1.0,
    beta_zi: Optional[np.ndarray] = None,
    g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    u_strength_zi: float = 0.0,
    tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    pos_dist: str = "gamma",
    gamma_shape: float = 2.0,
    lognormal_sigma: float = 1.0,
    include_oracle: bool = True,
    seed: Optional[int] = None,
) -> None
```

Generate synthetic causal inference datasets with controllable confounding,
treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

**Data model (high level)**

- confounders X ∈ R^k are drawn from user-specified distributions.
- Binary treatment D is assigned by a logistic model:
  D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
  where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
- Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
  outcome_type = "continuous":
  Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε, ε ~ N(0, sigma_y^2)
  outcome_type = "binary":
  logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "poisson":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "gamma":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

**Returned columns**

- y: outcome
- d: binary treatment (0/1)
- x1..xk (or user-provided names)
- m: true propensity P(T=1 | X) marginalized over U
- m_obs: realized propensity P(T=1 | X, U)
- tau_link: tau(X) on the structural (link) scale
- g0: E[Y | X, T=0] on the natural outcome scale marginalized over U .,9
- g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
- cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

Notes on effect scale:

- For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
- For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
- For "poisson" and "gamma", tau acts on the *log-mean* scale. `cate` is reported on the mean scale.

**Parameters:**

- **theta** (<code>[float](#float)</code>) – Constant treatment effect used if `tau` is None.
- **tau** (<code>[callable](#callable)</code>) – Function tau(X) -> array-like shape (n,) for heterogeneous effects.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the outcome baseline f_y(X).
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the treatment score f_d(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
- **g_d** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson/Gamma).
- **alpha_d** (<code>[float](#float)</code>) – Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
- **sigma_y** (<code>[float](#float)</code>) – Std. dev. of the Gaussian noise for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma', 'tweedie')</code>) – Outcome family and link as defined above.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders. See `_gaussian_copula` for details.
- **k** (<code>[int](#int)</code>) – Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
- **use_copula** (<code>[bool](#bool)</code>) – If True and `confounder_specs` provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
- **u_strength_d** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in treatment assignment.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in the outcome.
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales the X-driven treatment score to adjust positivity difficulty.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Attributes:**

- [**rng**](#causalis.dgp.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.dgp.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

##### `alpha_d`

```python
alpha_d: float = 0.0
```

##### `alpha_y`

```python
alpha_y: float = 0.0
```

##### `alpha_zi`

```python
alpha_zi: float = -1.0
```

##### `beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

##### `beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

##### `beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

##### `confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

##### `copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

##### `g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `gamma_shape`

```python
gamma_shape: float = 2.0
```

##### `generate`

```python
generate(n: int, U: Optional[np.ndarray] = None) -> pd.DataFrame
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

##### `include_oracle`

```python
include_oracle: bool = True
```

##### `k`

```python
k: int = 5
```

##### `lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

##### `oracle_nuisance`

```python
oracle_nuisance(num_quad: int = 21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

##### `outcome_type`

```python
outcome_type: str = 'continuous'
```

##### `pos_dist`

```python
pos_dist: str = 'gamma'
```

##### `propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

##### `rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

##### `score_bounding`

```python
score_bounding: Optional[float] = None
```

##### `seed`

```python
seed: Optional[int] = None
```

##### `sigma_y`

```python
sigma_y: float = 1.0
```

##### `target_d_rate`

```python
target_d_rate: Optional[float] = None
```

##### `tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `theta`

```python
theta: float = 1.0
```

##### `to_causal_data`

```python
to_causal_data(
    n: int, confounders: Optional[Union[str, List[str]]] = None
) -> CausalData
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

##### `u_strength_d`

```python
u_strength_d: float = 0.0
```

##### `u_strength_y`

```python
u_strength_y: float = 0.0
```

##### `u_strength_zi`

```python
u_strength_zi: float = 0.0
```

##### `use_copula`

```python
use_copula: bool = False
```

##### `x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

#### `base`

**Functions:**

- [**estimate_gaussian_copula_corr**](#causalis.dgp.base.estimate_gaussian_copula_corr) – Estimate a Gaussian copula correlation matrix from observational data_contracts.

##### `estimate_gaussian_copula_corr`

```python
estimate_gaussian_copula_corr(df: pd.DataFrame, cols: list[str]) -> np.ndarray
```

Estimate a Gaussian copula correlation matrix from observational data_contracts.
Uses rank -> normal scores -> Pearson correlation approach.

#### `causaldata`

**Modules:**

- [**base**](#causalis.dgp.causaldata.base) –
- [**functional**](#causalis.dgp.causaldata.functional) –
- [**preperiod**](#causalis.dgp.causaldata.preperiod) –

**Classes:**

- [**CausalData**](#causalis.dgp.causaldata.CausalData) – Container for causal inference datasets.
- [**CausalDatasetGenerator**](#causalis.dgp.causaldata.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,

**Functions:**

- [**classic_rct_gamma**](#causalis.dgp.causaldata.classic_rct_gamma) – Generate a classic RCT dataset with three binary confounders and a gamma outcome.
- [**classic_rct_gamma_26**](#causalis.dgp.causaldata.classic_rct_gamma_26) – A pre-configured classic RCT dataset with a gamma outcome.
- [**generate_classic_rct**](#causalis.dgp.causaldata.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.dgp.causaldata.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_cuped_binary**](#causalis.dgp.causaldata.generate_cuped_binary) – Binary CUPED-oriented DGP with richer confounders and structured HTE.
- [**generate_cuped_tweedie_26**](#causalis.dgp.causaldata.generate_cuped_tweedie_26) – Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
- [**generate_obs_hte_26**](#causalis.dgp.causaldata.generate_obs_hte_26) – Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
- [**generate_obs_hte_26_rich**](#causalis.dgp.causaldata.generate_obs_hte_26_rich) – Observational dataset with richer confounding, nonlinear outcome model,
- [**generate_obs_hte_binary_26**](#causalis.dgp.causaldata.generate_obs_hte_binary_26) – Observational binary-outcome dataset with nonlinear confounding and
- [**generate_rct**](#causalis.dgp.causaldata.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_cuped_binary_26**](#causalis.dgp.causaldata.make_cuped_binary_26) – Binary CUPED benchmark with richer confounders and structured HTE.
- [**make_cuped_tweedie**](#causalis.dgp.causaldata.make_cuped_tweedie) – Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_gold_linear**](#causalis.dgp.causaldata.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.dgp.causaldata.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.dgp.causaldata.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

##### `CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data_contracts contract.

**Attributes:**

- [**df**](#causalis.dgp.causaldata.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.dgp.causaldata.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.dgp.causaldata.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.dgp.causaldata.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.dgp.causaldata.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.dgp.causaldata.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.dgp.causaldata.CausalData.get_df) – Get a DataFrame with specified columns.

###### `X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

###### `confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

###### `confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

###### `df`

```python
df: pd.DataFrame
```

###### `from_df`

```python
from_df(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    **kwargs: Any
) -> "CausalData"
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code> – A validated CausalData instance.

###### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_treatment: bool = True,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_user_id: bool = False,
) -> pd.DataFrame
```

Get a DataFrame with specified columns.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

###### `model_config`

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True, populate_by_name=True, extra="forbid"
)

```

###### `outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

###### `outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

###### `treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

###### `treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

###### `user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

###### `user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

##### `CausalDatasetGenerator`

```python
CausalDatasetGenerator(
    theta: float = 1.0,
    tau: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha_y: float = 0.0,
    alpha_d: float = 0.0,
    sigma_y: float = 1.0,
    outcome_type: str = "continuous",
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    use_copula: bool = False,
    copula_corr: Optional[np.ndarray] = None,
    target_d_rate: Optional[float] = None,
    u_strength_d: float = 0.0,
    u_strength_y: float = 0.0,
    propensity_sharpness: float = 1.0,
    score_bounding: Optional[float] = None,
    alpha_zi: float = -1.0,
    beta_zi: Optional[np.ndarray] = None,
    g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    u_strength_zi: float = 0.0,
    tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    pos_dist: str = "gamma",
    gamma_shape: float = 2.0,
    lognormal_sigma: float = 1.0,
    include_oracle: bool = True,
    seed: Optional[int] = None,
) -> None
```

Generate synthetic causal inference datasets with controllable confounding,
treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

**Data model (high level)**

- confounders X ∈ R^k are drawn from user-specified distributions.
- Binary treatment D is assigned by a logistic model:
  D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
  where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
- Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
  outcome_type = "continuous":
  Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε, ε ~ N(0, sigma_y^2)
  outcome_type = "binary":
  logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "poisson":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "gamma":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

**Returned columns**

- y: outcome
- d: binary treatment (0/1)
- x1..xk (or user-provided names)
- m: true propensity P(T=1 | X) marginalized over U
- m_obs: realized propensity P(T=1 | X, U)
- tau_link: tau(X) on the structural (link) scale
- g0: E[Y | X, T=0] on the natural outcome scale marginalized over U .,9
- g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
- cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

Notes on effect scale:

- For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
- For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
- For "poisson" and "gamma", tau acts on the *log-mean* scale. `cate` is reported on the mean scale.

**Parameters:**

- **theta** (<code>[float](#float)</code>) – Constant treatment effect used if `tau` is None.
- **tau** (<code>[callable](#callable)</code>) – Function tau(X) -> array-like shape (n,) for heterogeneous effects.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the outcome baseline f_y(X).
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the treatment score f_d(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
- **g_d** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson/Gamma).
- **alpha_d** (<code>[float](#float)</code>) – Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
- **sigma_y** (<code>[float](#float)</code>) – Std. dev. of the Gaussian noise for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma', 'tweedie')</code>) – Outcome family and link as defined above.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders. See `_gaussian_copula` for details.
- **k** (<code>[int](#int)</code>) – Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
- **use_copula** (<code>[bool](#bool)</code>) – If True and `confounder_specs` provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
- **u_strength_d** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in treatment assignment.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in the outcome.
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales the X-driven treatment score to adjust positivity difficulty.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Attributes:**

- [**rng**](#causalis.dgp.causaldata.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.dgp.causaldata.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.causaldata.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.causaldata.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

###### `alpha_d`

```python
alpha_d: float = 0.0
```

###### `alpha_y`

```python
alpha_y: float = 0.0
```

###### `alpha_zi`

```python
alpha_zi: float = -1.0
```

###### `beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

###### `beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

###### `beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

###### `confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

###### `copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

###### `g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `gamma_shape`

```python
gamma_shape: float = 2.0
```

###### `generate`

```python
generate(n: int, U: Optional[np.ndarray] = None) -> pd.DataFrame
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

###### `include_oracle`

```python
include_oracle: bool = True
```

###### `k`

```python
k: int = 5
```

###### `lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

###### `oracle_nuisance`

```python
oracle_nuisance(num_quad: int = 21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

###### `outcome_type`

```python
outcome_type: str = 'continuous'
```

###### `pos_dist`

```python
pos_dist: str = 'gamma'
```

###### `propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

###### `rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

###### `score_bounding`

```python
score_bounding: Optional[float] = None
```

###### `seed`

```python
seed: Optional[int] = None
```

###### `sigma_y`

```python
sigma_y: float = 1.0
```

###### `target_d_rate`

```python
target_d_rate: Optional[float] = None
```

###### `tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `theta`

```python
theta: float = 1.0
```

###### `to_causal_data`

```python
to_causal_data(
    n: int, confounders: Optional[Union[str, List[str]]] = None
) -> CausalData
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

###### `u_strength_d`

```python
u_strength_d: float = 0.0
```

###### `u_strength_y`

```python
u_strength_y: float = 0.0
```

###### `u_strength_zi`

```python
u_strength_zi: float = 0.0
```

###### `use_copula`

```python
use_copula: bool = False
```

###### `x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `base`

**Classes:**

- [**CausalDatasetGenerator**](#causalis.dgp.causaldata.base.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,

###### `CausalDatasetGenerator`

```python
CausalDatasetGenerator(
    theta: float = 1.0,
    tau: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha_y: float = 0.0,
    alpha_d: float = 0.0,
    sigma_y: float = 1.0,
    outcome_type: str = "continuous",
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    use_copula: bool = False,
    copula_corr: Optional[np.ndarray] = None,
    target_d_rate: Optional[float] = None,
    u_strength_d: float = 0.0,
    u_strength_y: float = 0.0,
    propensity_sharpness: float = 1.0,
    score_bounding: Optional[float] = None,
    alpha_zi: float = -1.0,
    beta_zi: Optional[np.ndarray] = None,
    g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    u_strength_zi: float = 0.0,
    tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    pos_dist: str = "gamma",
    gamma_shape: float = 2.0,
    lognormal_sigma: float = 1.0,
    include_oracle: bool = True,
    seed: Optional[int] = None,
) -> None
```

Generate synthetic causal inference datasets with controllable confounding,
treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

**Data model (high level)**

- confounders X ∈ R^k are drawn from user-specified distributions.
- Binary treatment D is assigned by a logistic model:
  D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
  where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
- Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
  outcome_type = "continuous":
  Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε, ε ~ N(0, sigma_y^2)
  outcome_type = "binary":
  logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "poisson":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "gamma":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

**Returned columns**

- y: outcome
- d: binary treatment (0/1)
- x1..xk (or user-provided names)
- m: true propensity P(T=1 | X) marginalized over U
- m_obs: realized propensity P(T=1 | X, U)
- tau_link: tau(X) on the structural (link) scale
- g0: E[Y | X, T=0] on the natural outcome scale marginalized over U .,9
- g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
- cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

Notes on effect scale:

- For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
- For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
- For "poisson" and "gamma", tau acts on the *log-mean* scale. `cate` is reported on the mean scale.

**Parameters:**

- **theta** (<code>[float](#float)</code>) – Constant treatment effect used if `tau` is None.
- **tau** (<code>[callable](#callable)</code>) – Function tau(X) -> array-like shape (n,) for heterogeneous effects.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the outcome baseline f_y(X).
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the treatment score f_d(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
- **g_d** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson/Gamma).
- **alpha_d** (<code>[float](#float)</code>) – Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
- **sigma_y** (<code>[float](#float)</code>) – Std. dev. of the Gaussian noise for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma', 'tweedie')</code>) – Outcome family and link as defined above.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders. See `_gaussian_copula` for details.
- **k** (<code>[int](#int)</code>) – Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
- **use_copula** (<code>[bool](#bool)</code>) – If True and `confounder_specs` provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
- **u_strength_d** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in treatment assignment.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in the outcome.
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales the X-driven treatment score to adjust positivity difficulty.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Attributes:**

- [**rng**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

####### `alpha_d`

```python
alpha_d: float = 0.0
```

####### `alpha_y`

```python
alpha_y: float = 0.0
```

####### `alpha_zi`

```python
alpha_zi: float = -1.0
```

####### `beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

####### `beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

####### `beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

####### `confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

####### `copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

####### `g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `gamma_shape`

```python
gamma_shape: float = 2.0
```

####### `generate`

```python
generate(n: int, U: Optional[np.ndarray] = None) -> pd.DataFrame
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

####### `include_oracle`

```python
include_oracle: bool = True
```

####### `k`

```python
k: int = 5
```

####### `lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

####### `oracle_nuisance`

```python
oracle_nuisance(num_quad: int = 21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

####### `outcome_type`

```python
outcome_type: str = 'continuous'
```

####### `pos_dist`

```python
pos_dist: str = 'gamma'
```

####### `propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

####### `rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

####### `score_bounding`

```python
score_bounding: Optional[float] = None
```

####### `seed`

```python
seed: Optional[int] = None
```

####### `sigma_y`

```python
sigma_y: float = 1.0
```

####### `target_d_rate`

```python
target_d_rate: Optional[float] = None
```

####### `tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `theta`

```python
theta: float = 1.0
```

####### `to_causal_data`

```python
to_causal_data(
    n: int, confounders: Optional[Union[str, List[str]]] = None
) -> CausalData
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

####### `u_strength_d`

```python
u_strength_d: float = 0.0
```

####### `u_strength_y`

```python
u_strength_y: float = 0.0
```

####### `u_strength_zi`

```python
u_strength_zi: float = 0.0
```

####### `use_copula`

```python
use_copula: bool = False
```

####### `x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `classic_rct_gamma`

```python
classic_rct_gamma(
    n: int = 10000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    **kwargs: bool
) -> Union[pd.DataFrame, CausalData]
```

Generate a classic RCT dataset with three binary confounders and a gamma outcome.

The gamma outcome uses a log-mean link, so treatment effects are multiplicative
on the mean scale. The default parameters are chosen to resemble a skewed
real-world metric (e.g., spend or revenue).

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Gamma parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
  Mean = shape * scale.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the log-mean outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre with post-outcome in control group.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- \*\***kwargs** – Additional arguments passed to `generate_rct` (e.g., pre_name, g_y, use_prognostic).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset with gamma outcome.

##### `classic_rct_gamma_26`

```python
classic_rct_gamma_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = True,
    **kwargs: bool
)
```

A pre-configured classic RCT dataset with a gamma outcome.
n=10000, split=0.5, mean uplift ~10%.
Includes deterministic `user_id` and ancillary columns.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre').
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **n** (<code>[int](#int)</code>) – Number of samples.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **outcome_params** (<code>[dict](#dict)</code>) – Gamma outcome parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- \*\***kwargs** – Additional arguments passed to `classic_rct_gamma`.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

##### `functional`

**Functions:**

- [**classic_rct_gamma**](#causalis.dgp.causaldata.functional.classic_rct_gamma) – Generate a classic RCT dataset with three binary confounders and a gamma outcome.
- [**generate_classic_rct**](#causalis.dgp.causaldata.functional.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_cuped_binary**](#causalis.dgp.causaldata.functional.generate_cuped_binary) – Binary CUPED-oriented DGP with richer confounders and structured HTE.
- [**generate_rct**](#causalis.dgp.causaldata.functional.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_cuped_tweedie**](#causalis.dgp.causaldata.functional.make_cuped_tweedie) – Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_gold_linear**](#causalis.dgp.causaldata.functional.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_effect**](#causalis.dgp.causaldata.functional.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

###### `classic_rct_gamma`

```python
classic_rct_gamma(
    n: int = 10000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    **kwargs: bool
) -> Union[pd.DataFrame, CausalData]
```

Generate a classic RCT dataset with three binary confounders and a gamma outcome.

The gamma outcome uses a log-mean link, so treatment effects are multiplicative
on the mean scale. The default parameters are chosen to resemble a skewed
real-world metric (e.g., spend or revenue).

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Gamma parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
  Mean = shape * scale.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the log-mean outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre with post-outcome in control group.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- \*\***kwargs** – Additional arguments passed to `generate_rct` (e.g., pre_name, g_y, use_prognostic).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset with gamma outcome.

###### `generate_classic_rct`

```python
generate_classic_rct(
    n: int = 10000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    return_causal_data: bool = False,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    **kwargs: bool
) -> Union[pd.DataFrame, CausalData]
```

Generate a classic RCT dataset with three binary confounders:
platform_ios, country_usa, and source_paid.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal (passed to generate_rct).
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre (passed to generate_rct).
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- \*\***kwargs** – Additional arguments passed to `generate_rct`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

###### `generate_cuped_binary`

```python
generate_cuped_binary(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Binary CUPED-oriented DGP with richer confounders and structured HTE.

Designed for CUPED benchmarking with randomized treatment and a calibrated
pre-period covariate while preserving exact oracle cate under include_oracle.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in the control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration.
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like m, g0, g1, cate.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_logit** (<code>[float](#float)</code>) – Baseline log-odds uplift scale for heterogeneous treatment effects.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

###### `generate_rct`

```python
generate_rct(
    n: int = 20000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_type: str = "binary",
    outcome_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_corr: float = 0.7,
    prognostic_scale: float = 1.0,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    use_prognostic: Optional[bool] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
) -> Union[pd.DataFrame, CausalData]
```

Generate an RCT dataset with randomized treatment assignment.

Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
Specifically designed for benchmarking variance reduction techniques like CUPED.

**Notes on effect scale**

How `outcome_params` maps into the structural effect:

- outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
- outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
- outcome_type="poisson" or "gamma": treatment shifts the log-mean by log(lam_B / lam_A).

Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
avoiding outcome leakage and post-treatment adjustment issues.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_type** (<code>('binary', 'normal', 'poisson', 'gamma')</code>) – Distribution family of the outcome.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.12}} for binary, or
  {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}} for poisson/gamma.
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add descriptive columns like 'age', 'platform', etc.
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation between `y_pre` and the outcome Y in the control group.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of the prognostic signal derived from confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic RCT dataset.

###### `make_cuped_tweedie`

```python
make_cuped_tweedie(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.6,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.2,
) -> Union[pd.DataFrame, CausalData]
```

Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail. Suitable for CUPED benchmarking.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

###### `make_gold_linear`

```python
make_gold_linear(n: int = 10000, seed: int = 42) -> CausalData
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

###### `obs_linear_effect`

```python
obs_linear_effect(
    n: int = 10000,
    theta: float = 1.0,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    target_d_rate: Optional[float] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    include_oracle: bool = True,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
) -> pd.DataFrame
```

Generate an observational dataset with linear effects of confounders and a constant treatment effect.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **theta** (<code>[float](#float)</code>) – Constant treatment effect.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma')</code>) – Family of the outcome distribution.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the treatment model.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **add_ancillary** (<code>[bool](#bool)</code>) – If True, adds standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – If True, generates deterministic user IDs.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic observational dataset.

##### `generate_classic_rct`

```python
generate_classic_rct(
    n: int = 10000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    return_causal_data: bool = False,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    **kwargs: bool
) -> Union[pd.DataFrame, CausalData]
```

Generate a classic RCT dataset with three binary confounders:
platform_ios, country_usa, and source_paid.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal (passed to generate_rct).
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre (passed to generate_rct).
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- \*\***kwargs** – Additional arguments passed to `generate_rct`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

##### `generate_classic_rct_26`

```python
generate_classic_rct_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = False,
    deterministic_ids: bool = True,
    **kwargs: bool
)
```

A pre-configured classic RCT dataset with 3 binary confounders.
n=10000, split=0.5, outcome is conversion (binary). Baseline control p=0.10
and treatment p=0.11 are set on the log-odds scale (X=0), so marginal rates
and ATE can differ once covariate effects are included. Includes a
deterministic `user_id` column.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **n** (<code>[int](#int)</code>) – Number of samples.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **outcome_params** (<code>[dict](#dict)</code>) – Binary outcome parameters, e.g. {"p": {"A": 0.10, "B": 0.11}}.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- \*\***kwargs** – Additional arguments passed to `generate_classic_rct`.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

##### `generate_cuped_binary`

```python
generate_cuped_binary(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Binary CUPED-oriented DGP with richer confounders and structured HTE.

Designed for CUPED benchmarking with randomized treatment and a calibrated
pre-period covariate while preserving exact oracle cate under include_oracle.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in the control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration.
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like m, g0, g1, cate.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_logit** (<code>[float](#float)</code>) – Baseline log-odds uplift scale for heterogeneous treatment effects.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

##### `generate_cuped_tweedie_26`

```python
generate_cuped_tweedie_26(
    n: int = 20000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_name_2: Optional[str] = None,
    pre_target_corr: float = 0.82,
    pre_target_corr_2: Optional[float] = None,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail.
Includes two pre-period covariates by default: 'y_pre' and 'y_pre_2'.
Wrapper for make_tweedie().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add pre-period covariates.
- **pre_name** (<code>[str](#str)</code>) – Name of the first pre-period covariate column.
- **pre_name_2** (<code>[str](#str)</code>) – Name of the second pre-period covariate column.
  Defaults to `f"{pre_name}_2"`.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between the first pre covariate and post-outcome y in control group.
- **pre_target_corr_2** (<code>[float](#float)</code>) – Target correlation for the second pre covariate. Defaults to a
  moderate value based on `pre_target_corr` to reduce collinearity.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

##### `generate_obs_hte_26`

```python
generate_obs_hte_26(
    n: int = 10000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, CausalData]
```

Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
and a heterogeneous (nonlinear) treatment effect tau(X).
Based on the scenario in notebooks/cases/dml_atte.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

##### `generate_obs_hte_26_rich`

```python
generate_obs_hte_26_rich(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, CausalData]
```

Observational dataset with richer confounding, nonlinear outcome model,
nonlinear treatment assignment, and heterogeneous treatment effects.
Adds additional realistic covariates and dependencies to mimic real data.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

##### `generate_obs_hte_binary_26`

```python
generate_obs_hte_binary_26(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, CausalData]
```

Observational binary-outcome dataset with nonlinear confounding and
heterogeneous treatment effects.

This scenario follows the structure of `generate_obs_hte_26_rich`, but uses
a binary outcome model and a modified confounder set.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

##### `generate_rct`

```python
generate_rct(
    n: int = 20000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_type: str = "binary",
    outcome_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_corr: float = 0.7,
    prognostic_scale: float = 1.0,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    use_prognostic: Optional[bool] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
) -> Union[pd.DataFrame, CausalData]
```

Generate an RCT dataset with randomized treatment assignment.

Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
Specifically designed for benchmarking variance reduction techniques like CUPED.

**Notes on effect scale**

How `outcome_params` maps into the structural effect:

- outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
- outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
- outcome_type="poisson" or "gamma": treatment shifts the log-mean by log(lam_B / lam_A).

Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
avoiding outcome leakage and post-treatment adjustment issues.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_type** (<code>('binary', 'normal', 'poisson', 'gamma')</code>) – Distribution family of the outcome.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.12}} for binary, or
  {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}} for poisson/gamma.
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add descriptive columns like 'age', 'platform', etc.
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation between `y_pre` and the outcome Y in the control group.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of the prognostic signal derived from confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic RCT dataset.

##### `make_cuped_binary_26`

```python
make_cuped_binary_26(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Binary CUPED benchmark with richer confounders and structured HTE.
Includes a calibrated pre-period covariate 'y_pre' by default.
Wrapper for generate_cuped_binary().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in the control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like 'cate', 'g0', and 'g1'.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_logit** (<code>[float](#float)</code>) – Baseline log-odds uplift scale for heterogeneous treatment effects.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

##### `make_cuped_tweedie`

```python
make_cuped_tweedie(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.6,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.2,
) -> Union[pd.DataFrame, CausalData]
```

Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail. Suitable for CUPED benchmarking.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

##### `make_gold_linear`

```python
make_gold_linear(n: int = 10000, seed: int = 42) -> CausalData
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

##### `obs_linear_26_dataset`

```python
obs_linear_26_dataset(
    n: int = 10000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
)
```

A pre-configured observational linear dataset with 5 standard confounders.
Based on the scenario in docs/cases/dml_ate.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

##### `obs_linear_effect`

```python
obs_linear_effect(
    n: int = 10000,
    theta: float = 1.0,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    target_d_rate: Optional[float] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    include_oracle: bool = True,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
) -> pd.DataFrame
```

Generate an observational dataset with linear effects of confounders and a constant treatment effect.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **theta** (<code>[float](#float)</code>) – Constant treatment effect.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma')</code>) – Family of the outcome distribution.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the treatment model.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **add_ancillary** (<code>[bool](#bool)</code>) – If True, adds standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – If True, generates deterministic user IDs.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic observational dataset.

##### `preperiod`

**Classes:**

- [**PreCorrSpec**](#causalis.dgp.causaldata.preperiod.PreCorrSpec) –

**Functions:**

- [**add_preperiod_covariate**](#causalis.dgp.causaldata.preperiod.add_preperiod_covariate) – Standardized utility to add a calibrated pre-period covariate to a DataFrame.
- [**calibrate_sigma_for_target_corr**](#causalis.dgp.causaldata.preperiod.calibrate_sigma_for_target_corr) – Find sigma such that Corr(T(y_pre_base + sigma\*eps), T(y_post)) ~ target_corr.
- [**corr_on_scale**](#causalis.dgp.causaldata.preperiod.corr_on_scale) –

###### `CorrMethod`

```python
CorrMethod = Literal['pearson', 'spearman']
```

###### `PreCorrSpec`

```python
PreCorrSpec(
    target_corr: float = 0.7,
    transform: Transform = "log1p",
    winsor_q: Optional[float] = 0.999,
    method: CorrMethod = "pearson",
    sigma_lo: float = 0.0,
    sigma_hi: float = 50.0,
    sigma_tol: float = 0.001,
    max_iter: int = 40,
) -> None
```

####### `max_iter`

```python
max_iter: int = 40
```

####### `method`

```python
method: CorrMethod = 'pearson'
```

####### `sigma_hi`

```python
sigma_hi: float = 50.0
```

####### `sigma_lo`

```python
sigma_lo: float = 0.0
```

####### `sigma_tol`

```python
sigma_tol: float = 0.001
```

####### `target_corr`

```python
target_corr: float = 0.7
```

####### `transform`

```python
transform: Transform = 'log1p'
```

####### `winsor_q`

```python
winsor_q: Optional[float] = 0.999
```

###### `Transform`

```python
Transform = Literal['none', 'log1p', 'rank']
```

###### `add_preperiod_covariate`

```python
add_preperiod_covariate(
    df: pd.DataFrame,
    y_col: str,
    d_col: str,
    pre_name: str,
    base_builder: Callable[[pd.DataFrame], np.ndarray],
    spec: PreCorrSpec,
    rng: np.random.Generator,
    mask: Optional[np.ndarray] = None,
) -> pd.DataFrame
```

Standardized utility to add a calibrated pre-period covariate to a DataFrame.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The dataset.
- **y_col** (<code>[str](#str)</code>) – Name of the outcome column.
- **d_col** (<code>[str](#str)</code>) – Name of the treatment column.
- **pre_name** (<code>[str](#str)</code>) – Name of the new pre-period covariate column.
- **base_builder** (<code>[callable](#callable)</code>) – Function df -> y_pre_base (np.ndarray) providing the shared signal.
- **spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Specification for target correlation and scale.
- **rng** (<code>[Generator](#numpy.random.Generator)</code>) – Random number generator.
- **mask** (<code>[ndarray](#numpy.ndarray)</code>) – Boolean mask of rows to use for calibration (e.g. control group).
  If None, use control group (d == 0).

###### `calibrate_sigma_for_target_corr`

```python
calibrate_sigma_for_target_corr(
    y_pre_base: np.ndarray,
    y_post: np.ndarray,
    rng: np.random.Generator,
    spec: PreCorrSpec,
    *,
    noise: Optional[np.ndarray] = None
) -> Tuple[float, float]
```

Find sigma such that Corr(T(y_pre_base + sigma\*eps), T(y_post)) ~ target_corr.
Returns (sigma, achieved_corr).

###### `corr_on_scale`

```python
corr_on_scale(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    *,
    transform: Transform = "log1p",
    winsor_q: Optional[float] = 0.999,
    method: CorrMethod = "pearson"
) -> float
```

#### `causaldata_instrumental`

**Modules:**

- [**base**](#causalis.dgp.causaldata_instrumental.base) –
- [**functional**](#causalis.dgp.causaldata_instrumental.functional) –

**Classes:**

- [**InstrumentalGenerator**](#causalis.dgp.causaldata_instrumental.InstrumentalGenerator) – Generator for synthetic causal inference datasets with instrumental variables.

**Functions:**

- [**generate_iv_data**](#causalis.dgp.causaldata_instrumental.generate_iv_data) – Generate synthetic dataset with instrumental variables.

##### `InstrumentalGenerator`

```python
InstrumentalGenerator(seed: Optional[int] = None) -> None
```

Generator for synthetic causal inference datasets with instrumental variables.

Placeholder implementation for future use.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Functions:**

- [**generate**](#causalis.dgp.causaldata_instrumental.InstrumentalGenerator.generate) – Draw a synthetic dataset of size `n`.

###### `generate`

```python
generate(n: int) -> pd.DataFrame
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – An empty DataFrame (placeholder).

###### `seed`

```python
seed: Optional[int] = None
```

##### `base`

**Classes:**

- [**InstrumentalGenerator**](#causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator) – Generator for synthetic causal inference datasets with instrumental variables.

###### `InstrumentalGenerator`

```python
InstrumentalGenerator(seed: Optional[int] = None) -> None
```

Generator for synthetic causal inference datasets with instrumental variables.

Placeholder implementation for future use.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Functions:**

- [**generate**](#causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator.generate) – Draw a synthetic dataset of size `n`.

####### `generate`

```python
generate(n: int) -> pd.DataFrame
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – An empty DataFrame (placeholder).

####### `seed`

```python
seed: Optional[int] = None
```

##### `functional`

**Functions:**

- [**generate_iv_data**](#causalis.dgp.causaldata_instrumental.functional.generate_iv_data) – Generate synthetic dataset with instrumental variables.

###### `generate_iv_data`

```python
generate_iv_data(n: int = 1000) -> pd.DataFrame
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

##### `generate_iv_data`

```python
generate_iv_data(n: int = 1000) -> pd.DataFrame
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

#### `classic_rct_gamma`

```python
classic_rct_gamma(
    n: int = 10000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    **kwargs: bool
) -> Union[pd.DataFrame, CausalData]
```

Generate a classic RCT dataset with three binary confounders and a gamma outcome.

The gamma outcome uses a log-mean link, so treatment effects are multiplicative
on the mean scale. The default parameters are chosen to resemble a skewed
real-world metric (e.g., spend or revenue).

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Gamma parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
  Mean = shape * scale.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the log-mean outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre with post-outcome in control group.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- \*\***kwargs** – Additional arguments passed to `generate_rct` (e.g., pre_name, g_y, use_prognostic).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset with gamma outcome.

#### `classic_rct_gamma_26`

```python
classic_rct_gamma_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = True,
    **kwargs: bool
)
```

A pre-configured classic RCT dataset with a gamma outcome.
n=10000, split=0.5, mean uplift ~10%.
Includes deterministic `user_id` and ancillary columns.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre').
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **n** (<code>[int](#int)</code>) – Number of samples.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **outcome_params** (<code>[dict](#dict)</code>) – Gamma outcome parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- \*\***kwargs** – Additional arguments passed to `classic_rct_gamma`.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

#### `generate_classic_rct`

```python
generate_classic_rct(
    n: int = 10000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    return_causal_data: bool = False,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    **kwargs: bool
) -> Union[pd.DataFrame, CausalData]
```

Generate a classic RCT dataset with three binary confounders:
platform_ios, country_usa, and source_paid.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal (passed to generate_rct).
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre (passed to generate_rct).
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- \*\***kwargs** – Additional arguments passed to `generate_rct`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

#### `generate_classic_rct_26`

```python
generate_classic_rct_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = False,
    deterministic_ids: bool = True,
    **kwargs: bool
)
```

A pre-configured classic RCT dataset with 3 binary confounders.
n=10000, split=0.5, outcome is conversion (binary). Baseline control p=0.10
and treatment p=0.11 are set on the log-odds scale (X=0), so marginal rates
and ATE can differ once covariate effects are included. Includes a
deterministic `user_id` column.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **n** (<code>[int](#int)</code>) – Number of samples.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **outcome_params** (<code>[dict](#dict)</code>) – Binary outcome parameters, e.g. {"p": {"A": 0.10, "B": 0.11}}.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- \*\***kwargs** – Additional arguments passed to `generate_classic_rct`.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

#### `generate_cuped_binary`

```python
generate_cuped_binary(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Binary CUPED-oriented DGP with richer confounders and structured HTE.

Designed for CUPED benchmarking with randomized treatment and a calibrated
pre-period covariate while preserving exact oracle cate under include_oracle.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in the control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration.
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like m, g0, g1, cate.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_logit** (<code>[float](#float)</code>) – Baseline log-odds uplift scale for heterogeneous treatment effects.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

#### `generate_cuped_tweedie_26`

```python
generate_cuped_tweedie_26(
    n: int = 20000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_name_2: Optional[str] = None,
    pre_target_corr: float = 0.82,
    pre_target_corr_2: Optional[float] = None,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail.
Includes two pre-period covariates by default: 'y_pre' and 'y_pre_2'.
Wrapper for make_tweedie().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add pre-period covariates.
- **pre_name** (<code>[str](#str)</code>) – Name of the first pre-period covariate column.
- **pre_name_2** (<code>[str](#str)</code>) – Name of the second pre-period covariate column.
  Defaults to `f"{pre_name}_2"`.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between the first pre covariate and post-outcome y in control group.
- **pre_target_corr_2** (<code>[float](#float)</code>) – Target correlation for the second pre covariate. Defaults to a
  moderate value based on `pre_target_corr` to reduce collinearity.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

#### `generate_iv_data`

```python
generate_iv_data(n: int = 1000) -> pd.DataFrame
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

#### `generate_rct`

```python
generate_rct(
    n: int = 20000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_type: str = "binary",
    outcome_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_corr: float = 0.7,
    prognostic_scale: float = 1.0,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    use_prognostic: Optional[bool] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
) -> Union[pd.DataFrame, CausalData]
```

Generate an RCT dataset with randomized treatment assignment.

Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
Specifically designed for benchmarking variance reduction techniques like CUPED.

**Notes on effect scale**

How `outcome_params` maps into the structural effect:

- outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
- outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
- outcome_type="poisson" or "gamma": treatment shifts the log-mean by log(lam_B / lam_A).

Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
avoiding outcome leakage and post-treatment adjustment issues.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_type** (<code>('binary', 'normal', 'poisson', 'gamma')</code>) – Distribution family of the outcome.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.12}} for binary, or
  {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}} for poisson/gamma.
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add descriptive columns like 'age', 'platform', etc.
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation between `y_pre` and the outcome Y in the control group.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of the prognostic signal derived from confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic RCT dataset.

#### `generate_scm_data`

```python
generate_scm_data(
    n_donors: int = 5,
    n_pre_periods: int = 20,
    n_post_periods: int = 10,
    treatment_effect: float = 2.0,
    treatment_effect_slope: float = 0.0,
    donor_noise_std: float = 0.2,
    treated_noise_std: float = 0.1,
    common_factor_std: float = 0.15,
    time_start: int = 1,
    treated_unit: Hashable = "treated",
    donor_prefix: str = "donor_",
    random_state: Optional[int] = 42,
    return_panel_data: bool = True,
    dirichlet_alpha: float = 1.0,
    rho_common: float = 0.0,
    rho_donor: float = 0.0,
    n_latent_factors: int = 0,
    latent_factor_std: float = 0.2,
    latent_loading_std: float = 0.35,
    rho_latent: float = 0.0,
    prefit_mismatch_std: float = 0.0,
    rho_prefit_mismatch: float = 0.0,
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive",
) -> PanelOutput
```

Medium-level wrapper for Gaussian SCM panel generation.

<details class="note" open markdown="1">
<summary>Notes</summary>

The Gaussian treated counterfactual path is exact SCM only when both
`treated_noise_std=0.0` and `prefit_mismatch_std=0.0`. With either term
non-zero, the generator intentionally produces an approximate-SCM fit path.

</details>

#### `make_cuped_binary_26`

```python
make_cuped_binary_26(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Binary CUPED benchmark with richer confounders and structured HTE.
Includes a calibrated pre-period covariate 'y_pre' by default.
Wrapper for generate_cuped_binary().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in the control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like 'cate', 'g0', and 'g1'.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_logit** (<code>[float](#float)</code>) – Baseline log-odds uplift scale for heterogeneous treatment effects.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

#### `make_cuped_tweedie`

```python
make_cuped_tweedie(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.6,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.2,
) -> Union[pd.DataFrame, CausalData]
```

Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail. Suitable for CUPED benchmarking.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

#### `make_gold_linear`

```python
make_gold_linear(n: int = 10000, seed: int = 42) -> CausalData
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

#### `multicausaldata`

**Modules:**

- [**base**](#causalis.dgp.multicausaldata.base) –
- [**functional**](#causalis.dgp.multicausaldata.functional) –

**Classes:**

- [**MultiCausalData**](#causalis.dgp.multicausaldata.MultiCausalData) – Data contract for cross-sectional causal data with multi-class one-hot treatments.
- [**MultiCausalDatasetGenerator**](#causalis.dgp.multicausaldata.MultiCausalDatasetGenerator) – Generate synthetic causal datasets with multi-class (one-hot) treatments.

**Functions:**

- [**generate_multitreatment**](#causalis.dgp.multicausaldata.generate_multitreatment) – Generate a multi-treatment dataset using MultiCausalDatasetGenerator.
- [**generate_multitreatment_binary_26**](#causalis.dgp.multicausaldata.generate_multitreatment_binary_26) –
- [**generate_multitreatment_gamma_26**](#causalis.dgp.multicausaldata.generate_multitreatment_gamma_26) –
- [**generate_multitreatment_irm_26**](#causalis.dgp.multicausaldata.generate_multitreatment_irm_26) –

##### `MultiCausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Data contract for cross-sectional causal data with multi-class one-hot treatments.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the causal data.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome column.
- **treatment_names** (<code>[List](#typing.List)\[[str](#str)\]</code>) – The names of the treatment columns.
- **confounders** (<code>[List](#typing.List)\[[str](#str)\]</code>) – The names of the confounder columns, by default [].
- **user_id** (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) – The name of the user ID column, by default None.
- **control_treatment** (<code>[str](#str)</code>) – Name of the control/baseline treatment column.

<details class="note" open markdown="1">
<summary>Notes</summary>

This class enforces several constraints on the data, including:

- Maximum number of treatment_names (default 15).
- No duplicate column names in the input DataFrame.
- Disjoint roles for columns (outcome, treatment_names, confounders, user_id).
- Non-empty normalized names for outcome and user_id (if provided).
- Existence of all specified columns in the DataFrame.
- Numeric or boolean types for outcome and confounders.
- Finite values for outcome, confounders, and treatment_names.
- Non-constant values for outcome, treatment_names, and confounders.
- No NaN values in used columns.
- Binary (0/1) encoding for treatment columns.
- One-hot treatment assignment (exactly one active treatment per row).
- A stable control treatment in position 0.
- No identical values between different columns.
- Unique values for user_id (if specified).

</details>

**Functions:**

- [**from_df**](#causalis.dgp.multicausaldata.MultiCausalData.from_df) – Create a MultiCausalData instance from a pandas DataFrame.
- [**get_df**](#causalis.dgp.multicausaldata.MultiCausalData.get_df) – Get a subset of the underlying DataFrame.

###### `FLOAT_TOL`

```python
FLOAT_TOL: float = 1e-12
```

###### `MAX_TREATMENTS`

```python
MAX_TREATMENTS: int = 15
```

###### `X`

```python
X: pd.DataFrame
```

Return the confounder columns as a pandas DataFrame.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The confounder columns.

###### `confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

###### `control_treatment`

```python
control_treatment: str
```

###### `df`

```python
df: pd.DataFrame
```

###### `from_df`

```python
from_df(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment_names: Union[str, List[str]],
    confounders: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    control_treatment: str,
    **kwargs: Any
) -> "MultiCausalData"
```

Create a MultiCausalData instance from a pandas DataFrame.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The input DataFrame.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome column.
- **treatment_names** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – The name(s) of the treatment column(s).
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – The name(s) of the confounder column(s), by default None.
- **user_id** (<code>[str](#str)</code>) – The name of the user ID column, by default None.
- **control_treatment** (<code>[str](#str)</code>) – Name of the control treatment column.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional keyword arguments passed to the constructor.

**Returns:**

- <code>[MultiCausalData](#causalis.data_contracts.multicausaldata.MultiCausalData)</code> – An instance of MultiCausalData.

###### `get_df`

```python
get_df(
    columns: Optional[List[str]] = None,
    include_outcome: bool = True,
    include_confounders: bool = True,
    include_treatments: bool = True,
    include_user_id: bool = False,
) -> pd.DataFrame
```

Get a subset of the underlying DataFrame.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific columns to include, by default None.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column, by default True.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns, by default True.
- **include_treatments** (<code>[bool](#bool)</code>) – Whether to include treatment columns, by default True.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user ID column, by default False.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the requested DataFrame subset.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any of the requested columns do not exist.

###### `model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
```

###### `outcome`

```python
outcome: str
```

###### `treatment`

```python
treatment: pd.Series
```

Return the single treatment column as a pandas Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

**Raises:**

- <code>[AttributeError](#AttributeError)</code> – If there is more than one treatment column.

###### `treatment_names`

```python
treatment_names: List[str]
```

###### `treatments`

```python
treatments: pd.DataFrame
```

Return the treatment columns as a pandas DataFrame.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The treatment columns.

###### `user_id`

```python
user_id: Optional[str] = None
```

##### `MultiCausalDatasetGenerator`

```python
MultiCausalDatasetGenerator(
    n_treatments: int = 3,
    d_names: Optional[List[str]] = None,
    theta: Optional[Union[float, List[float], np.ndarray]] = 1.0,
    tau: Optional[
        Union[
            Callable[[np.ndarray], np.ndarray],
            List[Optional[Callable[[np.ndarray], np.ndarray]]],
        ]
    ] = None,
    beta_y: Optional[np.ndarray] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha_y: float = 0.0,
    sigma_y: float = 1.0,
    outcome_type: str = "continuous",
    gamma_shape: float = 2.0,
    u_strength_y: float = 0.0,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    use_copula: bool = False,
    copula_corr: Optional[np.ndarray] = None,
    beta_d: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    g_d: Optional[
        Union[
            Callable[[np.ndarray], np.ndarray],
            List[Optional[Callable[[np.ndarray], np.ndarray]]],
        ]
    ] = None,
    alpha_d: Optional[Union[float, List[float], np.ndarray]] = None,
    u_strength_d: Union[float, List[float], np.ndarray] = 0.0,
    propensity_sharpness: float = 1.0,
    target_d_rate: Optional[Union[List[float], np.ndarray]] = None,
    include_oracle: bool = True,
    seed: Optional[int] = None,
) -> None
```

Generate synthetic causal datasets with multi-class (one-hot) treatments.

Treatment assignment is modeled via a multinomial logistic (softmax) model:
P(D=k | X, U) = softmax_k(alpha_d[k] + f_k(X) + u_strength_d[k] * U)

Outcome depends on confounders and the assigned treatment class:
outcome_type = "continuous":
Y = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X)) + eps
outcome_type = "binary":
logit P(Y=1|X,D,U) = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))
outcome_type = "poisson":
log E[Y|X,D,U] = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))
outcome_type = "gamma":
log E[Y|X,D,U] = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))

**Parameters:**

- **n_treatments** (<code>[int](#int)</code>) – Number of treatment classes (including control). Column 0 is treated as control.
  Generated treatment columns are a full one-hot encoding that sums to 1.
- **d_names** (<code>list of str</code>) – Names of treatment columns. If None, uses ["d_0", "d_1", ...].
- **theta** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Constant treatment effects on the link scale for each class.
  If scalar, applied to all non-control classes (control effect = 0).
  If length K-1, prepends 0 for control. If length K, uses as provided.
- **tau** (<code>callable or list of callables</code>) – Heterogeneous effects for each class. If callable, applied to non-control classes.
  Effects are additive with theta on the link scale:
  tau_link_k(X) = theta_k + tau_k(X).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for baseline outcome f_y(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear baseline outcome function g_y(X).
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept on link scale.
- **sigma_y** (<code>[float](#float)</code>) – Std dev for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma')</code>) – Outcome family.
- **gamma_shape** (<code>[float](#float)</code>) – Shape parameter for gamma outcomes.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of unobserved confounder in outcome.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders (same format as CausalDatasetGenerator).
- **k** (<code>[int](#int)</code>) – Number of confounders if confounder_specs is None.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray.
- **use_copula** (<code>[bool](#bool)</code>) – If True and confounder_specs provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **beta_d** (<code>[array](#array) - [like](#like) or [list](#list)</code>) – Linear coefficients for treatment assignment. If array of shape (k,),
  applies to all non-control classes. If shape (K,k), uses per class.
- **g_d** (<code>callable or list of callables</code>) – Nonlinear treatment score per class. If callable, applies to non-control classes.
- **alpha_d** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Intercepts for treatment scores. If scalar, applies to non-control classes.
- **u_strength_d** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Unobserved confounder strength in treatment assignment.
  If scalar, interpreted as [0, c, c, ...] so latent U perturbs non-control
  classes relative to control (and does not cancel in softmax).
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales treatment scores to adjust overlap.
- **target_d_rate** (<code>[array](#array) - [like](#like)</code>) – Target marginal class probabilities (length K). Calibrates alpha_d
  using iterative scaling (approximate when u_strength_d != 0).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns for propensities and potential outcomes.
- **seed** (<code>[int](#int)</code>) – Random seed.

**Functions:**

- [**generate**](#causalis.dgp.multicausaldata.MultiCausalDatasetGenerator.generate) –
- [**to_multicausal_data**](#causalis.dgp.multicausaldata.MultiCausalDatasetGenerator.to_multicausal_data) –

###### `alpha_d`

```python
alpha_d: Optional[Union[float, List[float], np.ndarray]] = None
```

###### `alpha_y`

```python
alpha_y: float = 0.0
```

###### `beta_d`

```python
beta_d: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None
```

###### `beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

###### `confounder_names_`

```python
confounder_names_: List[str] = field(init=False, default_factory=list)
```

###### `confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

###### `copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

###### `d_names`

```python
d_names: Optional[List[str]] = None
```

###### `g_d`

```python
g_d: Optional[
    Union[
        Callable[[np.ndarray], np.ndarray],
        List[Optional[Callable[[np.ndarray], np.ndarray]]],
    ]
] = None

```

###### `g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `gamma_shape`

```python
gamma_shape: float = 2.0
```

###### `generate`

```python
generate(n: int, U: Optional[np.ndarray] = None) -> pd.DataFrame
```

###### `include_oracle`

```python
include_oracle: bool = True
```

###### `k`

```python
k: int = 5
```

###### `n_treatments`

```python
n_treatments: int = 3
```

###### `outcome_type`

```python
outcome_type: str = 'continuous'
```

###### `propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

###### `rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

###### `seed`

```python
seed: Optional[int] = None
```

###### `sigma_y`

```python
sigma_y: float = 1.0
```

###### `target_d_rate`

```python
target_d_rate: Optional[Union[List[float], np.ndarray]] = None
```

###### `tau`

```python
tau: Optional[
    Union[
        Callable[[np.ndarray], np.ndarray],
        List[Optional[Callable[[np.ndarray], np.ndarray]]],
    ]
] = None

```

###### `theta`

```python
theta: Optional[Union[float, List[float], np.ndarray]] = 1.0
```

###### `to_multicausal_data`

```python
to_multicausal_data(
    n: int, confounders: Optional[Union[str, List[str]]] = None
) -> MultiCausalData
```

###### `u_strength_d`

```python
u_strength_d: Union[float, List[float], np.ndarray] = 0.0
```

###### `u_strength_y`

```python
u_strength_y: float = 0.0
```

###### `use_copula`

```python
use_copula: bool = False
```

###### `x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `base`

**Classes:**

- [**MultiCausalDatasetGenerator**](#causalis.dgp.multicausaldata.base.MultiCausalDatasetGenerator) – Generate synthetic causal datasets with multi-class (one-hot) treatments.

###### `MultiCausalDatasetGenerator`

```python
MultiCausalDatasetGenerator(
    n_treatments: int = 3,
    d_names: Optional[List[str]] = None,
    theta: Optional[Union[float, List[float], np.ndarray]] = 1.0,
    tau: Optional[
        Union[
            Callable[[np.ndarray], np.ndarray],
            List[Optional[Callable[[np.ndarray], np.ndarray]]],
        ]
    ] = None,
    beta_y: Optional[np.ndarray] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha_y: float = 0.0,
    sigma_y: float = 1.0,
    outcome_type: str = "continuous",
    gamma_shape: float = 2.0,
    u_strength_y: float = 0.0,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    use_copula: bool = False,
    copula_corr: Optional[np.ndarray] = None,
    beta_d: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    g_d: Optional[
        Union[
            Callable[[np.ndarray], np.ndarray],
            List[Optional[Callable[[np.ndarray], np.ndarray]]],
        ]
    ] = None,
    alpha_d: Optional[Union[float, List[float], np.ndarray]] = None,
    u_strength_d: Union[float, List[float], np.ndarray] = 0.0,
    propensity_sharpness: float = 1.0,
    target_d_rate: Optional[Union[List[float], np.ndarray]] = None,
    include_oracle: bool = True,
    seed: Optional[int] = None,
) -> None
```

Generate synthetic causal datasets with multi-class (one-hot) treatments.

Treatment assignment is modeled via a multinomial logistic (softmax) model:
P(D=k | X, U) = softmax_k(alpha_d[k] + f_k(X) + u_strength_d[k] * U)

Outcome depends on confounders and the assigned treatment class:
outcome_type = "continuous":
Y = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X)) + eps
outcome_type = "binary":
logit P(Y=1|X,D,U) = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))
outcome_type = "poisson":
log E[Y|X,D,U] = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))
outcome_type = "gamma":
log E[Y|X,D,U] = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))

**Parameters:**

- **n_treatments** (<code>[int](#int)</code>) – Number of treatment classes (including control). Column 0 is treated as control.
  Generated treatment columns are a full one-hot encoding that sums to 1.
- **d_names** (<code>list of str</code>) – Names of treatment columns. If None, uses ["d_0", "d_1", ...].
- **theta** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Constant treatment effects on the link scale for each class.
  If scalar, applied to all non-control classes (control effect = 0).
  If length K-1, prepends 0 for control. If length K, uses as provided.
- **tau** (<code>callable or list of callables</code>) – Heterogeneous effects for each class. If callable, applied to non-control classes.
  Effects are additive with theta on the link scale:
  tau_link_k(X) = theta_k + tau_k(X).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for baseline outcome f_y(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear baseline outcome function g_y(X).
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept on link scale.
- **sigma_y** (<code>[float](#float)</code>) – Std dev for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma')</code>) – Outcome family.
- **gamma_shape** (<code>[float](#float)</code>) – Shape parameter for gamma outcomes.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of unobserved confounder in outcome.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders (same format as CausalDatasetGenerator).
- **k** (<code>[int](#int)</code>) – Number of confounders if confounder_specs is None.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray.
- **use_copula** (<code>[bool](#bool)</code>) – If True and confounder_specs provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **beta_d** (<code>[array](#array) - [like](#like) or [list](#list)</code>) – Linear coefficients for treatment assignment. If array of shape (k,),
  applies to all non-control classes. If shape (K,k), uses per class.
- **g_d** (<code>callable or list of callables</code>) – Nonlinear treatment score per class. If callable, applies to non-control classes.
- **alpha_d** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Intercepts for treatment scores. If scalar, applies to non-control classes.
- **u_strength_d** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Unobserved confounder strength in treatment assignment.
  If scalar, interpreted as [0, c, c, ...] so latent U perturbs non-control
  classes relative to control (and does not cancel in softmax).
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales treatment scores to adjust overlap.
- **target_d_rate** (<code>[array](#array) - [like](#like)</code>) – Target marginal class probabilities (length K). Calibrates alpha_d
  using iterative scaling (approximate when u_strength_d != 0).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns for propensities and potential outcomes.
- **seed** (<code>[int](#int)</code>) – Random seed.

**Functions:**

- [**generate**](#causalis.dgp.multicausaldata.base.MultiCausalDatasetGenerator.generate) –
- [**to_multicausal_data**](#causalis.dgp.multicausaldata.base.MultiCausalDatasetGenerator.to_multicausal_data) –

####### `alpha_d`

```python
alpha_d: Optional[Union[float, List[float], np.ndarray]] = None
```

####### `alpha_y`

```python
alpha_y: float = 0.0
```

####### `beta_d`

```python
beta_d: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None
```

####### `beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

####### `confounder_names_`

```python
confounder_names_: List[str] = field(init=False, default_factory=list)
```

####### `confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

####### `copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

####### `d_names`

```python
d_names: Optional[List[str]] = None
```

####### `g_d`

```python
g_d: Optional[
    Union[
        Callable[[np.ndarray], np.ndarray],
        List[Optional[Callable[[np.ndarray], np.ndarray]]],
    ]
] = None

```

####### `g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `gamma_shape`

```python
gamma_shape: float = 2.0
```

####### `generate`

```python
generate(n: int, U: Optional[np.ndarray] = None) -> pd.DataFrame
```

####### `include_oracle`

```python
include_oracle: bool = True
```

####### `k`

```python
k: int = 5
```

####### `n_treatments`

```python
n_treatments: int = 3
```

####### `outcome_type`

```python
outcome_type: str = 'continuous'
```

####### `propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

####### `rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

####### `seed`

```python
seed: Optional[int] = None
```

####### `sigma_y`

```python
sigma_y: float = 1.0
```

####### `target_d_rate`

```python
target_d_rate: Optional[Union[List[float], np.ndarray]] = None
```

####### `tau`

```python
tau: Optional[
    Union[
        Callable[[np.ndarray], np.ndarray],
        List[Optional[Callable[[np.ndarray], np.ndarray]]],
    ]
] = None

```

####### `theta`

```python
theta: Optional[Union[float, List[float], np.ndarray]] = 1.0
```

####### `to_multicausal_data`

```python
to_multicausal_data(
    n: int, confounders: Optional[Union[str, List[str]]] = None
) -> MultiCausalData
```

####### `u_strength_d`

```python
u_strength_d: Union[float, List[float], np.ndarray] = 0.0
```

####### `u_strength_y`

```python
u_strength_y: float = 0.0
```

####### `use_copula`

```python
use_copula: bool = False
```

####### `x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `functional`

**Functions:**

- [**generate_multitreatment**](#causalis.dgp.multicausaldata.functional.generate_multitreatment) – Generate a multi-treatment dataset using MultiCausalDatasetGenerator.

###### `generate_multitreatment`

```python
generate_multitreatment(
    n: int = 10000,
    n_treatments: int = 3,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    alpha_y: float = 0.0,
    gamma_shape: float = 2.0,
    tau: Optional[Any] = None,
    target_d_rate: Optional[Union[List[float], Any]] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[Any] = None,
    beta_d: Optional[Any] = None,
    theta: Optional[Any] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Any] = None,
    use_copula: bool = False,
    copula_corr: Optional[Any] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    d_names: Optional[List[str]] = None,
) -> Union[pd.DataFrame, MultiCausalData]
```

Generate a multi-treatment dataset using MultiCausalDatasetGenerator.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **n_treatments** (<code>[int](#int)</code>) – Number of treatment classes (including control).
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma')</code>) – Outcome family.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept on the structural link scale.
- **gamma_shape** (<code>[float](#float)</code>) – Shape parameter when `outcome_type="gamma"`.
- **tau** (<code>callable or list of callables</code>) – Heterogeneous treatment effects per class. When provided with `theta`,
  the structural effect is additive on link scale (`theta + tau(X)`).
- **target_d_rate** (<code>[array](#array) - [like](#like)</code>) – Target marginal class probabilities (length K).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for treatment model.
- **theta** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Constant treatment effects per class (adds to `tau` when both are provided).
- **random_state** (<code>[int](#int)</code>) – Random seed.
- **k** (<code>[int](#int)</code>) – Number of confounders if confounder_specs is None.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **use_copula** (<code>[bool](#bool)</code>) – Whether to use Gaussian copula sampling for confounders.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix used when `use_copula=True`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a MultiCausalData object.
- **d_names** (<code>list of str</code>) – Names of treatment columns.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [MultiCausalData](#causalis.data_contracts.multicausaldata.MultiCausalData)</code> –

##### `generate_multitreatment`

```python
generate_multitreatment(
    n: int = 10000,
    n_treatments: int = 3,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    alpha_y: float = 0.0,
    gamma_shape: float = 2.0,
    tau: Optional[Any] = None,
    target_d_rate: Optional[Union[List[float], Any]] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[Any] = None,
    beta_d: Optional[Any] = None,
    theta: Optional[Any] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Any] = None,
    use_copula: bool = False,
    copula_corr: Optional[Any] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    d_names: Optional[List[str]] = None,
) -> Union[pd.DataFrame, MultiCausalData]
```

Generate a multi-treatment dataset using MultiCausalDatasetGenerator.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **n_treatments** (<code>[int](#int)</code>) – Number of treatment classes (including control).
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma')</code>) – Outcome family.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept on the structural link scale.
- **gamma_shape** (<code>[float](#float)</code>) – Shape parameter when `outcome_type="gamma"`.
- **tau** (<code>callable or list of callables</code>) – Heterogeneous treatment effects per class. When provided with `theta`,
  the structural effect is additive on link scale (`theta + tau(X)`).
- **target_d_rate** (<code>[array](#array) - [like](#like)</code>) – Target marginal class probabilities (length K).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for treatment model.
- **theta** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – Constant treatment effects per class (adds to `tau` when both are provided).
- **random_state** (<code>[int](#int)</code>) – Random seed.
- **k** (<code>[int](#int)</code>) – Number of confounders if confounder_specs is None.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **use_copula** (<code>[bool](#bool)</code>) – Whether to use Gaussian copula sampling for confounders.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix used when `use_copula=True`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a MultiCausalData object.
- **d_names** (<code>list of str</code>) – Names of treatment columns.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [MultiCausalData](#causalis.data_contracts.multicausaldata.MultiCausalData)</code> –

##### `generate_multitreatment_binary_26`

```python
generate_multitreatment_binary_26(*args, **kwargs)
```

##### `generate_multitreatment_gamma_26`

```python
generate_multitreatment_gamma_26(*args, **kwargs)
```

##### `generate_multitreatment_irm_26`

```python
generate_multitreatment_irm_26(*args, **kwargs)
```

#### `obs_linear_26_dataset`

```python
obs_linear_26_dataset(
    n: int = 10000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
)
```

A pre-configured observational linear dataset with 5 standard confounders.
Based on the scenario in docs/cases/dml_ate.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

#### `obs_linear_effect`

```python
obs_linear_effect(
    n: int = 10000,
    theta: float = 1.0,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    target_d_rate: Optional[float] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    include_oracle: bool = True,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
) -> pd.DataFrame
```

Generate an observational dataset with linear effects of confounders and a constant treatment effect.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **theta** (<code>[float](#float)</code>) – Constant treatment effect.
- **outcome_type** (<code>('continuous', 'binary', 'poisson', 'gamma')</code>) – Family of the outcome distribution.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the treatment model.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **add_ancillary** (<code>[bool](#bool)</code>) – If True, adds standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – If True, generates deterministic user IDs.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic observational dataset.

#### `panel_data_scm`

**Modules:**

- [**base**](#causalis.dgp.panel_data_scm.base) –
- [**functional**](#causalis.dgp.panel_data_scm.functional) –

**Classes:**

- [**PanelSCMGenerator**](#causalis.dgp.panel_data_scm.PanelSCMGenerator) – Low-level panel SCM generator supporting Gaussian, Gamma, and Poisson outcomes.
- [**PanelSCMGeneratorConfig**](#causalis.dgp.panel_data_scm.PanelSCMGeneratorConfig) –

**Functions:**

- [**generate_scm_data**](#causalis.dgp.panel_data_scm.generate_scm_data) – Medium-level wrapper for Gaussian SCM panel generation.
- [**generate_scm_gamma_data**](#causalis.dgp.panel_data_scm.generate_scm_gamma_data) – Medium-level wrapper for realistic Gamma SCM panel generation.
- [**generate_scm_poisson_data**](#causalis.dgp.panel_data_scm.generate_scm_poisson_data) – Medium-level wrapper for realistic Poisson SCM panel generation.

##### `PanelSCMGenerator`

```python
PanelSCMGenerator(config: PanelSCMGeneratorConfig)
```

Low-level panel SCM generator supporting Gaussian, Gamma, and Poisson outcomes.

**Functions:**

- [**generate**](#causalis.dgp.panel_data_scm.PanelSCMGenerator.generate) –

###### `config`

```python
config = config
```

###### `generate`

```python
generate(
    *, return_panel_data: Optional[bool] = None
) -> Union[pd.DataFrame, PanelDataSCM]
```

##### `PanelSCMGeneratorConfig`

```python
PanelSCMGeneratorConfig(
    n_donors: int = 5,
    n_pre_periods: int = 20,
    n_post_periods: int = 10,
    time_start: int = 1,
    time_freq: str = "M",
    calendar_start: str = "2000-01",
    treated_unit: Hashable = "treated",
    donor_prefix: str = "donor_",
    random_state: Optional[int] = 42,
    return_panel_data: bool = True,
    dirichlet_alpha: float = 1.0,
    rho_common: float = 0.0,
    rho_donor: float = 0.0,
    n_latent_factors: int = 0,
    latent_loading_std: float = 0.35,
    rho_latent: float = 0.0,
    rho_prefit_mismatch: float = 0.0,
    outcome_distribution: Literal["gaussian", "gamma", "poisson"] = "gaussian",
    treatment_effect: float = 2.0,
    treatment_effect_slope: float = 0.0,
    donor_noise_std: float = 0.2,
    treated_noise_std: float = 0.1,
    common_factor_std: float = 0.15,
    latent_factor_std: float = 0.2,
    prefit_mismatch_std: float = 0.0,
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive",
    treatment_effect_rate: float = 0.12,
    gamma_shape: float = 6.0,
    donor_noise_std_log: float = 0.15,
    common_factor_std_log: float = 0.1,
    latent_factor_std_log: float = 0.1,
    prefit_mismatch_std_log: float = 0.08,
) -> None
```

###### `calendar_start`

```python
calendar_start: str = '2000-01'
```

###### `common_factor_std`

```python
common_factor_std: float = 0.15
```

###### `common_factor_std_log`

```python
common_factor_std_log: float = 0.1
```

###### `dirichlet_alpha`

```python
dirichlet_alpha: float = 1.0
```

###### `donor_noise_std`

```python
donor_noise_std: float = 0.2
```

###### `donor_noise_std_log`

```python
donor_noise_std_log: float = 0.15
```

###### `donor_prefix`

```python
donor_prefix: str = 'donor_'
```

###### `gamma_shape`

```python
gamma_shape: float = 6.0
```

###### `latent_factor_std`

```python
latent_factor_std: float = 0.2
```

###### `latent_factor_std_log`

```python
latent_factor_std_log: float = 0.1
```

###### `latent_loading_std`

```python
latent_loading_std: float = 0.35
```

###### `n_donors`

```python
n_donors: int = 5
```

###### `n_latent_factors`

```python
n_latent_factors: int = 0
```

###### `n_post_periods`

```python
n_post_periods: int = 10
```

###### `n_pre_periods`

```python
n_pre_periods: int = 20
```

###### `outcome_distribution`

```python
outcome_distribution: Literal['gaussian', 'gamma', 'poisson'] = 'gaussian'
```

###### `prefit_mismatch_std`

```python
prefit_mismatch_std: float = 0.0
```

###### `prefit_mismatch_std_log`

```python
prefit_mismatch_std_log: float = 0.08
```

###### `random_state`

```python
random_state: Optional[int] = 42
```

###### `return_panel_data`

```python
return_panel_data: bool = True
```

###### `rho_common`

```python
rho_common: float = 0.0
```

###### `rho_donor`

```python
rho_donor: float = 0.0
```

###### `rho_latent`

```python
rho_latent: float = 0.0
```

###### `rho_prefit_mismatch`

```python
rho_prefit_mismatch: float = 0.0
```

###### `time_freq`

```python
time_freq: str = 'M'
```

###### `time_start`

```python
time_start: int = 1
```

###### `treated_noise_std`

```python
treated_noise_std: float = 0.1
```

###### `treated_unit`

```python
treated_unit: Hashable = 'treated'
```

###### `treatment_effect`

```python
treatment_effect: float = 2.0
```

###### `treatment_effect_mode`

```python
treatment_effect_mode: Literal['additive', 'multiplicative'] = 'additive'
```

###### `treatment_effect_rate`

```python
treatment_effect_rate: float = 0.12
```

###### `treatment_effect_slope`

```python
treatment_effect_slope: float = 0.0
```

##### `base`

**Classes:**

- [**PanelSCMGenerator**](#causalis.dgp.panel_data_scm.base.PanelSCMGenerator) – Low-level panel SCM generator supporting Gaussian, Gamma, and Poisson outcomes.
- [**PanelSCMGeneratorConfig**](#causalis.dgp.panel_data_scm.base.PanelSCMGeneratorConfig) –

###### `PanelSCMGenerator`

```python
PanelSCMGenerator(config: PanelSCMGeneratorConfig)
```

Low-level panel SCM generator supporting Gaussian, Gamma, and Poisson outcomes.

**Functions:**

- [**generate**](#causalis.dgp.panel_data_scm.base.PanelSCMGenerator.generate) –

####### `config`

```python
config = config
```

####### `generate`

```python
generate(
    *, return_panel_data: Optional[bool] = None
) -> Union[pd.DataFrame, PanelDataSCM]
```

###### `PanelSCMGeneratorConfig`

```python
PanelSCMGeneratorConfig(
    n_donors: int = 5,
    n_pre_periods: int = 20,
    n_post_periods: int = 10,
    time_start: int = 1,
    time_freq: str = "M",
    calendar_start: str = "2000-01",
    treated_unit: Hashable = "treated",
    donor_prefix: str = "donor_",
    random_state: Optional[int] = 42,
    return_panel_data: bool = True,
    dirichlet_alpha: float = 1.0,
    rho_common: float = 0.0,
    rho_donor: float = 0.0,
    n_latent_factors: int = 0,
    latent_loading_std: float = 0.35,
    rho_latent: float = 0.0,
    rho_prefit_mismatch: float = 0.0,
    outcome_distribution: Literal["gaussian", "gamma", "poisson"] = "gaussian",
    treatment_effect: float = 2.0,
    treatment_effect_slope: float = 0.0,
    donor_noise_std: float = 0.2,
    treated_noise_std: float = 0.1,
    common_factor_std: float = 0.15,
    latent_factor_std: float = 0.2,
    prefit_mismatch_std: float = 0.0,
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive",
    treatment_effect_rate: float = 0.12,
    gamma_shape: float = 6.0,
    donor_noise_std_log: float = 0.15,
    common_factor_std_log: float = 0.1,
    latent_factor_std_log: float = 0.1,
    prefit_mismatch_std_log: float = 0.08,
) -> None
```

####### `calendar_start`

```python
calendar_start: str = '2000-01'
```

####### `common_factor_std`

```python
common_factor_std: float = 0.15
```

####### `common_factor_std_log`

```python
common_factor_std_log: float = 0.1
```

####### `dirichlet_alpha`

```python
dirichlet_alpha: float = 1.0
```

####### `donor_noise_std`

```python
donor_noise_std: float = 0.2
```

####### `donor_noise_std_log`

```python
donor_noise_std_log: float = 0.15
```

####### `donor_prefix`

```python
donor_prefix: str = 'donor_'
```

####### `gamma_shape`

```python
gamma_shape: float = 6.0
```

####### `latent_factor_std`

```python
latent_factor_std: float = 0.2
```

####### `latent_factor_std_log`

```python
latent_factor_std_log: float = 0.1
```

####### `latent_loading_std`

```python
latent_loading_std: float = 0.35
```

####### `n_donors`

```python
n_donors: int = 5
```

####### `n_latent_factors`

```python
n_latent_factors: int = 0
```

####### `n_post_periods`

```python
n_post_periods: int = 10
```

####### `n_pre_periods`

```python
n_pre_periods: int = 20
```

####### `outcome_distribution`

```python
outcome_distribution: Literal['gaussian', 'gamma', 'poisson'] = 'gaussian'
```

####### `prefit_mismatch_std`

```python
prefit_mismatch_std: float = 0.0
```

####### `prefit_mismatch_std_log`

```python
prefit_mismatch_std_log: float = 0.08
```

####### `random_state`

```python
random_state: Optional[int] = 42
```

####### `return_panel_data`

```python
return_panel_data: bool = True
```

####### `rho_common`

```python
rho_common: float = 0.0
```

####### `rho_donor`

```python
rho_donor: float = 0.0
```

####### `rho_latent`

```python
rho_latent: float = 0.0
```

####### `rho_prefit_mismatch`

```python
rho_prefit_mismatch: float = 0.0
```

####### `time_freq`

```python
time_freq: str = 'M'
```

####### `time_start`

```python
time_start: int = 1
```

####### `treated_noise_std`

```python
treated_noise_std: float = 0.1
```

####### `treated_unit`

```python
treated_unit: Hashable = 'treated'
```

####### `treatment_effect`

```python
treatment_effect: float = 2.0
```

####### `treatment_effect_mode`

```python
treatment_effect_mode: Literal['additive', 'multiplicative'] = 'additive'
```

####### `treatment_effect_rate`

```python
treatment_effect_rate: float = 0.12
```

####### `treatment_effect_slope`

```python
treatment_effect_slope: float = 0.0
```

##### `functional`

**Functions:**

- [**generate_scm_data**](#causalis.dgp.panel_data_scm.functional.generate_scm_data) – Medium-level wrapper for Gaussian SCM panel generation.
- [**generate_scm_gamma_26_data**](#causalis.dgp.panel_data_scm.functional.generate_scm_gamma_26_data) –
- [**generate_scm_gamma_data**](#causalis.dgp.panel_data_scm.functional.generate_scm_gamma_data) – Medium-level wrapper for realistic Gamma SCM panel generation.
- [**generate_scm_poisson_26_data**](#causalis.dgp.panel_data_scm.functional.generate_scm_poisson_26_data) –
- [**generate_scm_poisson_data**](#causalis.dgp.panel_data_scm.functional.generate_scm_poisson_data) – Medium-level wrapper for realistic Poisson SCM panel generation.

###### `PanelOutput`

```python
PanelOutput = Union[pd.DataFrame, PanelDataSCM]
```

###### `generate_scm_data`

```python
generate_scm_data(
    n_donors: int = 5,
    n_pre_periods: int = 20,
    n_post_periods: int = 10,
    treatment_effect: float = 2.0,
    treatment_effect_slope: float = 0.0,
    donor_noise_std: float = 0.2,
    treated_noise_std: float = 0.1,
    common_factor_std: float = 0.15,
    time_start: int = 1,
    treated_unit: Hashable = "treated",
    donor_prefix: str = "donor_",
    random_state: Optional[int] = 42,
    return_panel_data: bool = True,
    dirichlet_alpha: float = 1.0,
    rho_common: float = 0.0,
    rho_donor: float = 0.0,
    n_latent_factors: int = 0,
    latent_factor_std: float = 0.2,
    latent_loading_std: float = 0.35,
    rho_latent: float = 0.0,
    prefit_mismatch_std: float = 0.0,
    rho_prefit_mismatch: float = 0.0,
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive",
) -> PanelOutput
```

Medium-level wrapper for Gaussian SCM panel generation.

<details class="note" open markdown="1">
<summary>Notes</summary>

The Gaussian treated counterfactual path is exact SCM only when both
`treated_noise_std=0.0` and `prefit_mismatch_std=0.0`. With either term
non-zero, the generator intentionally produces an approximate-SCM fit path.

</details>

###### `generate_scm_gamma_26_data`

```python
generate_scm_gamma_26_data(
    *,
    seed: int,
    return_panel_data: bool,
    include_oracles: bool,
    n_donors: int,
    n_pre_periods: Optional[int],
    n_post_periods: Optional[int],
    treatment_effect_rate: float,
    treatment_effect_slope: float,
    advanced_params: dict[str, Any]
) -> PanelOutput
```

###### `generate_scm_gamma_data`

```python
generate_scm_gamma_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.12,
    treatment_effect_slope: float = 0.01,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params: Optional[int]
) -> PanelOutput
```

Medium-level wrapper for realistic Gamma SCM panel generation.

Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
are omitted, they are inferred from `n`.
The post-treatment effect path uses a ramp-in: at the first post period, the
effective relative lift is
`treatment_effect_rate * (1 - exp(-1 / 2.5))` (about 0.33x of the parameter
when slope is zero).

###### `generate_scm_poisson_26_data`

```python
generate_scm_poisson_26_data(
    *,
    seed: int,
    return_panel_data: bool,
    include_oracles: bool,
    n_donors: int,
    n_pre_periods: Optional[int],
    n_post_periods: Optional[int],
    treatment_effect_rate: float,
    treatment_effect_slope: float,
    advanced_params: dict[str, Any]
) -> PanelOutput
```

###### `generate_scm_poisson_data`

```python
generate_scm_poisson_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.1,
    treatment_effect_slope: float = 0.005,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params: Optional[int]
) -> PanelOutput
```

Medium-level wrapper for realistic Poisson SCM panel generation.

Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
are omitted, they are inferred from `n`.
The post-treatment effect path uses a ramp-in: at the first post period, the
effective relative lift is
`treatment_effect_rate * (1 - exp(-1 / 2.5))` (about 0.33x of the parameter
when slope is zero).

##### `generate_scm_data`

```python
generate_scm_data(
    n_donors: int = 5,
    n_pre_periods: int = 20,
    n_post_periods: int = 10,
    treatment_effect: float = 2.0,
    treatment_effect_slope: float = 0.0,
    donor_noise_std: float = 0.2,
    treated_noise_std: float = 0.1,
    common_factor_std: float = 0.15,
    time_start: int = 1,
    treated_unit: Hashable = "treated",
    donor_prefix: str = "donor_",
    random_state: Optional[int] = 42,
    return_panel_data: bool = True,
    dirichlet_alpha: float = 1.0,
    rho_common: float = 0.0,
    rho_donor: float = 0.0,
    n_latent_factors: int = 0,
    latent_factor_std: float = 0.2,
    latent_loading_std: float = 0.35,
    rho_latent: float = 0.0,
    prefit_mismatch_std: float = 0.0,
    rho_prefit_mismatch: float = 0.0,
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive",
) -> PanelOutput
```

Medium-level wrapper for Gaussian SCM panel generation.

<details class="note" open markdown="1">
<summary>Notes</summary>

The Gaussian treated counterfactual path is exact SCM only when both
`treated_noise_std=0.0` and `prefit_mismatch_std=0.0`. With either term
non-zero, the generator intentionally produces an approximate-SCM fit path.

</details>

##### `generate_scm_gamma_data`

```python
generate_scm_gamma_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.12,
    treatment_effect_slope: float = 0.01,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params: Optional[int]
) -> PanelOutput
```

Medium-level wrapper for realistic Gamma SCM panel generation.

Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
are omitted, they are inferred from `n`.
The post-treatment effect path uses a ramp-in: at the first post period, the
effective relative lift is
`treatment_effect_rate * (1 - exp(-1 / 2.5))` (about 0.33x of the parameter
when slope is zero).

##### `generate_scm_poisson_data`

```python
generate_scm_poisson_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.1,
    treatment_effect_slope: float = 0.005,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params: Optional[int]
) -> PanelOutput
```

Medium-level wrapper for realistic Poisson SCM panel generation.

Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
are omitted, they are inferred from `n`.
The post-treatment effect path uses a ramp-in: at the first post period, the
effective relative lift is
`treatment_effect_rate * (1 - exp(-1 / 2.5))` (about 0.33x of the parameter
when slope is zero).

### `causalis.scenarios`

**Modules:**

- [**classic_rct**](#causalis.scenarios.classic_rct) –
- [**cuped**](#causalis.scenarios.cuped) –
- [**gate**](#causalis.scenarios.gate) – Group Average Treatment Effect (GATE) estimation utilities.
- [**multi_unconfoundedness**](#causalis.scenarios.multi_unconfoundedness) –
- [**synthetic_control**](#causalis.scenarios.synthetic_control) –
- [**unconfoundedness**](#causalis.scenarios.unconfoundedness) –

#### `classic_rct`

**Modules:**

- [**conversion_ztest**](#causalis.scenarios.classic_rct.conversion_ztest) – Two-proportion z-test
- [**dgp**](#causalis.scenarios.classic_rct.dgp) –
- [**inference**](#causalis.scenarios.classic_rct.inference) – Inference helpers for the classic RCT scenario.
- [**model**](#causalis.scenarios.classic_rct.model) –
- [**rct_design**](#causalis.scenarios.classic_rct.rct_design) – Design module for experimental rct_design utilities.
- [**ttest**](#causalis.scenarios.classic_rct.ttest) – T-test inference for Diff_in_Means model

**Classes:**

- [**DiffInMeans**](#causalis.scenarios.classic_rct.DiffInMeans) – Difference-in-means model for CausalData.
- [**SRMResult**](#causalis.scenarios.classic_rct.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**bootstrap_diff_means**](#causalis.scenarios.classic_rct.bootstrap_diff_means) – Bootstrap inference for difference in means between treated and control groups.
- [**check_srm**](#causalis.scenarios.classic_rct.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

##### `DiffInMeans`

```python
DiffInMeans() -> None
```

Difference-in-means model for CausalData.
Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.

**Functions:**

- [**estimate**](#causalis.scenarios.classic_rct.DiffInMeans.estimate) – Compute the treatment effect using the specified method.
- [**fit**](#causalis.scenarios.classic_rct.DiffInMeans.fit) – Fit the model by storing the CausalData object.

###### `data`

```python
data: Optional[CausalData] = None
```

###### `estimate`

```python
estimate(
    method: Literal["ttest", "bootstrap", "conversion_ztest"] = "ttest",
    alpha: float = 0.05,
    diagnostic_data: bool = True,
    **kwargs: Any
) -> CausalEstimate
```

Compute the treatment effect using the specified method.

**Parameters:**

- **method** (<code>('ttest', 'bootstrap', 'conversion_ztest')</code>) – The inference method to use.
- "ttest": Standard independent two-sample t-test.
- "bootstrap": Bootstrap-based inference for difference in means.
- "conversion_ztest": Two-proportion z-test for binary outcomes.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the underlying inference function.
- For "bootstrap": can pass `n_simul`, `batch_size`, `seed`, `index_dtype`.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

###### `fit`

```python
fit(data: CausalData) -> DiffInMeans
```

Fit the model by storing the CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

**Returns:**

- <code>[DiffInMeans](#causalis.scenarios.classic_rct.model.DiffInMeans)</code> – The fitted model.

##### `SRMResult`

```python
SRMResult(
    chi2: float,
    p_value: float,
    expected: Dict[Hashable, float],
    observed: Dict[Hashable, int],
    alpha: float,
    is_srm: bool,
    warning: str | None = None,
) -> None
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.scenarios.classic_rct.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**p_value**](#causalis.scenarios.classic_rct.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test, rounded to 5 decimals.
- [**expected**](#causalis.scenarios.classic_rct.SRMResult.expected) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.scenarios.classic_rct.SRMResult.observed) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.scenarios.classic_rct.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.scenarios.classic_rct.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (chi-square p-value < alpha), False otherwise.
- [**warning**](#causalis.scenarios.classic_rct.SRMResult.warning) (<code>[str](#str) or None</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `alpha`

```python
alpha: float
```

###### `chi2`

```python
chi2: float
```

###### `expected`

```python
expected: Dict[Hashable, float]
```

###### `is_srm`

```python
is_srm: bool
```

###### `observed`

```python
observed: Dict[Hashable, int]
```

###### `p_value`

```python
p_value: float
```

###### `warning`

```python
warning: str | None = None
```

##### `bootstrap_diff_means`

```python
bootstrap_diff_means(
    data: CausalData,
    alpha: float = 0.05,
    n_simul: int = 10000,
    *,
    batch_size: int = 512,
    seed: Optional[int] = None,
    index_dtype: Optional[int] = np.int32
) -> Dict[str, Any]
```

Bootstrap inference for difference in means between treated and control groups.

This function computes the ATE-style difference in means (treated - control)
and provides a two-sided p-value using a normal approximation with bootstrap
standard error, a percentile confidence interval for the absolute difference,
and relative difference with its corresponding confidence interval.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.
- **batch_size** (<code>[int](#int)</code>) – Number of bootstrap samples to process per batch.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **index_dtype** (<code>numpy dtype</code>) – Integer dtype for bootstrap indices to reduce memory usage.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value using normal approximation.
- absolute_difference: The absolute difference (treated - control).
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
- relative_difference: The relative difference (%) relative to control mean.
- relative_ci: Tuple of (lower, upper) bounds for the relative difference CI (delta method).

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

##### `check_srm`

```python
check_srm(
    assignments: Union[
        Iterable[Hashable], pd.Series, CausalData, Mapping[Hashable, Number]
    ],
    target_allocation: Dict[Hashable, Number],
    alpha: float = 0.001,
    min_expected: float = 5.0,
    strict_variants: bool = True,
) -> SRMResult
```

Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData) or [Mapping](#collections.abc.Mapping)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Observed variant assignments. If iterable or Series, elements are labels per
  unit (user_id, session_id, etc.). If CausalData is provided, the treatment
  column is used. If a mapping is provided, it is treated as
  `{variant: observed_count}` with non-negative integer counts.
- **target_allocation** (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping `{variant: p}` describing intended allocation as probabilities.
- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Target allocation probabilities must sum to 1 within numerical tolerance.
- `is_srm` is computed using the unrounded p-value; the returned
  `p_value` is rounded to 5 decimals.
- Missing assignments are dropped and reported via `warning`.
- Requires SciPy for p-value computation.

</details>

**Examples:**

```pycon
>>> assignments = ["control"] * 50 + ["treatment"] * 50
>>> check_srm(assignments, {"control": 0.5, "treatment": 0.5}, alpha=1e-3)
SRMResult(status=no SRM, p_value=1.00000, chi2=0.0000)
```

```pycon
>>> counts = {"control": 70, "treatment": 30}
>>> check_srm(counts, {"control": 0.5, "treatment": 0.5})
SRMResult(status=SRM DETECTED, p_value=0.00006, chi2=16.0000)
```

##### `conversion_ztest`

Two-proportion z-test

Compares conversion rates between treated (D=1) and control (D=0) groups.
Returns p-value, absolute/relative differences, and their confidence intervals

**Functions:**

- [**conversion_ztest**](#causalis.scenarios.classic_rct.conversion_ztest.conversion_ztest) – Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

###### `conversion_ztest`

```python
conversion_ztest(
    data: CausalData,
    alpha: float = 0.05,
    ci_method: Literal["newcombe", "wald_unpooled", "wald_pooled"] = "newcombe",
    se_for_test: Literal["pooled", "unpooled"] = "pooled",
) -> Dict[str, Any]
```

Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **ci_method** (<code>([newcombe](#newcombe), [wald_unpooled](#wald_unpooled), [wald_pooled](#wald_pooled))</code>) – Method for calculating the confidence interval for the absolute difference.
  "newcombe" is the most robust default for conversion rates.
- **se_for_test** (<code>([pooled](#pooled), [unpooled](#unpooled))</code>) – Method for calculating the standard error for the z-test p-value.
  "pooled" (score test) is generally preferred for testing equality of proportions.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value from the z-test
- absolute_difference: Difference in conversion rates (treated - control)
- absolute_ci: Tuple (lower, upper) for the absolute difference CI
- relative_difference: Percentage change relative to control rate
- relative_ci: Tuple (lower, upper) for the relative difference CI (delta method)

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment/outcome are missing, treatment is not binary, outcome is not binary,
  groups are empty, or alpha is outside (0, 1).

##### `dgp`

**Functions:**

- [**classic_rct_gamma_26**](#causalis.scenarios.classic_rct.dgp.classic_rct_gamma_26) – A pre-configured classic RCT dataset with a gamma outcome.
- [**generate_classic_rct_26**](#causalis.scenarios.classic_rct.dgp.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.

###### `classic_rct_gamma_26`

```python
classic_rct_gamma_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = True,
    **kwargs: bool
)
```

A pre-configured classic RCT dataset with a gamma outcome.
n=10000, split=0.5, mean uplift ~10%.
Includes deterministic `user_id` and ancillary columns.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre').
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **n** (<code>[int](#int)</code>) – Number of samples.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **outcome_params** (<code>[dict](#dict)</code>) – Gamma outcome parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- \*\***kwargs** – Additional arguments passed to `classic_rct_gamma`.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

###### `generate_classic_rct_26`

```python
generate_classic_rct_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = False,
    deterministic_ids: bool = True,
    **kwargs: bool
)
```

A pre-configured classic RCT dataset with 3 binary confounders.
n=10000, split=0.5, outcome is conversion (binary). Baseline control p=0.10
and treatment p=0.11 are set on the log-odds scale (X=0), so marginal rates
and ATE can differ once covariate effects are included. Includes a
deterministic `user_id` column.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **n** (<code>[int](#int)</code>) – Number of samples.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **outcome_params** (<code>[dict](#dict)</code>) – Binary outcome parameters, e.g. {"p": {"A": 0.10, "B": 0.11}}.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- \*\***kwargs** – Additional arguments passed to `generate_classic_rct`.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

##### `inference`

Inference helpers for the classic RCT scenario.

**Modules:**

- [**bootstrap_diff_in_means**](#causalis.scenarios.classic_rct.inference.bootstrap_diff_in_means) – Bootstrap difference-in-means inference.
- [**conversion_ztest**](#causalis.scenarios.classic_rct.inference.conversion_ztest) – Two-proportion z-test
- [**ttest**](#causalis.scenarios.classic_rct.inference.ttest) – T-test inference for Diff_in_Means model

**Functions:**

- [**bootstrap_diff_means**](#causalis.scenarios.classic_rct.inference.bootstrap_diff_means) – Bootstrap inference for difference in means between treated and control groups.

###### `bootstrap_diff_in_means`

Bootstrap difference-in-means inference.

This module computes the ATE-style difference in means (treated - control) and provides:

- Two-sided p-value using a normal approximation with bootstrap standard error.
- Percentile confidence interval for the absolute difference.
- Relative difference (%) and corresponding CI relative to control mean.

**Functions:**

- [**bootstrap_diff_means**](#causalis.scenarios.classic_rct.inference.bootstrap_diff_in_means.bootstrap_diff_means) – Bootstrap inference for difference in means between treated and control groups.

####### `bootstrap_diff_means`

```python
bootstrap_diff_means(
    data: CausalData,
    alpha: float = 0.05,
    n_simul: int = 10000,
    *,
    batch_size: int = 512,
    seed: Optional[int] = None,
    index_dtype: Optional[int] = np.int32
) -> Dict[str, Any]
```

Bootstrap inference for difference in means between treated and control groups.

This function computes the ATE-style difference in means (treated - control)
and provides a two-sided p-value using a normal approximation with bootstrap
standard error, a percentile confidence interval for the absolute difference,
and relative difference with its corresponding confidence interval.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.
- **batch_size** (<code>[int](#int)</code>) – Number of bootstrap samples to process per batch.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **index_dtype** (<code>numpy dtype</code>) – Integer dtype for bootstrap indices to reduce memory usage.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value using normal approximation.
- absolute_difference: The absolute difference (treated - control).
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
- relative_difference: The relative difference (%) relative to control mean.
- relative_ci: Tuple of (lower, upper) bounds for the relative difference CI (delta method).

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

###### `bootstrap_diff_means`

```python
bootstrap_diff_means(
    data: CausalData,
    alpha: float = 0.05,
    n_simul: int = 10000,
    *,
    batch_size: int = 512,
    seed: Optional[int] = None,
    index_dtype: Optional[int] = np.int32
) -> Dict[str, Any]
```

Bootstrap inference for difference in means between treated and control groups.

This function computes the ATE-style difference in means (treated - control)
and provides a two-sided p-value using a normal approximation with bootstrap
standard error, a percentile confidence interval for the absolute difference,
and relative difference with its corresponding confidence interval.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.
- **batch_size** (<code>[int](#int)</code>) – Number of bootstrap samples to process per batch.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **index_dtype** (<code>numpy dtype</code>) – Integer dtype for bootstrap indices to reduce memory usage.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value using normal approximation.
- absolute_difference: The absolute difference (treated - control).
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
- relative_difference: The relative difference (%) relative to control mean.
- relative_ci: Tuple of (lower, upper) bounds for the relative difference CI (delta method).

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

###### `conversion_ztest`

Two-proportion z-test

Compares conversion rates between treated (D=1) and control (D=0) groups.
Returns p-value, absolute/relative differences, and their confidence intervals

**Functions:**

- [**conversion_ztest**](#causalis.scenarios.classic_rct.inference.conversion_ztest.conversion_ztest) – Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

####### `conversion_ztest`

```python
conversion_ztest(
    data: CausalData,
    alpha: float = 0.05,
    ci_method: Literal["newcombe", "wald_unpooled", "wald_pooled"] = "newcombe",
    se_for_test: Literal["pooled", "unpooled"] = "pooled",
) -> Dict[str, Any]
```

Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **ci_method** (<code>([newcombe](#newcombe), [wald_unpooled](#wald_unpooled), [wald_pooled](#wald_pooled))</code>) – Method for calculating the confidence interval for the absolute difference.
  "newcombe" is the most robust default for conversion rates.
- **se_for_test** (<code>([pooled](#pooled), [unpooled](#unpooled))</code>) – Method for calculating the standard error for the z-test p-value.
  "pooled" (score test) is generally preferred for testing equality of proportions.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value from the z-test
- absolute_difference: Difference in conversion rates (treated - control)
- absolute_ci: Tuple (lower, upper) for the absolute difference CI
- relative_difference: Percentage change relative to control rate
- relative_ci: Tuple (lower, upper) for the relative difference CI (delta method)

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment/outcome are missing, treatment is not binary, outcome is not binary,
  groups are empty, or alpha is outside (0, 1).

###### `ttest`

T-test inference for Diff_in_Means model

**Functions:**

- [**ttest**](#causalis.scenarios.classic_rct.inference.ttest.ttest) – Perform a Welch two-sample t-test comparing outcomes between treated (D=1)

####### `ttest`

```python
ttest(data: CausalData, alpha: float = 0.05) -> Dict[str, Any]
```

Perform a Welch two-sample t-test comparing outcomes between treated (D=1)
and control (D=0) groups.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – - p_value: Welch t-test p-value for H0: E[Y|D=1] - E[Y|D=0] = 0
- absolute_difference: treatment_mean - control_mean
- absolute_ci: (lower, upper) CI for absolute_difference using Welch df
- relative_difference: signed percent change = 100 * (treatment_mean / control_mean - 1)
- relative_se: delta-method SE of relative_difference (percent scale)
- relative_ci: (lower, upper) CI for relative_difference using delta method (+ Satterthwaite df)

<details class="note" open markdown="1">
<summary>Notes</summary>

Delta method for relative percent change:
r_hat = 100 * (Ybar1/Ybar0 - 1)

With independent groups and CLT:
Var(Ybar1) ≈ s1^2/n1
Var(Ybar0) ≈ s0^2/n2
Cov(Ybar1, Ybar0) ≈ 0

Gradient of g(a,b)=a/b - 1 is (1/b, -a/b^2), so:
Var(r_hat/100) ≈ (1/Ybar0)^2 * (s1^2/n1) + (Ybar1/Ybar0^2)^2 * (s0^2/n2)

CI uses t-critical with Satterthwaite df; falls back to z if df is invalid.
If control_mean is near 0, relative stats are undefined/unstable and return inf/nan sentinels.

</details>

##### `model`

**Classes:**

- [**DiffInMeans**](#causalis.scenarios.classic_rct.model.DiffInMeans) – Difference-in-means model for CausalData.

###### `DiffInMeans`

```python
DiffInMeans() -> None
```

Difference-in-means model for CausalData.
Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.

**Functions:**

- [**estimate**](#causalis.scenarios.classic_rct.model.DiffInMeans.estimate) – Compute the treatment effect using the specified method.
- [**fit**](#causalis.scenarios.classic_rct.model.DiffInMeans.fit) – Fit the model by storing the CausalData object.

####### `data`

```python
data: Optional[CausalData] = None
```

####### `estimate`

```python
estimate(
    method: Literal["ttest", "bootstrap", "conversion_ztest"] = "ttest",
    alpha: float = 0.05,
    diagnostic_data: bool = True,
    **kwargs: Any
) -> CausalEstimate
```

Compute the treatment effect using the specified method.

**Parameters:**

- **method** (<code>('ttest', 'bootstrap', 'conversion_ztest')</code>) – The inference method to use.
- "ttest": Standard independent two-sample t-test.
- "bootstrap": Bootstrap-based inference for difference in means.
- "conversion_ztest": Two-proportion z-test for binary outcomes.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the underlying inference function.
- For "bootstrap": can pass `n_simul`, `batch_size`, `seed`, `index_dtype`.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

####### `fit`

```python
fit(data: CausalData) -> DiffInMeans
```

Fit the model by storing the CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

**Returns:**

- <code>[DiffInMeans](#causalis.scenarios.classic_rct.model.DiffInMeans)</code> – The fitted model.

##### `rct_design`

Design module for experimental rct_design utilities.

**Classes:**

- [**SRMResult**](#causalis.scenarios.classic_rct.rct_design.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**assign_variants_df**](#causalis.scenarios.classic_rct.rct_design.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.
- [**calculate_mde**](#causalis.scenarios.classic_rct.rct_design.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.
- [**check_srm**](#causalis.scenarios.classic_rct.rct_design.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

###### `SRMResult`

```python
SRMResult(
    chi2: float,
    p_value: float,
    expected: Dict[Hashable, float],
    observed: Dict[Hashable, int],
    alpha: float,
    is_srm: bool,
    warning: str | None = None,
) -> None
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.scenarios.classic_rct.rct_design.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**p_value**](#causalis.scenarios.classic_rct.rct_design.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test, rounded to 5 decimals.
- [**expected**](#causalis.scenarios.classic_rct.rct_design.SRMResult.expected) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.scenarios.classic_rct.rct_design.SRMResult.observed) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.scenarios.classic_rct.rct_design.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.scenarios.classic_rct.rct_design.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (chi-square p-value < alpha), False otherwise.
- [**warning**](#causalis.scenarios.classic_rct.rct_design.SRMResult.warning) (<code>[str](#str) or None</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

####### `alpha`

```python
alpha: float
```

####### `chi2`

```python
chi2: float
```

####### `expected`

```python
expected: Dict[Hashable, float]
```

####### `is_srm`

```python
is_srm: bool
```

####### `observed`

```python
observed: Dict[Hashable, int]
```

####### `p_value`

```python
p_value: float
```

####### `warning`

```python
warning: str | None = None
```

###### `assign_variants_df`

```python
assign_variants_df(
    df: pd.DataFrame,
    id_col: str,
    experiment_id: str,
    variants: Dict[str, float],
    *,
    salt: str = "global_ab_salt",
    layer_id: str = "default",
    variant_col: str = "variant"
) -> pd.DataFrame
```

Deterministically assign variants for each row in df based on id_col.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – Input DataFrame with an identifier column.
- **id_col** (<code>[str](#str)</code>) – Column name in df containing entity identifiers (user_id, session_id, etc.).
- **experiment_id** (<code>[str](#str)</code>) – Unique identifier for the experiment (versioned for reruns).
- **variants** (<code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code>) – Mapping from variant name to weight (coverage). Weights must be non-negative
  and their sum must be in (0, 1\]. If the sum is < 1, the remaining mass
  corresponds to "not in experiment" and the assignment will be None.
- **salt** (<code>[str](#str)</code>) – Secret string to de-correlate from other hash uses and make assignments
  non-gameable.
- **layer_id** (<code>[str](#str)</code>) – Identifier for mutual exclusivity layer or surface. In this case work like
  another random
- **variant_col** (<code>[str](#str)</code>) – Name of output column to store assigned variant labels.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of df with an extra column `variant_col`.
  Entities outside experiment coverage will have None in the variant column.

###### `calculate_mde`

```python
calculate_mde(
    sample_size: Union[int, Tuple[int, int]],
    baseline_rate: Optional[float] = None,
    variance: Optional[Union[float, Tuple[float, float]]] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    data_type: str = "conversion",
    ratio: float = 0.5,
) -> Dict[str, Any]
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data_contracts) or baseline mean (for continuous data_contracts).
  Required for conversion data_contracts.
- **variance** (<code>float or tuple of float</code>) – Variance of the data_contracts. For conversion data_contracts, this is calculated from the baseline rate if not provided.
  For continuous data_contracts, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data_contracts. Either 'conversion' for binary/conversion data_contracts or 'continuous' for continuous data_contracts.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data_contracts with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data_contracts with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

###### `check_srm`

```python
check_srm(
    assignments: Union[
        Iterable[Hashable], pd.Series, CausalData, Mapping[Hashable, Number]
    ],
    target_allocation: Dict[Hashable, Number],
    alpha: float = 0.001,
    min_expected: float = 5.0,
    strict_variants: bool = True,
) -> SRMResult
```

Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData) or [Mapping](#collections.abc.Mapping)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Observed variant assignments. If iterable or Series, elements are labels per
  unit (user_id, session_id, etc.). If CausalData is provided, the treatment
  column is used. If a mapping is provided, it is treated as
  `{variant: observed_count}` with non-negative integer counts.
- **target_allocation** (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping `{variant: p}` describing intended allocation as probabilities.
- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Target allocation probabilities must sum to 1 within numerical tolerance.
- `is_srm` is computed using the unrounded p-value; the returned
  `p_value` is rounded to 5 decimals.
- Missing assignments are dropped and reported via `warning`.
- Requires SciPy for p-value computation.

</details>

**Examples:**

```pycon
>>> assignments = ["control"] * 50 + ["treatment"] * 50
>>> check_srm(assignments, {"control": 0.5, "treatment": 0.5}, alpha=1e-3)
SRMResult(status=no SRM, p_value=1.00000, chi2=0.0000)
```

```pycon
>>> counts = {"control": 70, "treatment": 30}
>>> check_srm(counts, {"control": 0.5, "treatment": 0.5})
SRMResult(status=SRM DETECTED, p_value=0.00006, chi2=16.0000)
```

##### `ttest`

T-test inference for Diff_in_Means model

**Functions:**

- [**ttest**](#causalis.scenarios.classic_rct.ttest.ttest) – Perform a Welch two-sample t-test comparing outcomes between treated (D=1)

###### `ttest`

```python
ttest(data: CausalData, alpha: float = 0.05) -> Dict[str, Any]
```

Perform a Welch two-sample t-test comparing outcomes between treated (D=1)
and control (D=0) groups.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – - p_value: Welch t-test p-value for H0: E[Y|D=1] - E[Y|D=0] = 0
- absolute_difference: treatment_mean - control_mean
- absolute_ci: (lower, upper) CI for absolute_difference using Welch df
- relative_difference: signed percent change = 100 * (treatment_mean / control_mean - 1)
- relative_se: delta-method SE of relative_difference (percent scale)
- relative_ci: (lower, upper) CI for relative_difference using delta method (+ Satterthwaite df)

<details class="note" open markdown="1">
<summary>Notes</summary>

Delta method for relative percent change:
r_hat = 100 * (Ybar1/Ybar0 - 1)

With independent groups and CLT:
Var(Ybar1) ≈ s1^2/n1
Var(Ybar0) ≈ s0^2/n2
Cov(Ybar1, Ybar0) ≈ 0

Gradient of g(a,b)=a/b - 1 is (1/b, -a/b^2), so:
Var(r_hat/100) ≈ (1/Ybar0)^2 * (s1^2/n1) + (Ybar1/Ybar0^2)^2 * (s0^2/n2)

CI uses t-critical with Satterthwaite df; falls back to z if df is invalid.
If control_mean is near 0, relative stats are undefined/unstable and return inf/nan sentinels.

</details>

#### `cuped`

**Modules:**

- [**dgp**](#causalis.scenarios.cuped.dgp) –
- [**diagnostics**](#causalis.scenarios.cuped.diagnostics) –
- [**model**](#causalis.scenarios.cuped.model) –

**Classes:**

- [**CUPEDModel**](#causalis.scenarios.cuped.CUPEDModel) – CUPED-style regression adjustment estimator for ATE/ITT in randomized experiments.

**Functions:**

- [**cuped_forest_plot**](#causalis.scenarios.cuped.cuped_forest_plot) – Forest plot of absolute estimates and CIs for CUPED vs non-CUPED.
- [**regression_assumptions_table_from_data**](#causalis.scenarios.cuped.regression_assumptions_table_from_data) – Fit CUPED on `CausalData` and return the assumptions flag table.
- [**regression_assumptions_table_from_estimate**](#causalis.scenarios.cuped.regression_assumptions_table_from_estimate) – Build assumptions table from a CUPED estimate.
- [**style_regression_assumptions_table**](#causalis.scenarios.cuped.style_regression_assumptions_table) – Return pandas Styler with colored flag cells for notebook display.

##### `CUPEDModel`

```python
CUPEDModel(
    cov_type: str = "HC2",
    alpha: float = 0.05,
    strict_binary_treatment: bool = True,
    use_t: Optional[bool] = None,
    use_t_auto_n_threshold: int = 5000,
    relative_ci_method: Literal["delta_nocov", "bootstrap"] = "delta_nocov",
    relative_ci_bootstrap_draws: int = 1000,
    relative_ci_bootstrap_seed: Optional[int] = None,
    covariate_variance_min: float = 1e-12,
    condition_number_warn_threshold: float = 100000000.0,
    run_regression_checks: bool = True,
    check_action: Literal["ignore", "raise"] = "ignore",
    raise_on_yellow: bool = False,
    corr_near_one_tol: float = 1e-10,
    vif_warn_threshold: float = 20.0,
    winsor_q: Optional[float] = 0.01,
    tiny_one_minus_h_tol: float = 1e-08,
) -> None
```

CUPED-style regression adjustment estimator for ATE/ITT in randomized experiments.

Fits an outcome regression with pre-treatment covariates (always centered
over the full sample, never within treatment groups)
implemented as Lin (2013) fully interacted OLS:

```
Y ~ 1 + D + X^c + D * X^c
```

The reported effect is the coefficient on D, with robust covariance as requested.
This specification ensures the coefficient on D is the ATE/ITT even if the
treatment effect is heterogeneous with respect to covariates.
This is broader than canonical single-theta CUPED (`Y - theta*(X - mean(X))`).

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
  Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
- **alpha** (<code>[float](#float)</code>) – Significance level for confidence intervals.
- **strict_binary_treatment** (<code>[bool](#bool)</code>) – If True, require treatment to be binary {0,1}.
- **use_t** (<code>[bool](#bool) | None</code>) – If bool, passed to statsmodels `.fit(..., use_t=use_t)` directly.
  If None, automatic policy is used: for robust HC\* covariances,
  `use_t=True` when `n < use_t_auto_n_threshold`, else `False`.
  For non-robust covariance, `use_t=True`.
- **use_t_auto_n_threshold** (<code>[int](#int)</code>) – Sample-size threshold for automatic `use_t` selection when `use_t=None`
  and covariance is HC\* robust.
- **relative_ci_method** (<code>('delta_nocov', 'bootstrap')</code>) – Method for relative CI of `100 * tau / mu_c`.
- "delta_nocov": delta method using robust `Var(tau)` and `Var(mu_c)` while
  setting `Cov(tau, mu_c)=0` (safe fallback without unsupported hybrid IF covariance).
- "bootstrap": percentile bootstrap CI on the relative effect.
- **relative_ci_bootstrap_draws** (<code>[int](#int)</code>) – Number of bootstrap resamples used when `relative_ci_method="bootstrap"`.
- **relative_ci_bootstrap_seed** (<code>[int](#int) | None</code>) – RNG seed used for bootstrap relative CI.
- **covariate_variance_min** (<code>[float](#float)</code>) – Minimum variance threshold for retaining a CUPED covariate. Covariates with
  variance less than or equal to this threshold are dropped before fitting.
- **condition_number_warn_threshold** (<code>[float](#float)</code>) – Trigger diagnostics signal when the design matrix condition number exceeds this threshold.
- **run_regression_checks** (<code>[bool](#bool)</code>) – Whether to compute regression diagnostics payload during `fit()`.
- **check_action** (<code>('ignore', 'raise')</code>) – Action used when a diagnostics threshold is violated.
- **raise_on_yellow** (<code>[bool](#bool)</code>) – When `check_action="raise"`, also raise on YELLOW assumption flags.
- **corr_near_one_tol** (<code>[float](#float)</code>) – Correlation tolerance used to mark near-duplicate centered covariates.
- **vif_warn_threshold** (<code>[float](#float)</code>) – VIF threshold that triggers a diagnostics signal.
- **winsor_q** (<code>[float](#float) | None</code>) – Quantile used for winsor sensitivity refit. Set `None` to disable.
- **tiny_one_minus_h_tol** (<code>[float](#float)</code>) – Threshold for flagging near-degenerate `1 - leverage` terms in HC2/HC3.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
- Covariates are globally centered over the full sample only. This centering
  convention is required so the treatment coefficient in the Lin specification
  remains the ATE/ITT.
- The Lin (2013) specification is recommended as a robust regression-adjustment default
  in RCTs.

</details>

**Functions:**

- [**assumptions_table**](#causalis.scenarios.cuped.CUPEDModel.assumptions_table) – Return fitted regression assumptions table (GREEN/YELLOW/RED) when available.
- [**estimate**](#causalis.scenarios.cuped.CUPEDModel.estimate) – Return the adjusted ATE/ITT estimate and inference.
- [**fit**](#causalis.scenarios.cuped.CUPEDModel.fit) – Fit CUPED-style regression adjustment (Lin-interacted OLS) on a CausalData object.
- [**summary_dict**](#causalis.scenarios.cuped.CUPEDModel.summary_dict) – Convenience JSON/logging output.

###### `adjustment`

```python
adjustment: Literal['lin'] = 'lin'
```

###### `alpha`

```python
alpha = float(alpha)
```

###### `assumptions_table`

```python
assumptions_table() -> Optional[pd.DataFrame]
```

Return fitted regression assumptions table (GREEN/YELLOW/RED) when available.

###### `center_covariates`

```python
center_covariates = True
```

###### `centering_scope`

```python
centering_scope: Literal['global'] = 'global'
```

###### `check_action`

```python
check_action: Literal['ignore', 'raise'] = check_action
```

###### `condition_number_warn_threshold`

```python
condition_number_warn_threshold = float(condition_number_warn_threshold)
```

###### `corr_near_one_tol`

```python
corr_near_one_tol = float(corr_near_one_tol)
```

###### `cov_type`

```python
cov_type = str(cov_type)
```

###### `covariate_variance_min`

```python
covariate_variance_min = float(covariate_variance_min)
```

###### `estimate`

```python
estimate(
    alpha: Optional[float] = None, diagnostic_data: bool = True
) -> CausalEstimate
```

Return the adjusted ATE/ITT estimate and inference.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

###### `fit`

```python
fit(
    data: CausalData,
    covariates: Optional[Sequence[str]] = None,
    run_checks: Optional[bool] = None,
) -> CUPEDModel
```

Fit CUPED-style regression adjustment (Lin-interacted OLS) on a CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
- **covariates** (<code>([Sequence](#typing.Sequence)\[[str](#str)\], [required](#required))</code>) – Explicit subset of `data_contracts.confounders_names` to use as CUPED covariates.
  Pass `[]` for an unadjusted (naive) fit.
- **run_checks** (<code>[bool](#bool) | None</code>) – Override whether regression checks are computed in this fit call.
  If `None`, uses `self.run_regression_checks`.

**Returns:**

- <code>[CUPEDModel](#causalis.scenarios.cuped.model.CUPEDModel)</code> – Fitted estimator.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If `covariates` is omitted, not a sequence of strings, contains columns missing from the
  DataFrame, contains columns outside `data_contracts.confounders_names`,
  treatment is not binary when `strict_binary_treatment=True`,
  or the design matrix is rank deficient.

###### `raise_on_yellow`

```python
raise_on_yellow = bool(raise_on_yellow)
```

###### `relative_ci_bootstrap_draws`

```python
relative_ci_bootstrap_draws = int(relative_ci_bootstrap_draws)
```

###### `relative_ci_bootstrap_seed`

```python
relative_ci_bootstrap_seed = relative_ci_bootstrap_seed
```

###### `relative_ci_method`

```python
relative_ci_method: Literal['delta_nocov', 'bootstrap'] = relative_ci_method
```

###### `run_regression_checks`

```python
run_regression_checks = bool(run_regression_checks)
```

###### `strict_binary_treatment`

```python
strict_binary_treatment = bool(strict_binary_treatment)
```

###### `summary_dict`

```python
summary_dict(alpha: Optional[float] = None) -> Dict[str, Any]
```

Convenience JSON/logging output.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with estimates, inference, and diagnostics.

###### `tiny_one_minus_h_tol`

```python
tiny_one_minus_h_tol = float(tiny_one_minus_h_tol)
```

###### `use_t`

```python
use_t = None if use_t is None else bool(use_t)
```

###### `use_t_auto_n_threshold`

```python
use_t_auto_n_threshold = int(use_t_auto_n_threshold)
```

###### `vif_warn_threshold`

```python
vif_warn_threshold = float(vif_warn_threshold)
```

###### `winsor_q`

```python
winsor_q = None
```

##### `cuped_forest_plot`

```python
cuped_forest_plot(
    estimate_with_cuped: CausalEstimate,
    estimate_without_cuped: Optional[CausalEstimate] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.5, 3.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    label_with_cuped: str = "With CUPED",
    label_without_cuped: str = "Without CUPED",
    color_with_cuped: str = "C0",
    color_without_cuped: str = "C1",
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure
```

Forest plot of absolute estimates and CIs for CUPED vs non-CUPED.

**Parameters:**

- **estimate_with_cuped** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Effect estimated with CUPED adjustment.
- **estimate_without_cuped** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Effect estimated without CUPED adjustment. If omitted, the function
  uses `estimate_with_cuped.diagnostic_data.ate_naive` and
  `estimate_with_cuped.diagnostic_data.se_naive` to build a normal-approx CI.

##### `dgp`

**Functions:**

- [**generate_cuped_tweedie_26**](#causalis.scenarios.cuped.dgp.generate_cuped_tweedie_26) – Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_cuped_binary_26**](#causalis.scenarios.cuped.dgp.make_cuped_binary_26) – Binary CUPED benchmark with richer confounders and structured HTE.

###### `generate_cuped_tweedie_26`

```python
generate_cuped_tweedie_26(
    n: int = 20000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_name_2: Optional[str] = None,
    pre_target_corr: float = 0.82,
    pre_target_corr_2: Optional[float] = None,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail.
Includes two pre-period covariates by default: 'y_pre' and 'y_pre_2'.
Wrapper for make_tweedie().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add pre-period covariates.
- **pre_name** (<code>[str](#str)</code>) – Name of the first pre-period covariate column.
- **pre_name_2** (<code>[str](#str)</code>) – Name of the second pre-period covariate column.
  Defaults to `f"{pre_name}_2"`.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between the first pre covariate and post-outcome y in control group.
- **pre_target_corr_2** (<code>[float](#float)</code>) – Target correlation for the second pre covariate. Defaults to a
  moderate value based on `pre_target_corr` to reduce collinearity.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

###### `make_cuped_binary_26`

```python
make_cuped_binary_26(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38,
) -> Union[pd.DataFrame, CausalData]
```

Binary CUPED benchmark with richer confounders and structured HTE.
Includes a calibrated pre-period covariate 'y_pre' by default.
Wrapper for generate_cuped_binary().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in the control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like 'cate', 'g0', and 'g1'.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_logit** (<code>[float](#float)</code>) – Baseline log-odds uplift scale for heterogeneous treatment effects.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

##### `diagnostics`

**Modules:**

- [**forest_plot**](#causalis.scenarios.cuped.diagnostics.forest_plot) –
- [**regression_checks**](#causalis.scenarios.cuped.diagnostics.regression_checks) –

**Classes:**

- [**RegressionChecks**](#causalis.scenarios.cuped.diagnostics.RegressionChecks) – Lightweight OLS/regression health checks for CUPED diagnostics.

**Functions:**

- [**assumption_ate_gap**](#causalis.scenarios.cuped.diagnostics.assumption_ate_gap) – Check adjusted-vs-naive ATE gap relative to naive SE.
- [**assumption_condition_number**](#causalis.scenarios.cuped.diagnostics.assumption_condition_number) – Check global collinearity via condition number.
- [**assumption_cooks**](#causalis.scenarios.cuped.diagnostics.assumption_cooks) – Check Cook's distance influence diagnostics.
- [**assumption_design_rank**](#causalis.scenarios.cuped.diagnostics.assumption_design_rank) – Check that the design matrix is full rank.
- [**assumption_hc23_stability**](#causalis.scenarios.cuped.diagnostics.assumption_hc23_stability) – Check HC2/HC3 stability when leverage terms approach one.
- [**assumption_leverage**](#causalis.scenarios.cuped.diagnostics.assumption_leverage) – Check leverage concentration.
- [**assumption_near_duplicates**](#causalis.scenarios.cuped.diagnostics.assumption_near_duplicates) – Check near-duplicate centered covariate pairs.
- [**assumption_residual_tails**](#causalis.scenarios.cuped.diagnostics.assumption_residual_tails) – Check residual extremes using max standardized residual only.
- [**assumption_vif**](#causalis.scenarios.cuped.diagnostics.assumption_vif) – Check VIF from centered main-effect covariates.
- [**assumption_winsor_sensitivity**](#causalis.scenarios.cuped.diagnostics.assumption_winsor_sensitivity) – Check sensitivity of adjusted ATE to winsorized-outcome refit.
- [**cuped_forest_plot**](#causalis.scenarios.cuped.diagnostics.cuped_forest_plot) – Forest plot of absolute estimates and CIs for CUPED vs non-CUPED.
- [**design_matrix_checks**](#causalis.scenarios.cuped.diagnostics.design_matrix_checks) – Return rank/conditioning diagnostics for a numeric design matrix.
- [**overall_assumption_flag**](#causalis.scenarios.cuped.diagnostics.overall_assumption_flag) – Return overall GREEN/YELLOW/RED status from an assumptions table.
- [**regression_assumption_rows_from_checks**](#causalis.scenarios.cuped.diagnostics.regression_assumption_rows_from_checks) – Run all CUPED regression assumption tests and return row payloads.
- [**regression_assumptions_table_from_checks**](#causalis.scenarios.cuped.diagnostics.regression_assumptions_table_from_checks) – Return a table of GREEN/YELLOW/RED assumption flags from checks payload.
- [**regression_assumptions_table_from_data**](#causalis.scenarios.cuped.diagnostics.regression_assumptions_table_from_data) – Fit CUPED on `CausalData` and return the assumptions flag table.
- [**regression_assumptions_table_from_diagnostic_data**](#causalis.scenarios.cuped.diagnostics.regression_assumptions_table_from_diagnostic_data) – Build assumption table from `CUPEDDiagnosticData` payload.
- [**regression_assumptions_table_from_estimate**](#causalis.scenarios.cuped.diagnostics.regression_assumptions_table_from_estimate) – Build assumptions table from a CUPED estimate.
- [**run_regression_checks**](#causalis.scenarios.cuped.diagnostics.run_regression_checks) – Build a compact payload with design, residual, and influence diagnostics.
- [**style_regression_assumptions_table**](#causalis.scenarios.cuped.diagnostics.style_regression_assumptions_table) – Return pandas Styler with colored flag cells for notebook display.

###### `FLAG_GREEN`

```python
FLAG_GREEN = 'GREEN'
```

###### `FLAG_RED`

```python
FLAG_RED = 'RED'
```

###### `FLAG_YELLOW`

```python
FLAG_YELLOW = 'YELLOW'
```

###### `RegressionChecks`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Lightweight OLS/regression health checks for CUPED diagnostics.

####### `ate_adj`

```python
ate_adj: float
```

####### `ate_adj_winsor`

```python
ate_adj_winsor: Optional[float] = None
```

####### `ate_adj_winsor_gap`

```python
ate_adj_winsor_gap: Optional[float] = None
```

####### `ate_gap`

```python
ate_gap: float
```

####### `ate_gap_over_se_naive`

```python
ate_gap_over_se_naive: Optional[float] = None
```

####### `ate_naive`

```python
ate_naive: float
```

####### `condition_number`

```python
condition_number: float
```

####### `cooks_cutoff`

```python
cooks_cutoff: float
```

####### `full_rank`

```python
full_rank: bool
```

####### `k`

```python
k: int
```

####### `leverage_cutoff`

```python
leverage_cutoff: float
```

####### `max_abs_std_resid`

```python
max_abs_std_resid: float
```

####### `max_cooks`

```python
max_cooks: float
```

####### `max_leverage`

```python
max_leverage: float
```

####### `min_one_minus_h`

```python
min_one_minus_h: float
```

####### `n_high_cooks`

```python
n_high_cooks: int
```

####### `n_high_leverage`

```python
n_high_leverage: int
```

####### `n_std_resid_gt_3`

```python
n_std_resid_gt_3: int
```

####### `n_std_resid_gt_4`

```python
n_std_resid_gt_4: int
```

####### `n_tiny_one_minus_h`

```python
n_tiny_one_minus_h: int
```

####### `near_duplicate_pairs`

```python
near_duplicate_pairs: list[tuple[str, str, float]] = Field(default_factory=list)

```

####### `p_main_covariates`

```python
p_main_covariates: int
```

####### `rank`

```python
rank: int
```

####### `resid_scale_mad`

```python
resid_scale_mad: float
```

####### `vif`

```python
vif: Optional[Dict[str, float]] = None
```

####### `winsor_q`

```python
winsor_q: Optional[float] = None
```

###### `assumption_ate_gap`

```python
assumption_ate_gap(
    checks: RegressionChecks,
    yellow_threshold: float = 2.0,
    red_threshold: float = 2.5,
) -> Dict[str, Any]
```

Check adjusted-vs-naive ATE gap relative to naive SE.

###### `assumption_condition_number`

```python
assumption_condition_number(
    checks: RegressionChecks,
    warn_threshold: float = 100000000.0,
    red_multiplier: float = 100.0,
) -> Dict[str, Any]
```

Check global collinearity via condition number.

###### `assumption_cooks`

```python
assumption_cooks(
    checks: RegressionChecks,
    yellow_threshold: float = 0.1,
    red_threshold: float = 1.0,
) -> Dict[str, Any]
```

Check Cook's distance influence diagnostics.

###### `assumption_design_rank`

```python
assumption_design_rank(checks: RegressionChecks) -> Dict[str, Any]
```

Check that the design matrix is full rank.

###### `assumption_hc23_stability`

```python
assumption_hc23_stability(
    checks: RegressionChecks, cov_type: str, tiny_one_minus_h_tol: float = 1e-08
) -> Dict[str, Any]
```

Check HC2/HC3 stability when leverage terms approach one.

###### `assumption_leverage`

```python
assumption_leverage(
    checks: RegressionChecks,
    yellow_multiplier: float = 5.0,
    red_multiplier: float = 10.0,
    red_floor: float = 0.5,
) -> Dict[str, Any]
```

Check leverage concentration.

###### `assumption_near_duplicates`

```python
assumption_near_duplicates(
    checks: RegressionChecks, red_pairs_threshold: int = 3
) -> Dict[str, Any]
```

Check near-duplicate centered covariate pairs.

###### `assumption_residual_tails`

```python
assumption_residual_tails(
    checks: RegressionChecks,
    yellow_abs_std_resid: float = 7.0,
    red_abs_std_resid: float = 10.0,
) -> Dict[str, Any]
```

Check residual extremes using max standardized residual only.

###### `assumption_vif`

```python
assumption_vif(
    checks: RegressionChecks,
    warn_threshold: float = 20.0,
    red_multiplier: float = 2.0,
) -> Dict[str, Any]
```

Check VIF from centered main-effect covariates.

###### `assumption_winsor_sensitivity`

```python
assumption_winsor_sensitivity(
    checks: RegressionChecks,
    winsor_reference_se: Optional[float] = None,
    yellow_sigma: float = 1.0,
    red_sigma: float = 2.0,
    yellow_ratio: float = 0.1,
    red_ratio: float = 0.25,
) -> Dict[str, Any]
```

Check sensitivity of adjusted ATE to winsorized-outcome refit.

###### `cuped_forest_plot`

```python
cuped_forest_plot(
    estimate_with_cuped: CausalEstimate,
    estimate_without_cuped: Optional[CausalEstimate] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.5, 3.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    label_with_cuped: str = "With CUPED",
    label_without_cuped: str = "Without CUPED",
    color_with_cuped: str = "C0",
    color_without_cuped: str = "C1",
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure
```

Forest plot of absolute estimates and CIs for CUPED vs non-CUPED.

**Parameters:**

- **estimate_with_cuped** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Effect estimated with CUPED adjustment.
- **estimate_without_cuped** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Effect estimated without CUPED adjustment. If omitted, the function
  uses `estimate_with_cuped.diagnostic_data.ate_naive` and
  `estimate_with_cuped.diagnostic_data.se_naive` to build a normal-approx CI.

###### `design_matrix_checks`

```python
design_matrix_checks(design: pd.DataFrame) -> tuple[int, int, bool, float]
```

Return rank/conditioning diagnostics for a numeric design matrix.

###### `forest_plot`

**Functions:**

- [**cuped_forest_plot**](#causalis.scenarios.cuped.diagnostics.forest_plot.cuped_forest_plot) – Forest plot of absolute estimates and CIs for CUPED vs non-CUPED.

####### `cuped_forest_plot`

```python
cuped_forest_plot(
    estimate_with_cuped: CausalEstimate,
    estimate_without_cuped: Optional[CausalEstimate] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.5, 3.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    label_with_cuped: str = "With CUPED",
    label_without_cuped: str = "Without CUPED",
    color_with_cuped: str = "C0",
    color_without_cuped: str = "C1",
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure
```

Forest plot of absolute estimates and CIs for CUPED vs non-CUPED.

**Parameters:**

- **estimate_with_cuped** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Effect estimated with CUPED adjustment.
- **estimate_without_cuped** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Effect estimated without CUPED adjustment. If omitted, the function
  uses `estimate_with_cuped.diagnostic_data.ate_naive` and
  `estimate_with_cuped.diagnostic_data.se_naive` to build a normal-approx CI.

###### `overall_assumption_flag`

```python
overall_assumption_flag(table: pd.DataFrame) -> str
```

Return overall GREEN/YELLOW/RED status from an assumptions table.

###### `regression_assumption_rows_from_checks`

```python
regression_assumption_rows_from_checks(
    checks: RegressionChecks,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
    winsor_reference_se: Optional[float] = None,
) -> list[Dict[str, Any]]
```

Run all CUPED regression assumption tests and return row payloads.

###### `regression_assumptions_table_from_checks`

```python
regression_assumptions_table_from_checks(
    checks: RegressionChecks,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
    winsor_reference_se: Optional[float] = None,
) -> pd.DataFrame
```

Return a table of GREEN/YELLOW/RED assumption flags from checks payload.

###### `regression_assumptions_table_from_data`

```python
regression_assumptions_table_from_data(
    data: CausalData,
    covariates: Sequence[str],
    model_kwargs: Optional[Dict[str, Any]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

Fit CUPED on `CausalData` and return the assumptions flag table.

###### `regression_assumptions_table_from_diagnostic_data`

```python
regression_assumptions_table_from_diagnostic_data(
    diagnostic_data: CUPEDDiagnosticData,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
    winsor_reference_se: Optional[float] = None,
) -> pd.DataFrame
```

Build assumption table from `CUPEDDiagnosticData` payload.

###### `regression_assumptions_table_from_estimate`

```python
regression_assumptions_table_from_estimate(
    data_or_estimate: CausalData | CausalEstimate,
    estimate: Optional[CausalEstimate] = None,
    style_regression_assumptions_table: Optional[
        Callable[[pd.DataFrame], Any]
    ] = None,
    cov_type: Optional[str] = None,
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
) -> Any
```

Build assumptions table from a CUPED estimate.

Supports both call styles:

1. `regression_assumptions_table_from_estimate(estimate, ...)`
1. `regression_assumptions_table_from_estimate(data, estimate, ...)`

###### `regression_checks`

**Classes:**

- [**RegressionChecks**](#causalis.scenarios.cuped.diagnostics.regression_checks.RegressionChecks) – Lightweight OLS/regression health checks for CUPED diagnostics.

**Functions:**

- [**assumption_ate_gap**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_ate_gap) – Check adjusted-vs-naive ATE gap relative to naive SE.
- [**assumption_condition_number**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_condition_number) – Check global collinearity via condition number.
- [**assumption_cooks**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_cooks) – Check Cook's distance influence diagnostics.
- [**assumption_design_rank**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_design_rank) – Check that the design matrix is full rank.
- [**assumption_hc23_stability**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_hc23_stability) – Check HC2/HC3 stability when leverage terms approach one.
- [**assumption_leverage**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_leverage) – Check leverage concentration.
- [**assumption_near_duplicates**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_near_duplicates) – Check near-duplicate centered covariate pairs.
- [**assumption_residual_tails**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_residual_tails) – Check residual extremes using max standardized residual only.
- [**assumption_vif**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_vif) – Check VIF from centered main-effect covariates.
- [**assumption_winsor_sensitivity**](#causalis.scenarios.cuped.diagnostics.regression_checks.assumption_winsor_sensitivity) – Check sensitivity of adjusted ATE to winsorized-outcome refit.
- [**design_matrix_checks**](#causalis.scenarios.cuped.diagnostics.regression_checks.design_matrix_checks) – Return rank/conditioning diagnostics for a numeric design matrix.
- [**leverage_and_cooks**](#causalis.scenarios.cuped.diagnostics.regression_checks.leverage_and_cooks) – Compute leverage, Cook's distance, and internally studentized residuals.
- [**near_duplicate_corr_pairs**](#causalis.scenarios.cuped.diagnostics.regression_checks.near_duplicate_corr_pairs) – Find pairs with absolute correlation very close to one.
- [**overall_assumption_flag**](#causalis.scenarios.cuped.diagnostics.regression_checks.overall_assumption_flag) – Return overall GREEN/YELLOW/RED status from an assumptions table.
- [**regression_assumption_rows_from_checks**](#causalis.scenarios.cuped.diagnostics.regression_checks.regression_assumption_rows_from_checks) – Run all CUPED regression assumption tests and return row payloads.
- [**regression_assumptions_table_from_checks**](#causalis.scenarios.cuped.diagnostics.regression_checks.regression_assumptions_table_from_checks) – Return a table of GREEN/YELLOW/RED assumption flags from checks payload.
- [**regression_assumptions_table_from_data**](#causalis.scenarios.cuped.diagnostics.regression_checks.regression_assumptions_table_from_data) – Fit CUPED on `CausalData` and return the assumptions flag table.
- [**regression_assumptions_table_from_diagnostic_data**](#causalis.scenarios.cuped.diagnostics.regression_checks.regression_assumptions_table_from_diagnostic_data) – Build assumption table from `CUPEDDiagnosticData` payload.
- [**regression_assumptions_table_from_estimate**](#causalis.scenarios.cuped.diagnostics.regression_checks.regression_assumptions_table_from_estimate) – Build assumptions table from a CUPED estimate.
- [**run_regression_checks**](#causalis.scenarios.cuped.diagnostics.regression_checks.run_regression_checks) – Build a compact payload with design, residual, and influence diagnostics.
- [**style_regression_assumptions_table**](#causalis.scenarios.cuped.diagnostics.regression_checks.style_regression_assumptions_table) – Return pandas Styler with colored flag cells for notebook display.
- [**vif_from_corr**](#causalis.scenarios.cuped.diagnostics.regression_checks.vif_from_corr) – Approximate VIF from inverse correlation matrix of standardized covariates..
- [**winsor_fit_tau**](#causalis.scenarios.cuped.diagnostics.regression_checks.winsor_fit_tau) – Refit OLS on winsorized outcome and return treatment coefficient.

####### `FLAG_COLOR`

```python
FLAG_COLOR = {
    FLAG_GREEN: "#2e7d32",
    FLAG_YELLOW: "#f9a825",
    FLAG_RED: "#c62828",
}

```

####### `FLAG_GREEN`

```python
FLAG_GREEN = 'GREEN'
```

####### `FLAG_LEVEL`

```python
FLAG_LEVEL = {FLAG_GREEN: 0, FLAG_YELLOW: 1, FLAG_RED: 2}
```

####### `FLAG_RED`

```python
FLAG_RED = 'RED'
```

####### `FLAG_YELLOW`

```python
FLAG_YELLOW = 'YELLOW'
```

####### `RegressionChecks`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Lightweight OLS/regression health checks for CUPED diagnostics.

######## `ate_adj`

```python
ate_adj: float
```

######## `ate_adj_winsor`

```python
ate_adj_winsor: Optional[float] = None
```

######## `ate_adj_winsor_gap`

```python
ate_adj_winsor_gap: Optional[float] = None
```

######## `ate_gap`

```python
ate_gap: float
```

######## `ate_gap_over_se_naive`

```python
ate_gap_over_se_naive: Optional[float] = None
```

######## `ate_naive`

```python
ate_naive: float
```

######## `condition_number`

```python
condition_number: float
```

######## `cooks_cutoff`

```python
cooks_cutoff: float
```

######## `full_rank`

```python
full_rank: bool
```

######## `k`

```python
k: int
```

######## `leverage_cutoff`

```python
leverage_cutoff: float
```

######## `max_abs_std_resid`

```python
max_abs_std_resid: float
```

######## `max_cooks`

```python
max_cooks: float
```

######## `max_leverage`

```python
max_leverage: float
```

######## `min_one_minus_h`

```python
min_one_minus_h: float
```

######## `n_high_cooks`

```python
n_high_cooks: int
```

######## `n_high_leverage`

```python
n_high_leverage: int
```

######## `n_std_resid_gt_3`

```python
n_std_resid_gt_3: int
```

######## `n_std_resid_gt_4`

```python
n_std_resid_gt_4: int
```

######## `n_tiny_one_minus_h`

```python
n_tiny_one_minus_h: int
```

######## `near_duplicate_pairs`

```python
near_duplicate_pairs: list[tuple[str, str, float]] = Field(default_factory=list)

```

######## `p_main_covariates`

```python
p_main_covariates: int
```

######## `rank`

```python
rank: int
```

######## `resid_scale_mad`

```python
resid_scale_mad: float
```

######## `vif`

```python
vif: Optional[Dict[str, float]] = None
```

######## `winsor_q`

```python
winsor_q: Optional[float] = None
```

####### `assumption_ate_gap`

```python
assumption_ate_gap(
    checks: RegressionChecks,
    yellow_threshold: float = 2.0,
    red_threshold: float = 2.5,
) -> Dict[str, Any]
```

Check adjusted-vs-naive ATE gap relative to naive SE.

####### `assumption_condition_number`

```python
assumption_condition_number(
    checks: RegressionChecks,
    warn_threshold: float = 100000000.0,
    red_multiplier: float = 100.0,
) -> Dict[str, Any]
```

Check global collinearity via condition number.

####### `assumption_cooks`

```python
assumption_cooks(
    checks: RegressionChecks,
    yellow_threshold: float = 0.1,
    red_threshold: float = 1.0,
) -> Dict[str, Any]
```

Check Cook's distance influence diagnostics.

####### `assumption_design_rank`

```python
assumption_design_rank(checks: RegressionChecks) -> Dict[str, Any]
```

Check that the design matrix is full rank.

####### `assumption_hc23_stability`

```python
assumption_hc23_stability(
    checks: RegressionChecks, cov_type: str, tiny_one_minus_h_tol: float = 1e-08
) -> Dict[str, Any]
```

Check HC2/HC3 stability when leverage terms approach one.

####### `assumption_leverage`

```python
assumption_leverage(
    checks: RegressionChecks,
    yellow_multiplier: float = 5.0,
    red_multiplier: float = 10.0,
    red_floor: float = 0.5,
) -> Dict[str, Any]
```

Check leverage concentration.

####### `assumption_near_duplicates`

```python
assumption_near_duplicates(
    checks: RegressionChecks, red_pairs_threshold: int = 3
) -> Dict[str, Any]
```

Check near-duplicate centered covariate pairs.

####### `assumption_residual_tails`

```python
assumption_residual_tails(
    checks: RegressionChecks,
    yellow_abs_std_resid: float = 7.0,
    red_abs_std_resid: float = 10.0,
) -> Dict[str, Any]
```

Check residual extremes using max standardized residual only.

####### `assumption_vif`

```python
assumption_vif(
    checks: RegressionChecks,
    warn_threshold: float = 20.0,
    red_multiplier: float = 2.0,
) -> Dict[str, Any]
```

Check VIF from centered main-effect covariates.

####### `assumption_winsor_sensitivity`

```python
assumption_winsor_sensitivity(
    checks: RegressionChecks,
    winsor_reference_se: Optional[float] = None,
    yellow_sigma: float = 1.0,
    red_sigma: float = 2.0,
    yellow_ratio: float = 0.1,
    red_ratio: float = 0.25,
) -> Dict[str, Any]
```

Check sensitivity of adjusted ATE to winsorized-outcome refit.

####### `design_matrix_checks`

```python
design_matrix_checks(design: pd.DataFrame) -> tuple[int, int, bool, float]
```

Return rank/conditioning diagnostics for a numeric design matrix.

####### `leverage_and_cooks`

```python
leverage_and_cooks(
    y: np.ndarray, z: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Compute leverage, Cook's distance, and internally studentized residuals.

####### `near_duplicate_corr_pairs`

```python
near_duplicate_corr_pairs(
    x: pd.DataFrame, tol: float, max_pairs: int = 50
) -> list[tuple[str, str, float]]
```

Find pairs with absolute correlation very close to one.

####### `overall_assumption_flag`

```python
overall_assumption_flag(table: pd.DataFrame) -> str
```

Return overall GREEN/YELLOW/RED status from an assumptions table.

####### `regression_assumption_rows_from_checks`

```python
regression_assumption_rows_from_checks(
    checks: RegressionChecks,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
    winsor_reference_se: Optional[float] = None,
) -> list[Dict[str, Any]]
```

Run all CUPED regression assumption tests and return row payloads.

####### `regression_assumptions_table_from_checks`

```python
regression_assumptions_table_from_checks(
    checks: RegressionChecks,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
    winsor_reference_se: Optional[float] = None,
) -> pd.DataFrame
```

Return a table of GREEN/YELLOW/RED assumption flags from checks payload.

####### `regression_assumptions_table_from_data`

```python
regression_assumptions_table_from_data(
    data: CausalData,
    covariates: Sequence[str],
    model_kwargs: Optional[Dict[str, Any]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

Fit CUPED on `CausalData` and return the assumptions flag table.

####### `regression_assumptions_table_from_diagnostic_data`

```python
regression_assumptions_table_from_diagnostic_data(
    diagnostic_data: CUPEDDiagnosticData,
    cov_type: str = "HC2",
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
    winsor_reference_se: Optional[float] = None,
) -> pd.DataFrame
```

Build assumption table from `CUPEDDiagnosticData` payload.

####### `regression_assumptions_table_from_estimate`

```python
regression_assumptions_table_from_estimate(
    data_or_estimate: CausalData | CausalEstimate,
    estimate: Optional[CausalEstimate] = None,
    style_regression_assumptions_table: Optional[
        Callable[[pd.DataFrame], Any]
    ] = None,
    cov_type: Optional[str] = None,
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
) -> Any
```

Build assumptions table from a CUPED estimate.

Supports both call styles:

1. `regression_assumptions_table_from_estimate(estimate, ...)`
1. `regression_assumptions_table_from_estimate(data, estimate, ...)`

####### `run_regression_checks`

```python
run_regression_checks(
    y: pd.Series,
    design: pd.DataFrame,
    result: Any,
    result_naive: Any,
    cov_type: str,
    use_t_fit: bool,
    corr_near_one_tol: float,
    tiny_one_minus_h_tol: float,
    winsor_q: Optional[float],
) -> RegressionChecks
```

Build a compact payload with design, residual, and influence diagnostics.

####### `style_regression_assumptions_table`

```python
style_regression_assumptions_table(table: pd.DataFrame)
```

Return pandas Styler with colored flag cells for notebook display.

####### `vif_from_corr`

```python
vif_from_corr(x: pd.DataFrame) -> Optional[Dict[str, float]]
```

Approximate VIF from inverse correlation matrix of standardized covariates..

####### `winsor_fit_tau`

```python
winsor_fit_tau(
    y: pd.Series,
    design: pd.DataFrame,
    cov_type: str,
    use_t_fit: bool,
    winsor_q: Optional[float],
) -> Optional[float]
```

Refit OLS on winsorized outcome and return treatment coefficient.

###### `run_regression_checks`

```python
run_regression_checks(
    y: pd.Series,
    design: pd.DataFrame,
    result: Any,
    result_naive: Any,
    cov_type: str,
    use_t_fit: bool,
    corr_near_one_tol: float,
    tiny_one_minus_h_tol: float,
    winsor_q: Optional[float],
) -> RegressionChecks
```

Build a compact payload with design, residual, and influence diagnostics.

###### `style_regression_assumptions_table`

```python
style_regression_assumptions_table(table: pd.DataFrame)
```

Return pandas Styler with colored flag cells for notebook display.

##### `model`

**Classes:**

- [**CUPEDModel**](#causalis.scenarios.cuped.model.CUPEDModel) – CUPED-style regression adjustment estimator for ATE/ITT in randomized experiments.

###### `CUPEDModel`

```python
CUPEDModel(
    cov_type: str = "HC2",
    alpha: float = 0.05,
    strict_binary_treatment: bool = True,
    use_t: Optional[bool] = None,
    use_t_auto_n_threshold: int = 5000,
    relative_ci_method: Literal["delta_nocov", "bootstrap"] = "delta_nocov",
    relative_ci_bootstrap_draws: int = 1000,
    relative_ci_bootstrap_seed: Optional[int] = None,
    covariate_variance_min: float = 1e-12,
    condition_number_warn_threshold: float = 100000000.0,
    run_regression_checks: bool = True,
    check_action: Literal["ignore", "raise"] = "ignore",
    raise_on_yellow: bool = False,
    corr_near_one_tol: float = 1e-10,
    vif_warn_threshold: float = 20.0,
    winsor_q: Optional[float] = 0.01,
    tiny_one_minus_h_tol: float = 1e-08,
) -> None
```

CUPED-style regression adjustment estimator for ATE/ITT in randomized experiments.

Fits an outcome regression with pre-treatment covariates (always centered
over the full sample, never within treatment groups)
implemented as Lin (2013) fully interacted OLS:

```
Y ~ 1 + D + X^c + D * X^c
```

The reported effect is the coefficient on D, with robust covariance as requested.
This specification ensures the coefficient on D is the ATE/ITT even if the
treatment effect is heterogeneous with respect to covariates.
This is broader than canonical single-theta CUPED (`Y - theta*(X - mean(X))`).

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
  Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
- **alpha** (<code>[float](#float)</code>) – Significance level for confidence intervals.
- **strict_binary_treatment** (<code>[bool](#bool)</code>) – If True, require treatment to be binary {0,1}.
- **use_t** (<code>[bool](#bool) | None</code>) – If bool, passed to statsmodels `.fit(..., use_t=use_t)` directly.
  If None, automatic policy is used: for robust HC\* covariances,
  `use_t=True` when `n < use_t_auto_n_threshold`, else `False`.
  For non-robust covariance, `use_t=True`.
- **use_t_auto_n_threshold** (<code>[int](#int)</code>) – Sample-size threshold for automatic `use_t` selection when `use_t=None`
  and covariance is HC\* robust.
- **relative_ci_method** (<code>('delta_nocov', 'bootstrap')</code>) – Method for relative CI of `100 * tau / mu_c`.
- "delta_nocov": delta method using robust `Var(tau)` and `Var(mu_c)` while
  setting `Cov(tau, mu_c)=0` (safe fallback without unsupported hybrid IF covariance).
- "bootstrap": percentile bootstrap CI on the relative effect.
- **relative_ci_bootstrap_draws** (<code>[int](#int)</code>) – Number of bootstrap resamples used when `relative_ci_method="bootstrap"`.
- **relative_ci_bootstrap_seed** (<code>[int](#int) | None</code>) – RNG seed used for bootstrap relative CI.
- **covariate_variance_min** (<code>[float](#float)</code>) – Minimum variance threshold for retaining a CUPED covariate. Covariates with
  variance less than or equal to this threshold are dropped before fitting.
- **condition_number_warn_threshold** (<code>[float](#float)</code>) – Trigger diagnostics signal when the design matrix condition number exceeds this threshold.
- **run_regression_checks** (<code>[bool](#bool)</code>) – Whether to compute regression diagnostics payload during `fit()`.
- **check_action** (<code>('ignore', 'raise')</code>) – Action used when a diagnostics threshold is violated.
- **raise_on_yellow** (<code>[bool](#bool)</code>) – When `check_action="raise"`, also raise on YELLOW assumption flags.
- **corr_near_one_tol** (<code>[float](#float)</code>) – Correlation tolerance used to mark near-duplicate centered covariates.
- **vif_warn_threshold** (<code>[float](#float)</code>) – VIF threshold that triggers a diagnostics signal.
- **winsor_q** (<code>[float](#float) | None</code>) – Quantile used for winsor sensitivity refit. Set `None` to disable.
- **tiny_one_minus_h_tol** (<code>[float](#float)</code>) – Threshold for flagging near-degenerate `1 - leverage` terms in HC2/HC3.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
- Covariates are globally centered over the full sample only. This centering
  convention is required so the treatment coefficient in the Lin specification
  remains the ATE/ITT.
- The Lin (2013) specification is recommended as a robust regression-adjustment default
  in RCTs.

</details>

**Functions:**

- [**assumptions_table**](#causalis.scenarios.cuped.model.CUPEDModel.assumptions_table) – Return fitted regression assumptions table (GREEN/YELLOW/RED) when available.
- [**estimate**](#causalis.scenarios.cuped.model.CUPEDModel.estimate) – Return the adjusted ATE/ITT estimate and inference.
- [**fit**](#causalis.scenarios.cuped.model.CUPEDModel.fit) – Fit CUPED-style regression adjustment (Lin-interacted OLS) on a CausalData object.
- [**summary_dict**](#causalis.scenarios.cuped.model.CUPEDModel.summary_dict) – Convenience JSON/logging output.

####### `adjustment`

```python
adjustment: Literal['lin'] = 'lin'
```

####### `alpha`

```python
alpha = float(alpha)
```

####### `assumptions_table`

```python
assumptions_table() -> Optional[pd.DataFrame]
```

Return fitted regression assumptions table (GREEN/YELLOW/RED) when available.

####### `center_covariates`

```python
center_covariates = True
```

####### `centering_scope`

```python
centering_scope: Literal['global'] = 'global'
```

####### `check_action`

```python
check_action: Literal['ignore', 'raise'] = check_action
```

####### `condition_number_warn_threshold`

```python
condition_number_warn_threshold = float(condition_number_warn_threshold)
```

####### `corr_near_one_tol`

```python
corr_near_one_tol = float(corr_near_one_tol)
```

####### `cov_type`

```python
cov_type = str(cov_type)
```

####### `covariate_variance_min`

```python
covariate_variance_min = float(covariate_variance_min)
```

####### `estimate`

```python
estimate(
    alpha: Optional[float] = None, diagnostic_data: bool = True
) -> CausalEstimate
```

Return the adjusted ATE/ITT estimate and inference.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

####### `fit`

```python
fit(
    data: CausalData,
    covariates: Optional[Sequence[str]] = None,
    run_checks: Optional[bool] = None,
) -> CUPEDModel
```

Fit CUPED-style regression adjustment (Lin-interacted OLS) on a CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
- **covariates** (<code>([Sequence](#typing.Sequence)\[[str](#str)\], [required](#required))</code>) – Explicit subset of `data_contracts.confounders_names` to use as CUPED covariates.
  Pass `[]` for an unadjusted (naive) fit.
- **run_checks** (<code>[bool](#bool) | None</code>) – Override whether regression checks are computed in this fit call.
  If `None`, uses `self.run_regression_checks`.

**Returns:**

- <code>[CUPEDModel](#causalis.scenarios.cuped.model.CUPEDModel)</code> – Fitted estimator.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If `covariates` is omitted, not a sequence of strings, contains columns missing from the
  DataFrame, contains columns outside `data_contracts.confounders_names`,
  treatment is not binary when `strict_binary_treatment=True`,
  or the design matrix is rank deficient.

####### `raise_on_yellow`

```python
raise_on_yellow = bool(raise_on_yellow)
```

####### `relative_ci_bootstrap_draws`

```python
relative_ci_bootstrap_draws = int(relative_ci_bootstrap_draws)
```

####### `relative_ci_bootstrap_seed`

```python
relative_ci_bootstrap_seed = relative_ci_bootstrap_seed
```

####### `relative_ci_method`

```python
relative_ci_method: Literal['delta_nocov', 'bootstrap'] = relative_ci_method
```

####### `run_regression_checks`

```python
run_regression_checks = bool(run_regression_checks)
```

####### `strict_binary_treatment`

```python
strict_binary_treatment = bool(strict_binary_treatment)
```

####### `summary_dict`

```python
summary_dict(alpha: Optional[float] = None) -> Dict[str, Any]
```

Convenience JSON/logging output.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with estimates, inference, and diagnostics.

####### `tiny_one_minus_h_tol`

```python
tiny_one_minus_h_tol = float(tiny_one_minus_h_tol)
```

####### `use_t`

```python
use_t = None if use_t is None else bool(use_t)
```

####### `use_t_auto_n_threshold`

```python
use_t_auto_n_threshold = int(use_t_auto_n_threshold)
```

####### `vif_warn_threshold`

```python
vif_warn_threshold = float(vif_warn_threshold)
```

####### `winsor_q`

```python
winsor_q = None
```

##### `regression_assumptions_table_from_data`

```python
regression_assumptions_table_from_data(
    data: CausalData,
    covariates: Sequence[str],
    model_kwargs: Optional[Dict[str, Any]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

Fit CUPED on `CausalData` and return the assumptions flag table.

##### `regression_assumptions_table_from_estimate`

```python
regression_assumptions_table_from_estimate(
    data_or_estimate: CausalData | CausalEstimate,
    estimate: Optional[CausalEstimate] = None,
    style_regression_assumptions_table: Optional[
        Callable[[pd.DataFrame], Any]
    ] = None,
    cov_type: Optional[str] = None,
    condition_number_warn_threshold: float = 100000000.0,
    vif_warn_threshold: float = 20.0,
    tiny_one_minus_h_tol: float = 1e-08,
) -> Any
```

Build assumptions table from a CUPED estimate.

Supports both call styles:

1. `regression_assumptions_table_from_estimate(estimate, ...)`
1. `regression_assumptions_table_from_estimate(data, estimate, ...)`

##### `style_regression_assumptions_table`

```python
style_regression_assumptions_table(table: pd.DataFrame)
```

Return pandas Styler with colored flag cells for notebook display.

#### `gate`

Group Average Treatment Effect (GATE) estimation utilities.

**Modules:**

- [**model**](#causalis.scenarios.gate.model) –

**Functions:**

- [**estimate_gate_from_irm**](#causalis.scenarios.gate.estimate_gate_from_irm) – Estimate strict GATEs from a fitted IRM via groupwise closed-form robust inference.

##### `estimate_gate_from_irm`

```python
estimate_gate_from_irm(
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series],
    alpha: float = 0.05,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
    diagnostic_data: bool = True,
) -> GateEstimate
```

Estimate strict GATEs from a fitted IRM via groupwise closed-form robust inference.

Groups are assumed to be pre-specified, pre-treatment, mutually exclusive, and exhaustive.

##### `model`

**Functions:**

- [**estimate_gate_from_irm**](#causalis.scenarios.gate.model.estimate_gate_from_irm) – Estimate strict GATEs from a fitted IRM via groupwise closed-form robust inference.

###### `estimate_gate_from_irm`

```python
estimate_gate_from_irm(
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series],
    alpha: float = 0.05,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
    diagnostic_data: bool = True,
) -> GateEstimate
```

Estimate strict GATEs from a fitted IRM via groupwise closed-form robust inference.

Groups are assumed to be pre-specified, pre-treatment, mutually exclusive, and exhaustive.

#### `multi_unconfoundedness`

**Modules:**

- [**dgp**](#causalis.scenarios.multi_unconfoundedness.dgp) –
- [**model**](#causalis.scenarios.multi_unconfoundedness.model) –
- [**refutation**](#causalis.scenarios.multi_unconfoundedness.refutation) – Refutation utilities for multi-treatment unconfoundedness.

**Classes:**

- [**MultiTreatmentIRM**](#causalis.scenarios.multi_unconfoundedness.MultiTreatmentIRM) – Interactive Regression Model for multi-treatment unconfoundedness.

##### `MultiTreatmentIRM`

```python
MultiTreatmentIRM(
    data: Optional[MultiCausalData] = None,
    ml_g: Any = None,
    ml_m: Any = None,
    *,
    n_folds: int = 5,
    n_rep: int = 1,
    normalize_ipw: bool = False,
    trimming_rule: str = "truncate",
    trimming_threshold: float = 0.01,
    random_state: Optional[int] = None,
    n_jobs: int = 1
)
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model for multi-treatment unconfoundedness.

DoubleML-style cross-fitting estimator consuming `MultiCausalData` and
producing pairwise ATE contrasts against baseline treatment (column 0).
Model supports >= 2 treatments.

<details class="----parameters" open markdown="1">
<summary>    Parameters</summary>

```
data : MultiCausalData
    Data container with outcome, one-hot multi-treatment indicators, and confounders.
ml_g : estimator
    Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
ml_m : classifier
    Learner for E[D|X] (generelized propensity score). Must support predict_proba() or predict() in (0,1).
n_folds : int, default 5
    Number of cross-fitting folds.
n_rep : int, default 1
    Number of repetitions of sample splitting. Currently only 1 is supported.
normalize_ipw : bool, default False
    Whether to normalize IPW terms within the score.
    This is a Hajek-style stabilization: it can reduce variance but may add
    small finite-sample bias.
trimming_rule : {"truncate"}, default "truncate"
    Trimming approach for propensity scores.
trimming_threshold : float, default 1e-2
    Threshold for trimming if rule is "truncate".
random_state : Optional[int], default None
    Random seed for fold creation.
n_jobs : int, default 1
    Number of parallel jobs for fold-level cross-fitting.
    Use `-1` to use all available CPUs.
    Practical guidance:
    - Start with `n_jobs=1` for stable, low-contention defaults.
    - Increase to `n_jobs=2/4/-1` when cross-fitting is the bottleneck.
    - If nuisance learners are already multithreaded (e.g. CatBoost with
      `thread_count=-1`), keep `n_jobs=1` or set learner threads to `1`
      to avoid CPU oversubscription.
```

</details>

**Functions:**

- [**confint**](#causalis.scenarios.multi_unconfoundedness.MultiTreatmentIRM.confint) –
- [**estimate**](#causalis.scenarios.multi_unconfoundedness.MultiTreatmentIRM.estimate) –
- [**fit**](#causalis.scenarios.multi_unconfoundedness.MultiTreatmentIRM.fit) –
- [**sensitivity_analysis**](#causalis.scenarios.multi_unconfoundedness.MultiTreatmentIRM.sensitivity_analysis) –

###### `coef`

```python
coef: np.ndarray
```

Return the estimated coefficient.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The estimated coefficient.

###### `confint`

```python
confint() -> pd.DataFrame
```

###### `data`

```python
data = data
```

###### `diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary containing 'm_hat', 'g_hat' and 'folds'.

###### `estimate`

```python
estimate(
    score: str = "ATE", alpha: float = 0.05, diagnostic_data: bool = True
) -> MultiCausalEstimate
```

###### `fit`

```python
fit(data: Optional[MultiCausalData] = None) -> 'MultiTreatmentIRM'
```

###### `ml_g`

```python
ml_g = ml_g
```

###### `ml_m`

```python
ml_m = ml_m
```

###### `n_folds`

```python
n_folds = int(n_folds)
```

###### `n_jobs`

```python
n_jobs = int(n_jobs)
```

###### `n_rep`

```python
n_rep = int(n_rep)
```

###### `normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

###### `orth_signal`

```python
orth_signal: np.ndarray
```

Return the cross-fitted orthogonal signal (psi_b).

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The orthogonal signal.

###### `pvalues`

```python
pvalues: np.ndarray
```

Return the p-values for the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The p-values.

###### `random_state`

```python
random_state = random_state
```

###### `score`

```python
score = 'ATE'
```

###### `se`

```python
se: np.ndarray
```

Return the standard error of the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The standard error.

###### `sensitivity_analysis`

```python
sensitivity_analysis(
    cf_y: Optional[float] = None,
    r2_d: Any = 0.0,
    rho: Any = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    *,
    r2_y: Optional[float] = None
) -> "MultiTreatmentIRM"
```

###### `summary`

```python
summary: pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The results summary.

###### `trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

###### `trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

##### `dgp`

**Functions:**

- [**generate_multitreatment_binary_26**](#causalis.scenarios.multi_unconfoundedness.dgp.generate_multitreatment_binary_26) – Pre-configured multi-treatment dataset with Binary outcome.
- [**generate_multitreatment_gamma_26**](#causalis.scenarios.multi_unconfoundedness.dgp.generate_multitreatment_gamma_26) – Pre-configured multi-treatment dataset with Gamma-distributed outcome.
- [**generate_multitreatment_irm_26**](#causalis.scenarios.multi_unconfoundedness.dgp.generate_multitreatment_irm_26) –

###### `generate_multitreatment_binary_26`

```python
generate_multitreatment_binary_26(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]
```

Pre-configured multi-treatment dataset with Binary outcome.

- 3 treatment classes: control + 2 treatments
- 8 confounders with realistic marginals
- Binary outcome with logistic-link linear confounding
- Heterogeneous treatment effects and correlated confounders via Gaussian copula

###### `generate_multitreatment_gamma_26`

```python
generate_multitreatment_gamma_26(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]
```

Pre-configured multi-treatment dataset with Gamma-distributed outcome.

- 3 treatment classes: control + 2 treatments
- 8 confounders with realistic marginals
- Gamma outcome with log-link linear confounding
- Heterogeneous treatment effects and correlated confounders via Gaussian copula

###### `generate_multitreatment_irm_26`

```python
generate_multitreatment_irm_26(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]
```

##### `model`

**Classes:**

- [**MultiTreatmentIRM**](#causalis.scenarios.multi_unconfoundedness.model.MultiTreatmentIRM) – Interactive Regression Model for multi-treatment unconfoundedness.

###### `HAS_CATBOOST`

```python
HAS_CATBOOST = True
```

###### `MultiTreatmentIRM`

```python
MultiTreatmentIRM(
    data: Optional[MultiCausalData] = None,
    ml_g: Any = None,
    ml_m: Any = None,
    *,
    n_folds: int = 5,
    n_rep: int = 1,
    normalize_ipw: bool = False,
    trimming_rule: str = "truncate",
    trimming_threshold: float = 0.01,
    random_state: Optional[int] = None,
    n_jobs: int = 1
)
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model for multi-treatment unconfoundedness.

DoubleML-style cross-fitting estimator consuming `MultiCausalData` and
producing pairwise ATE contrasts against baseline treatment (column 0).
Model supports >= 2 treatments.

<details class="----parameters" open markdown="1">
<summary>    Parameters</summary>

```
data : MultiCausalData
    Data container with outcome, one-hot multi-treatment indicators, and confounders.
ml_g : estimator
    Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
ml_m : classifier
    Learner for E[D|X] (generelized propensity score). Must support predict_proba() or predict() in (0,1).
n_folds : int, default 5
    Number of cross-fitting folds.
n_rep : int, default 1
    Number of repetitions of sample splitting. Currently only 1 is supported.
normalize_ipw : bool, default False
    Whether to normalize IPW terms within the score.
    This is a Hajek-style stabilization: it can reduce variance but may add
    small finite-sample bias.
trimming_rule : {"truncate"}, default "truncate"
    Trimming approach for propensity scores.
trimming_threshold : float, default 1e-2
    Threshold for trimming if rule is "truncate".
random_state : Optional[int], default None
    Random seed for fold creation.
n_jobs : int, default 1
    Number of parallel jobs for fold-level cross-fitting.
    Use `-1` to use all available CPUs.
    Practical guidance:
    - Start with `n_jobs=1` for stable, low-contention defaults.
    - Increase to `n_jobs=2/4/-1` when cross-fitting is the bottleneck.
    - If nuisance learners are already multithreaded (e.g. CatBoost with
      `thread_count=-1`), keep `n_jobs=1` or set learner threads to `1`
      to avoid CPU oversubscription.
```

</details>

**Functions:**

- [**confint**](#causalis.scenarios.multi_unconfoundedness.model.MultiTreatmentIRM.confint) –
- [**estimate**](#causalis.scenarios.multi_unconfoundedness.model.MultiTreatmentIRM.estimate) –
- [**fit**](#causalis.scenarios.multi_unconfoundedness.model.MultiTreatmentIRM.fit) –
- [**sensitivity_analysis**](#causalis.scenarios.multi_unconfoundedness.model.MultiTreatmentIRM.sensitivity_analysis) –

####### `coef`

```python
coef: np.ndarray
```

Return the estimated coefficient.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The estimated coefficient.

####### `confint`

```python
confint() -> pd.DataFrame
```

####### `data`

```python
data = data
```

####### `diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary containing 'm_hat', 'g_hat' and 'folds'.

####### `estimate`

```python
estimate(
    score: str = "ATE", alpha: float = 0.05, diagnostic_data: bool = True
) -> MultiCausalEstimate
```

####### `fit`

```python
fit(data: Optional[MultiCausalData] = None) -> 'MultiTreatmentIRM'
```

####### `ml_g`

```python
ml_g = ml_g
```

####### `ml_m`

```python
ml_m = ml_m
```

####### `n_folds`

```python
n_folds = int(n_folds)
```

####### `n_jobs`

```python
n_jobs = int(n_jobs)
```

####### `n_rep`

```python
n_rep = int(n_rep)
```

####### `normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

####### `orth_signal`

```python
orth_signal: np.ndarray
```

Return the cross-fitted orthogonal signal (psi_b).

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The orthogonal signal.

####### `pvalues`

```python
pvalues: np.ndarray
```

Return the p-values for the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The p-values.

####### `random_state`

```python
random_state = random_state
```

####### `score`

```python
score = 'ATE'
```

####### `se`

```python
se: np.ndarray
```

Return the standard error of the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The standard error.

####### `sensitivity_analysis`

```python
sensitivity_analysis(
    cf_y: Optional[float] = None,
    r2_d: Any = 0.0,
    rho: Any = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    *,
    r2_y: Optional[float] = None
) -> "MultiTreatmentIRM"
```

####### `summary`

```python
summary: pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The results summary.

####### `trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

####### `trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

##### `refutation`

Refutation utilities for multi-treatment unconfoundedness.

**Modules:**

- [**overlap**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap) –
- [**overlap_plot**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap_plot) –
- [**score**](#causalis.scenarios.multi_unconfoundedness.refutation.score) –
- [**unconfoundedness**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness) –

**Functions:**

- [**compute_bias_aware_ci**](#causalis.scenarios.multi_unconfoundedness.refutation.compute_bias_aware_ci) –
- [**get_sensitivity_summary**](#causalis.scenarios.multi_unconfoundedness.refutation.get_sensitivity_summary) –
- [**plot_m_overlap**](#causalis.scenarios.multi_unconfoundedness.refutation.plot_m_overlap) – Multi-treatment overlap plot for pairwise conditional propensity scores.
- [**plot_residual_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.plot_residual_diagnostics) – Plot residual diagnostics for multi-treatment nuisance models.
- [**run_overlap_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.run_overlap_diagnostics) – Run multi-treatment overlap diagnostics from data and estimate.
- [**run_score_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.run_score_diagnostics) – Run score diagnostics for multi-treatment baseline contrasts.
- [**run_unconfoundedness_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.run_unconfoundedness_diagnostics) – Run multi-treatment unconfoundedness diagnostics from data and estimate.
- [**sensitivity_analysis**](#causalis.scenarios.multi_unconfoundedness.refutation.sensitivity_analysis) –
- [**sensitivity_benchmark**](#causalis.scenarios.multi_unconfoundedness.refutation.sensitivity_benchmark) –
- [**validate_unconfoundedness_balance**](#causalis.scenarios.multi_unconfoundedness.refutation.validate_unconfoundedness_balance) – Convenience wrapper returning the balance block only.

###### `compute_bias_aware_ci`

```python
compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    cf_y: float = 0.0,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]
```

###### `get_sensitivity_summary`

```python
get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    label: Optional[str] = None,
) -> Optional[str]
```

###### `overlap`

**Modules:**

- [**overlap_plot**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap.overlap_plot) –
- [**overlap_validation**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap.overlap_validation) – Overlap diagnostics for multi-treatment unconfoundedness.

**Functions:**

- [**plot_m_overlap**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap.plot_m_overlap) – Multi-treatment overlap plot for pairwise conditional propensity scores.
- [**run_overlap_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap.run_overlap_diagnostics) – Run multi-treatment overlap diagnostics from data and estimate.

####### `overlap_plot`

**Functions:**

- [**overlap_plot**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap.overlap_plot.overlap_plot) – Convenience wrapper to match `overlap_plot(data, estimate)` API style.
- [**plot_m_overlap**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap.overlap_plot.plot_m_overlap) – Multi-treatment overlap plot for pairwise conditional propensity scores.

######## `overlap_plot`

```python
overlap_plot(
    data: MultiCausalData, estimate: MultiCausalEstimate, **kwargs: Any
) -> plt.Figure
```

Convenience wrapper to match `overlap_plot(data, estimate)` API style.

######## `plot_m_overlap`

```python
plot_m_overlap(
    diag: Union[
        MultiUnconfoundednessDiagnosticData, MultiCausalEstimate, dict, Any
    ],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
    *,
    treatment_idx: Optional[Union[int, List[int]]] = None,
    baseline_idx: int = 0,
    treatment_names: Optional[List[str]] = None
) -> plt.Figure
```

Multi-treatment overlap plot for pairwise conditional propensity scores.

For each comparison baseline (default 0) vs k, this plots
`P(D=k | X, D in {baseline, k}) = m_k(X) / (m_baseline(X) + m_k(X))`
on the observed pair sample `D in {baseline, k}`, comparing:

- units with D=k (treated for the pair),
- units with D=baseline (control for the pair).

Parameters:

- diag.d: (n, K) one-hot
- diag.m_hat / diag.m_hat_raw: (n, K) propensity
- treatment_idx:
  - None -> plot all k != baseline_idx (multi-panel)
  - int -> plot one comparison
  - list[int] -> plot selected comparisons
- ax: supported only for a single comparison (exactly one k)

Returns matplotlib.figure.Figure.

####### `overlap_validation`

Overlap diagnostics for multi-treatment unconfoundedness.

**Functions:**

- [**run_overlap_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap.overlap_validation.run_overlap_diagnostics) – Run multi-treatment overlap diagnostics from data and estimate.

######## `run_overlap_diagnostics`

```python
run_overlap_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05
) -> Dict[str, Any]
```

Run multi-treatment overlap diagnostics from data and estimate.

Diagnostics are computed pairwise between baseline treatment 0 and each
active treatment k using pairwise conditional propensity
P(D=k | X, D in {0, k}) as the comparison score.

####### `plot_m_overlap`

```python
plot_m_overlap(
    diag: Union[
        MultiUnconfoundednessDiagnosticData, MultiCausalEstimate, dict, Any
    ],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
    *,
    treatment_idx: Optional[Union[int, List[int]]] = None,
    baseline_idx: int = 0,
    treatment_names: Optional[List[str]] = None
) -> plt.Figure
```

Multi-treatment overlap plot for pairwise conditional propensity scores.

For each comparison baseline (default 0) vs k, this plots
`P(D=k | X, D in {baseline, k}) = m_k(X) / (m_baseline(X) + m_k(X))`
on the observed pair sample `D in {baseline, k}`, comparing:

- units with D=k (treated for the pair),
- units with D=baseline (control for the pair).

Parameters:

- diag.d: (n, K) one-hot
- diag.m_hat / diag.m_hat_raw: (n, K) propensity
- treatment_idx:
  - None -> plot all k != baseline_idx (multi-panel)
  - int -> plot one comparison
  - list[int] -> plot selected comparisons
- ax: supported only for a single comparison (exactly one k)

Returns matplotlib.figure.Figure.

####### `run_overlap_diagnostics`

```python
run_overlap_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05
) -> Dict[str, Any]
```

Run multi-treatment overlap diagnostics from data and estimate.

Diagnostics are computed pairwise between baseline treatment 0 and each
active treatment k using pairwise conditional propensity
P(D=k | X, D in {0, k}) as the comparison score.

###### `overlap_plot`

**Functions:**

- [**overlap_plot**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap_plot.overlap_plot) – Convenience wrapper to match `overlap_plot(data, estimate)` API style.
- [**plot_m_overlap**](#causalis.scenarios.multi_unconfoundedness.refutation.overlap_plot.plot_m_overlap) – Multi-treatment overlap plot for pairwise conditional propensity scores.

####### `overlap_plot`

```python
overlap_plot(
    data: MultiCausalData, estimate: MultiCausalEstimate, **kwargs: Any
) -> plt.Figure
```

Convenience wrapper to match `overlap_plot(data, estimate)` API style.

####### `plot_m_overlap`

```python
plot_m_overlap(
    diag: Union[
        MultiUnconfoundednessDiagnosticData, MultiCausalEstimate, dict, Any
    ],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
    *,
    treatment_idx: Optional[Union[int, List[int]]] = None,
    baseline_idx: int = 0,
    treatment_names: Optional[List[str]] = None
) -> plt.Figure
```

Multi-treatment overlap plot for pairwise conditional propensity scores.

For each comparison baseline (default 0) vs k, this plots
`P(D=k | X, D in {baseline, k}) = m_k(X) / (m_baseline(X) + m_k(X))`
on the observed pair sample `D in {baseline, k}`, comparing:

- units with D=k (treated for the pair),
- units with D=baseline (control for the pair).

Parameters:

- diag.d: (n, K) one-hot
- diag.m_hat / diag.m_hat_raw: (n, K) propensity
- treatment_idx:
  - None -> plot all k != baseline_idx (multi-panel)
  - int -> plot one comparison
  - list[int] -> plot selected comparisons
- ax: supported only for a single comparison (exactly one k)

Returns matplotlib.figure.Figure.

###### `plot_m_overlap`

```python
plot_m_overlap(
    diag: Union[
        MultiUnconfoundednessDiagnosticData, MultiCausalEstimate, dict, Any
    ],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
    *,
    treatment_idx: Optional[Union[int, List[int]]] = None,
    baseline_idx: int = 0,
    treatment_names: Optional[List[str]] = None
) -> plt.Figure
```

Multi-treatment overlap plot for pairwise conditional propensity scores.

For each comparison baseline (default 0) vs k, this plots
`P(D=k | X, D in {baseline, k}) = m_k(X) / (m_baseline(X) + m_k(X))`
on the observed pair sample `D in {baseline, k}`, comparing:

- units with D=k (treated for the pair),
- units with D=baseline (control for the pair).

Parameters:

- diag.d: (n, K) one-hot
- diag.m_hat / diag.m_hat_raw: (n, K) propensity
- treatment_idx:
  - None -> plot all k != baseline_idx (multi-panel)
  - int -> plot one comparison
  - list[int] -> plot selected comparisons
- ax: supported only for a single comparison (exactly one k)

Returns matplotlib.figure.Figure.

###### `plot_residual_diagnostics`

```python
plot_residual_diagnostics(
    estimate: MultiCausalEstimate,
    data: Optional[MultiCausalData] = None,
    *,
    clip_propensity: float = 1e-06,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot residual diagnostics for multi-treatment nuisance models.

<details class="panels" open markdown="1">
<summary>Panels</summary>

1..K. Arm-specific residual-vs-fitted:
`u_k = y - g_k` vs `g_k` within arm `D_k=1`.
K+1. Binned calibration error for each arm:
`E[D_k - m_k | m_k in bin]` vs binned `m_k`.

</details>

###### `run_overlap_diagnostics`

```python
run_overlap_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05
) -> Dict[str, Any]
```

Run multi-treatment overlap diagnostics from data and estimate.

Diagnostics are computed pairwise between baseline treatment 0 and each
active treatment k using pairwise conditional propensity
P(D=k | X, D in {0, k}) as the comparison score.

###### `run_score_diagnostics`

```python
run_score_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run score diagnostics for multi-treatment baseline contrasts.

###### `run_unconfoundedness_diagnostics`

```python
run_unconfoundedness_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run multi-treatment unconfoundedness diagnostics from data and estimate.

This implementation currently supports ATE diagnostics only and computes
pairwise balance between baseline treatment 0 and each active treatment k.

###### `score`

**Modules:**

- [**residual_plots**](#causalis.scenarios.multi_unconfoundedness.refutation.score.residual_plots) – Residual diagnostic plots for multi-treatment nuisance models g_k and m_k.
- [**score_validation**](#causalis.scenarios.multi_unconfoundedness.refutation.score.score_validation) – Score diagnostics for multi-treatment unconfoundedness.

**Functions:**

- [**plot_residual_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.score.plot_residual_diagnostics) – Plot residual diagnostics for multi-treatment nuisance models.
- [**run_score_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.score.run_score_diagnostics) – Run score diagnostics for multi-treatment baseline contrasts.

####### `plot_residual_diagnostics`

```python
plot_residual_diagnostics(
    estimate: MultiCausalEstimate,
    data: Optional[MultiCausalData] = None,
    *,
    clip_propensity: float = 1e-06,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot residual diagnostics for multi-treatment nuisance models.

<details class="panels" open markdown="1">
<summary>Panels</summary>

1..K. Arm-specific residual-vs-fitted:
`u_k = y - g_k` vs `g_k` within arm `D_k=1`.
K+1. Binned calibration error for each arm:
`E[D_k - m_k | m_k in bin]` vs binned `m_k`.

</details>

####### `residual_plots`

Residual diagnostic plots for multi-treatment nuisance models g_k and m_k.

**Functions:**

- [**plot_residual_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.score.residual_plots.plot_residual_diagnostics) – Plot residual diagnostics for multi-treatment nuisance models.

######## `plot_residual_diagnostics`

```python
plot_residual_diagnostics(
    estimate: MultiCausalEstimate,
    data: Optional[MultiCausalData] = None,
    *,
    clip_propensity: float = 1e-06,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot residual diagnostics for multi-treatment nuisance models.

<details class="panels" open markdown="1">
<summary>Panels</summary>

1..K. Arm-specific residual-vs-fitted:
`u_k = y - g_k` vs `g_k` within arm `D_k=1`.
K+1. Binned calibration error for each arm:
`E[D_k - m_k | m_k in bin]` vs binned `m_k`.

</details>

####### `run_score_diagnostics`

```python
run_score_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run score diagnostics for multi-treatment baseline contrasts.

####### `score_validation`

Score diagnostics for multi-treatment unconfoundedness.

**Functions:**

- [**run_score_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.score.score_validation.run_score_diagnostics) – Run score diagnostics for multi-treatment baseline contrasts.

######## `run_score_diagnostics`

```python
run_score_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run score diagnostics for multi-treatment baseline contrasts.

###### `sensitivity_analysis`

```python
sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    cf_y: Optional[float] = None,
    r2_y: Optional[float] = None,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]
```

###### `sensitivity_benchmark`

```python
sensitivity_benchmark(
    effect_estimation: Dict[str, Any] | Any,
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

###### `unconfoundedness`

**Modules:**

- [**sensitivity**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity) –
- [**unconfoundedness_validation**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation) – Unconfoundedness diagnostics for multi-treatment settings.

**Functions:**

- [**compute_bias_aware_ci**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.compute_bias_aware_ci) –
- [**get_sensitivity_summary**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.get_sensitivity_summary) –
- [**run_unconfoundedness_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.run_unconfoundedness_diagnostics) – Run multi-treatment unconfoundedness diagnostics from data and estimate.
- [**sensitivity_analysis**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity_analysis) –
- [**sensitivity_benchmark**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity_benchmark) –
- [**validate_unconfoundedness_balance**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.validate_unconfoundedness_balance) – Convenience wrapper returning the balance block only.

####### `compute_bias_aware_ci`

```python
compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    cf_y: float = 0.0,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]
```

####### `get_sensitivity_summary`

```python
get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    label: Optional[str] = None,
) -> Optional[str]
```

####### `run_unconfoundedness_diagnostics`

```python
run_unconfoundedness_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run multi-treatment unconfoundedness diagnostics from data and estimate.

This implementation currently supports ATE diagnostics only and computes
pairwise balance between baseline treatment 0 and each active treatment k.

####### `sensitivity`

**Functions:**

- [**compute_bias_aware_ci**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity.compute_bias_aware_ci) –
- [**get_sensitivity_summary**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity.get_sensitivity_summary) –
- [**sensitivity_analysis**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity.sensitivity_analysis) –
- [**sensitivity_benchmark**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity.sensitivity_benchmark) –

######## `compute_bias_aware_ci`

```python
compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    cf_y: float = 0.0,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]
```

######## `get_sensitivity_summary`

```python
get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    label: Optional[str] = None,
) -> Optional[str]
```

######## `sensitivity_analysis`

```python
sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    cf_y: Optional[float] = None,
    r2_y: Optional[float] = None,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]
```

######## `sensitivity_benchmark`

```python
sensitivity_benchmark(
    effect_estimation: Dict[str, Any] | Any,
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

####### `sensitivity_analysis`

```python
sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    _: Dict[str, Any] | Any = None,
    cf_y: Optional[float] = None,
    r2_y: Optional[float] = None,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]
```

####### `sensitivity_benchmark`

```python
sensitivity_benchmark(
    effect_estimation: Dict[str, Any] | Any,
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

####### `unconfoundedness_validation`

Unconfoundedness diagnostics for multi-treatment settings.

**Functions:**

- [**run_unconfoundedness_diagnostics**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation.run_unconfoundedness_diagnostics) – Run multi-treatment unconfoundedness diagnostics from data and estimate.
- [**validate_unconfoundedness_balance**](#causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation.validate_unconfoundedness_balance) – Convenience wrapper returning the balance block only.

######## `run_unconfoundedness_diagnostics`

```python
run_unconfoundedness_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run multi-treatment unconfoundedness diagnostics from data and estimate.

This implementation currently supports ATE diagnostics only and computes
pairwise balance between baseline treatment 0 and each active treatment k.

######## `validate_unconfoundedness_balance`

```python
validate_unconfoundedness_balance(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None
) -> Dict[str, Any]
```

Convenience wrapper returning the balance block only.

####### `validate_unconfoundedness_balance`

```python
validate_unconfoundedness_balance(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None
) -> Dict[str, Any]
```

Convenience wrapper returning the balance block only.

###### `validate_unconfoundedness_balance`

```python
validate_unconfoundedness_balance(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None
) -> Dict[str, Any]
```

Convenience wrapper returning the balance block only.

#### `synthetic_control`

**Modules:**

- [**dgp**](#causalis.scenarios.synthetic_control.dgp) –
- [**donors_diagnostics**](#causalis.scenarios.synthetic_control.donors_diagnostics) –
- [**model**](#causalis.scenarios.synthetic_control.model) –
- [**outcome_panel_plot**](#causalis.scenarios.synthetic_control.outcome_panel_plot) –
- [**refutation**](#causalis.scenarios.synthetic_control.refutation) –

**Classes:**

- [**AugmentedSyntheticControl**](#causalis.scenarios.synthetic_control.AugmentedSyntheticControl) – Ridge-augmented synthetic control with simplex anchor and aggregate-first inference.

**Functions:**

- [**gap_over_time_plot**](#causalis.scenarios.synthetic_control.gap_over_time_plot) – Plot observed-minus-synthetic gap over time with intervention boundary.
- [**generate_scm_gamma_26**](#causalis.scenarios.synthetic_control.generate_scm_gamma_26) – Generate realistic Gamma synthetic-control panel data.
- [**generate_scm_poisson_26**](#causalis.scenarios.synthetic_control.generate_scm_poisson_26) – Generate realistic Poisson synthetic-control panel data.
- [**leave_one_donor_out_sensitivity**](#causalis.scenarios.synthetic_control.leave_one_donor_out_sensitivity) – Re-fit ASCM dropping each donor once and report leave-one-donor-out sensitivity.
- [**observed_vs_synthetic_plot**](#causalis.scenarios.synthetic_control.observed_vs_synthetic_plot) – Plot observed treated path against augmented/SC synthetic paths.
- [**placebo_in_space_table**](#causalis.scenarios.synthetic_control.placebo_in_space_table) – Build Abadie-style placebo-in-space RMSPE ratio table.
- [**placebo_in_time_table**](#causalis.scenarios.synthetic_control.placebo_in_time_table) – Build pre-treatment-only placebo-in-time falsification table.
- [**run_placebo_tests**](#causalis.scenarios.synthetic_control.run_placebo_tests) – Run placebo-in-space and placebo-in-time robustness tests.
- [**run_scm_diagnostics**](#causalis.scenarios.synthetic_control.run_scm_diagnostics) – Run compact SCM diagnostics and return thresholded checks.
- [**run_scm_feasibility**](#causalis.scenarios.synthetic_control.run_scm_feasibility) – Return core SCM feasibility tables from panel data only (EDA phase).

##### `ASCM`

```python
ASCM = AugmentedSyntheticControl
```

##### `AugmentedSyntheticControl`

```python
AugmentedSyntheticControl(
    *,
    lambda_aug: float = 1.0,
    lambda_sc: float = 1e-06,
    max_iter: int = 2000,
    tol: float = 1e-09,
    enforce_sum_to_one_augmented: bool = True,
    alpha: float = 0.05,
    conformal_grid_size: int = 401,
    conformal_grid_min: float | None = None,
    conformal_grid_max: float | None = None,
    conformal_grid_scale_mult: float = 6.0,
    average_att_n_folds: int = 3,
    compute_average_att_ttest: bool = True,
    compute_pointwise_conformal: bool = False
) -> None
```

Ridge-augmented synthetic control with simplex anchor and aggregate-first inference.

<details class="note" open markdown="1">
<summary>Notes</summary>

Average ATT t-test inference is the default post-treatment inference layer.
Pointwise conformal intervals/p-values are optional and can be enabled for
dynamic path uncertainty quantification.

</details>

**Functions:**

- [**estimate**](#causalis.scenarios.synthetic_control.AugmentedSyntheticControl.estimate) – Return dynamic-path estimate object.
- [**fit**](#causalis.scenarios.synthetic_control.AugmentedSyntheticControl.fit) – Fit ASCM and compute inference outputs.

**Parameters:**

- **lambda_aug** (<code>[float](#float)</code>) – Ridge regularization for augmented weights.
- **lambda_sc** (<code>[float](#float)</code>) – Numerical regularization for simplex SCM weights.
- **max_iter** (<code>[int](#int)</code>) – Maximum iterations for constrained optimization routines.
- **tol** (<code>[float](#float)</code>) – Optimization tolerance.
- **enforce_sum_to_one_augmented** (<code>[bool](#bool)</code>) – Enforce sum-to-one constraint on augmented weights.
- **alpha** (<code>[float](#float)</code>) – Default significance level used by `estimate()` inference.
- **conformal_grid_size** (<code>[int](#int)</code>) – Default number of grid points used in pointwise conformal inversion.
- **conformal_grid_min** (<code>[float](#float) or None</code>) – Optional default fixed lower bound for conformal grid.
- **conformal_grid_max** (<code>[float](#float) or None</code>) – Optional default fixed upper bound for conformal grid.
- **conformal_grid_scale_mult** (<code>[float](#float)</code>) – Default scale multiplier for automatic conformal grid width.
- **average_att_n_folds** (<code>[int](#int)</code>) – Default requested number of folds for average ATT t-test inference.
- **compute_average_att_ttest** (<code>[bool](#bool)</code>) – Default toggle for average ATT t-test inference in `estimate()`.
- **compute_pointwise_conformal** (<code>[bool](#bool)</code>) – Default toggle for pointwise conformal CIs/p-values in `estimate()`.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any hyperparameter is invalid.

###### `alpha`

```python
alpha = float(alpha)
```

###### `average_att_n_folds`

```python
average_att_n_folds = int(average_att_n_folds)
```

###### `compute_average_att_ttest`

```python
compute_average_att_ttest = bool(compute_average_att_ttest)
```

###### `compute_pointwise_conformal`

```python
compute_pointwise_conformal = bool(compute_pointwise_conformal)
```

###### `conformal_grid_max`

```python
conformal_grid_max = (
    None if conformal_grid_max is None else float(conformal_grid_max)
)

```

###### `conformal_grid_min`

```python
conformal_grid_min = (
    None if conformal_grid_min is None else float(conformal_grid_min)
)

```

###### `conformal_grid_scale_mult`

```python
conformal_grid_scale_mult = float(conformal_grid_scale_mult)
```

###### `conformal_grid_size`

```python
conformal_grid_size = int(conformal_grid_size)
```

###### `enforce_sum_to_one_augmented`

```python
enforce_sum_to_one_augmented = bool(enforce_sum_to_one_augmented)
```

###### `estimate`

```python
estimate(
    *,
    alpha: float | None = None,
    conformal_grid_size: int | None = None,
    conformal_grid_min: float | None = None,
    conformal_grid_max: float | None = None,
    conformal_grid_scale_mult: float | None = None,
    average_att_n_folds: int | None = None,
    compute_average_att_ttest: bool | None = None,
    compute_pointwise_conformal: bool | None = None
) -> PanelEstimate
```

Return dynamic-path estimate object.

**Parameters:**

- **alpha** (<code>[float](#float) or None</code>) – Optional per-call significance level override.
- **conformal_grid_size** (<code>[int](#int) or None</code>) – Optional per-call pointwise conformal grid size override.
- **conformal_grid_min** (<code>[float](#float) or None</code>) – Optional per-call lower conformal grid bound override.
- **conformal_grid_max** (<code>[float](#float) or None</code>) – Optional per-call upper conformal grid bound override.
- **conformal_grid_scale_mult** (<code>[float](#float) or None</code>) – Optional per-call automatic conformal grid scale override.
- **average_att_n_folds** (<code>[int](#int) or None</code>) – Optional per-call fold-count override for average ATT inference.
- **compute_average_att_ttest** (<code>[bool](#bool) or None</code>) – Optional per-call toggle for average ATT t-test inference.
- **compute_pointwise_conformal** (<code>[bool](#bool) or None</code>) – Optional per-call toggle for pointwise conformal inference.

**Returns:**

- <code>[PanelEstimate](#causalis.data_contracts.panel_estimate.PanelEstimate)</code> – Dynamic path estimates with pointwise inference fields. Aggregate
  average ATT t-test outputs are provided in `diagnostics` and are
  the default formal inference layer. If pointwise conformal is not
  computed, pointwise p-values/CIs are returned as `NaN` placeholders.

**Raises:**

- <code>[RuntimeError](#RuntimeError)</code> – If the model is not fitted.

###### `fit`

```python
fit(data: PanelDataSCM) -> 'AugmentedSyntheticControl'
```

Fit ASCM and compute inference outputs.

**Parameters:**

- **data** (<code>[PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code>) – Validated synthetic-control panel data.

**Returns:**

- <code>[AugmentedSyntheticControl](#causalis.scenarios.synthetic_control.model.AugmentedSyntheticControl)</code> – Fitted estimator instance.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If input type is invalid or panel requirements are violated.

###### `lambda_aug`

```python
lambda_aug = float(lambda_aug)
```

###### `lambda_sc`

```python
lambda_sc = float(lambda_sc)
```

###### `max_iter`

```python
max_iter = int(max_iter)
```

###### `tol`

```python
tol = float(tol)
```

##### `dgp`

**Functions:**

- [**generate_scm_gamma_26**](#causalis.scenarios.synthetic_control.dgp.generate_scm_gamma_26) – Generate realistic Gamma synthetic-control panel data.
- [**generate_scm_poisson_26**](#causalis.scenarios.synthetic_control.dgp.generate_scm_poisson_26) – Generate realistic Poisson synthetic-control panel data.

###### `PanelOutput`

```python
PanelOutput = Union[pd.DataFrame, PanelDataSCM]
```

###### `generate_scm_gamma_26`

```python
generate_scm_gamma_26(
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 40,
    n_pre_periods: Optional[int] = 36,
    n_post_periods: Optional[int] = 6,
    treatment_effect_rate: float = 0.1,
    treatment_effect_slope: float = 0.002,
    **advanced_params: float
) -> PanelOutput
```

Generate realistic Gamma synthetic-control panel data.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **return_panel_data** (<code>[bool](#bool)</code>) – If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
  object. If False, return a pandas DataFrame.
- **include_oracles** (<code>[bool](#bool)</code>) – Whether to include oracle truth columns in the returned data:
  `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
  `mu_treated`, `tau_mean_true`.
  Scenario-level outputs always exclude synthetic covariates
  `exposure`, `macro_index`, `seasonality_index`.
- **n_donors** (<code>[int](#int)</code>) – Number of donor units.
- **n_pre_periods** (<code>[int](#int) or None</code>) – Number of pre-treatment periods. Preferred explicit horizon control.
  When both `n_pre_periods` and `n_post_periods` are omitted, scenario
  defaults are used (`36` pre, `6` post). The generated panel includes
  one explicit intervention-anchor period, so each unit has
  `n_pre_periods + 1 + n_post_periods` rows.
- **n_post_periods** (<code>[int](#int) or None</code>) – Number of post-treatment periods. Must be provided together with
  `n_pre_periods` when using explicit horizon control.
- **treatment_effect_rate** (<code>[float](#float)</code>) – Long-run post-treatment relative effect scale. The first post period is
  attenuated by a ramp factor `1 - exp(-1 / 2.5)` (about 0.33x when slope
  is zero).
- **treatment_effect_slope** (<code>[float](#float)</code>) – Linear slope of the post-treatment relative effect path.
- \*\***advanced_params** – Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_gamma_data`.
  Common advanced knobs include `time_start`, `calendar_start`,
  and latent-factor configuration. Defaults used by this wrapper are
  `gamma_shape=120`, `donor_noise_std_log=0.03`,
  `common_factor_std_log=0.03`, `latent_factor_std_log=0.0`,
  and `prefit_mismatch_std_log=0.0`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code> – Long panel data for SCM experiments.

<details class="note" open markdown="1">
<summary>Notes</summary>

DGP Math:
The data follows a hierarchical log-linear model for the mean: math:`\mu`.
For each donor unit :math:`j` at time :math:`t`, the mean is :math:`\mu_{tj} = E_{tj} \cdot \exp(\eta_{tj})`
where :math:`E_{tj}` is exposure (with growth and noise) and :math:`\eta_{tj}` includes seasonality,
common factors (macro index), latent factors, and unit-specific noise.
Outcomes are sampled as :math:`y_{tj} \sim \text{Gamma}(k, \mu_{tj}/k)`, where :math:`k` is `gamma_shape`.

The treated unit's counterfactual mean :math:`\mu_{t, cf}` is a weighted combination of donors
(via Dirichlet weights) with a potential pre-fit mismatch. The realized treated outcome
is :math:`y_{t, treated} = y_{t, cf} \cdot (1 + \tau_t^{rate})`, where :math:`\tau_t^{rate}` is the
relative treatment effect.

Time-axis semantics:

- `n_pre_periods`: number of periods strictly before the intervention anchor.
- One explicit intervention-anchor period is included in the output.
- `n_post_periods`: number of periods strictly after the intervention anchor.
- `time_start`: offset for the first `calendar_time` period relative to
  `calendar_start` (default `calendar_start="2000-01"` and `time_start=1`).
- `treated_time`: explicit 0/1 treatment-assignment indicator in returned
  data (`1` only for treated-unit rows at/after the first treated period;
  `0` otherwise).
- `PanelDataSCM` is built with required fields only:
  `df`, `y`, `unit_col`, `time_col`, `treated_time`.
- When `return_panel_data=True`, all contract metadata is derived from the
  final `treated_time` path. Because this scenario keeps one explicit anchor
  period in the panel, contract-level pre periods are
  `n_pre_periods + 1` and post periods are `n_post_periods`.
- With this function's default arguments, the explicit values are:
  `n_pre_periods=36`, `n_post_periods=6`, `calendar_start='2000-01'`,
  `time_start=1`, first treated period at `Period('2003-02', 'M')`,
  and intervention anchor at `Period('2003-01', 'M')`.

</details>

###### `generate_scm_poisson_26`

```python
generate_scm_poisson_26(
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 20,
    n_pre_periods: Optional[int] = 180,
    n_post_periods: Optional[int] = 4,
    treatment_effect_rate: float = 0.15,
    treatment_effect_slope: float = 0.0005,
    **advanced_params: float
) -> PanelOutput
```

Generate realistic Poisson synthetic-control panel data.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **return_panel_data** (<code>[bool](#bool)</code>) – If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
  object. If False, return a pandas DataFrame.
- **include_oracles** (<code>[bool](#bool)</code>) – Whether to include oracle truth columns in the returned data:
  `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
  `mu_treated`, `tau_mean_true`.
  Scenario-level outputs always exclude synthetic covariates
  `exposure`, `macro_index`, `seasonality_index`.
- **n_donors** (<code>[int](#int)</code>) – Number of donor units.
- **n_pre_periods** (<code>[int](#int) or None</code>) – Number of pre-treatment periods. Preferred explicit horizon control.
  When both `n_pre_periods` and `n_post_periods` are omitted, scenario
  defaults are used (`36` pre, `6` post). The generated panel includes
  one explicit intervention-anchor period, so each unit has
  `n_pre_periods + 1 + n_post_periods` rows.
- **n_post_periods** (<code>[int](#int) or None</code>) – Number of post-treatment periods. Must be provided together with
  `n_pre_periods` when using explicit horizon control.
- **treatment_effect_rate** (<code>[float](#float)</code>) – Long-run post-treatment relative effect scale. The first post period is
  attenuated by a ramp factor `1 - exp(-1 / 2.5)` (about 0.33x when slope
  is zero).
- **treatment_effect_slope** (<code>[float](#float)</code>) – Linear slope of the post-treatment relative effect path.
- \*\***advanced_params** – Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_poisson_data`.
  Common advanced knobs include `time_start`, `calendar_start`,
  and latent-factor configuration. Defaults used by this wrapper are
  `donor_noise_std_log=0.02`, `common_factor_std_log=0.02`,
  `latent_factor_std_log=0.0`, and `prefit_mismatch_std_log=0.0`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code> – Long panel data for SCM experiments.

<details class="note" open markdown="1">
<summary>Notes</summary>

DGP Math:
The data follows a hierarchical log-linear model for the mean :math:`\mu`.
For each donor unit :math:`j` at time :math:`t`, the mean is :math:`\mu_{tj} = E_{tj} \cdot \exp(\eta_{tj})`
where :math:`E_{tj}` is exposure and :math:`\eta_{tj}` includes seasonality, common factors,
latent factors, and unit-specific noise.
Outcomes are sampled as :math:`y_{tj} \sim \text{Poisson}(\mu_{tj})`.

The treated unit's counterfactual mean :math:`\mu_{t, cf}` is a weighted combination of donors.
The realized treated outcome :math:`y_{t, treated}` is sampled from a Poisson distribution
coupled with the counterfactual :math:`y_{t, cf}` via a thinning/superposition property to
maintain exact marginals while ensuring the realized effect is driven by the multiplier.

Time-axis semantics:

- `n_pre_periods`: number of periods strictly before the intervention anchor.
- One explicit intervention-anchor period is included in the output.
- `n_post_periods`: number of periods strictly after the intervention anchor.
- `time_start`: offset for the first `calendar_time` period relative to
  `calendar_start` (default `calendar_start="2000-01"` and `time_start=1`).
- `treated_time`: explicit 0/1 treatment-assignment indicator in returned
  data (`1` only for treated-unit rows at/after the first treated period;
  `0` otherwise).
- `PanelDataSCM` is built with required fields only:
  `df`, `y`, `unit_col`, `time_col`, `treated_time`.
- When `return_panel_data=True`, all contract metadata is derived from the
  final `treated_time` path. Because this scenario keeps one explicit anchor
  period in the panel, contract-level pre periods are
  `n_pre_periods + 1` and post periods are `n_post_periods`.
- With this function's default arguments, the explicit values are:
  `n_pre_periods=180`, `n_post_periods=4`, `calendar_start='2000-01'`,
  `time_start=1`, first treated period at `Period('2003-02', 'M')`,
  and intervention anchor at `Period('2003-01', 'M')`.

</details>

##### `donors_diagnostics`

**Functions:**

- [**donors_diagnostics**](#causalis.scenarios.synthetic_control.donors_diagnostics.donors_diagnostics) – Build one donor-level diagnostics table for SCM feasibility checks.
- [**run_scm_feasibility**](#causalis.scenarios.synthetic_control.donors_diagnostics.run_scm_feasibility) – Return core SCM feasibility tables from panel data only (EDA phase).

###### `donors_diagnostics`

```python
donors_diagnostics(panel_data: PanelDataSCM) -> pd.DataFrame
```

Build one donor-level diagnostics table for SCM feasibility checks.

###### `run_scm_feasibility`

```python
run_scm_feasibility(paneldata: PanelDataSCM) -> Dict[str, pd.DataFrame]
```

Return core SCM feasibility tables from panel data only (EDA phase).

##### `gap_over_time_plot`

```python
gap_over_time_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1
) -> plt.Figure
```

Plot observed-minus-synthetic gap over time with intervention boundary.

##### `generate_scm_gamma_26`

```python
generate_scm_gamma_26(
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 40,
    n_pre_periods: Optional[int] = 36,
    n_post_periods: Optional[int] = 6,
    treatment_effect_rate: float = 0.1,
    treatment_effect_slope: float = 0.002,
    **advanced_params: float
) -> PanelOutput
```

Generate realistic Gamma synthetic-control panel data.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **return_panel_data** (<code>[bool](#bool)</code>) – If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
  object. If False, return a pandas DataFrame.
- **include_oracles** (<code>[bool](#bool)</code>) – Whether to include oracle truth columns in the returned data:
  `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
  `mu_treated`, `tau_mean_true`.
  Scenario-level outputs always exclude synthetic covariates
  `exposure`, `macro_index`, `seasonality_index`.
- **n_donors** (<code>[int](#int)</code>) – Number of donor units.
- **n_pre_periods** (<code>[int](#int) or None</code>) – Number of pre-treatment periods. Preferred explicit horizon control.
  When both `n_pre_periods` and `n_post_periods` are omitted, scenario
  defaults are used (`36` pre, `6` post). The generated panel includes
  one explicit intervention-anchor period, so each unit has
  `n_pre_periods + 1 + n_post_periods` rows.
- **n_post_periods** (<code>[int](#int) or None</code>) – Number of post-treatment periods. Must be provided together with
  `n_pre_periods` when using explicit horizon control.
- **treatment_effect_rate** (<code>[float](#float)</code>) – Long-run post-treatment relative effect scale. The first post period is
  attenuated by a ramp factor `1 - exp(-1 / 2.5)` (about 0.33x when slope
  is zero).
- **treatment_effect_slope** (<code>[float](#float)</code>) – Linear slope of the post-treatment relative effect path.
- \*\***advanced_params** – Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_gamma_data`.
  Common advanced knobs include `time_start`, `calendar_start`,
  and latent-factor configuration. Defaults used by this wrapper are
  `gamma_shape=120`, `donor_noise_std_log=0.03`,
  `common_factor_std_log=0.03`, `latent_factor_std_log=0.0`,
  and `prefit_mismatch_std_log=0.0`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code> – Long panel data for SCM experiments.

<details class="note" open markdown="1">
<summary>Notes</summary>

DGP Math:
The data follows a hierarchical log-linear model for the mean: math:`\mu`.
For each donor unit :math:`j` at time :math:`t`, the mean is :math:`\mu_{tj} = E_{tj} \cdot \exp(\eta_{tj})`
where :math:`E_{tj}` is exposure (with growth and noise) and :math:`\eta_{tj}` includes seasonality,
common factors (macro index), latent factors, and unit-specific noise.
Outcomes are sampled as :math:`y_{tj} \sim \text{Gamma}(k, \mu_{tj}/k)`, where :math:`k` is `gamma_shape`.

The treated unit's counterfactual mean :math:`\mu_{t, cf}` is a weighted combination of donors
(via Dirichlet weights) with a potential pre-fit mismatch. The realized treated outcome
is :math:`y_{t, treated} = y_{t, cf} \cdot (1 + \tau_t^{rate})`, where :math:`\tau_t^{rate}` is the
relative treatment effect.

Time-axis semantics:

- `n_pre_periods`: number of periods strictly before the intervention anchor.
- One explicit intervention-anchor period is included in the output.
- `n_post_periods`: number of periods strictly after the intervention anchor.
- `time_start`: offset for the first `calendar_time` period relative to
  `calendar_start` (default `calendar_start="2000-01"` and `time_start=1`).
- `treated_time`: explicit 0/1 treatment-assignment indicator in returned
  data (`1` only for treated-unit rows at/after the first treated period;
  `0` otherwise).
- `PanelDataSCM` is built with required fields only:
  `df`, `y`, `unit_col`, `time_col`, `treated_time`.
- When `return_panel_data=True`, all contract metadata is derived from the
  final `treated_time` path. Because this scenario keeps one explicit anchor
  period in the panel, contract-level pre periods are
  `n_pre_periods + 1` and post periods are `n_post_periods`.
- With this function's default arguments, the explicit values are:
  `n_pre_periods=36`, `n_post_periods=6`, `calendar_start='2000-01'`,
  `time_start=1`, first treated period at `Period('2003-02', 'M')`,
  and intervention anchor at `Period('2003-01', 'M')`.

</details>

##### `generate_scm_poisson_26`

```python
generate_scm_poisson_26(
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 20,
    n_pre_periods: Optional[int] = 180,
    n_post_periods: Optional[int] = 4,
    treatment_effect_rate: float = 0.15,
    treatment_effect_slope: float = 0.0005,
    **advanced_params: float
) -> PanelOutput
```

Generate realistic Poisson synthetic-control panel data.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **return_panel_data** (<code>[bool](#bool)</code>) – If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
  object. If False, return a pandas DataFrame.
- **include_oracles** (<code>[bool](#bool)</code>) – Whether to include oracle truth columns in the returned data:
  `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
  `mu_treated`, `tau_mean_true`.
  Scenario-level outputs always exclude synthetic covariates
  `exposure`, `macro_index`, `seasonality_index`.
- **n_donors** (<code>[int](#int)</code>) – Number of donor units.
- **n_pre_periods** (<code>[int](#int) or None</code>) – Number of pre-treatment periods. Preferred explicit horizon control.
  When both `n_pre_periods` and `n_post_periods` are omitted, scenario
  defaults are used (`36` pre, `6` post). The generated panel includes
  one explicit intervention-anchor period, so each unit has
  `n_pre_periods + 1 + n_post_periods` rows.
- **n_post_periods** (<code>[int](#int) or None</code>) – Number of post-treatment periods. Must be provided together with
  `n_pre_periods` when using explicit horizon control.
- **treatment_effect_rate** (<code>[float](#float)</code>) – Long-run post-treatment relative effect scale. The first post period is
  attenuated by a ramp factor `1 - exp(-1 / 2.5)` (about 0.33x when slope
  is zero).
- **treatment_effect_slope** (<code>[float](#float)</code>) – Linear slope of the post-treatment relative effect path.
- \*\***advanced_params** – Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_poisson_data`.
  Common advanced knobs include `time_start`, `calendar_start`,
  and latent-factor configuration. Defaults used by this wrapper are
  `donor_noise_std_log=0.02`, `common_factor_std_log=0.02`,
  `latent_factor_std_log=0.0`, and `prefit_mismatch_std_log=0.0`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code> – Long panel data for SCM experiments.

<details class="note" open markdown="1">
<summary>Notes</summary>

DGP Math:
The data follows a hierarchical log-linear model for the mean :math:`\mu`.
For each donor unit :math:`j` at time :math:`t`, the mean is :math:`\mu_{tj} = E_{tj} \cdot \exp(\eta_{tj})`
where :math:`E_{tj}` is exposure and :math:`\eta_{tj}` includes seasonality, common factors,
latent factors, and unit-specific noise.
Outcomes are sampled as :math:`y_{tj} \sim \text{Poisson}(\mu_{tj})`.

The treated unit's counterfactual mean :math:`\mu_{t, cf}` is a weighted combination of donors.
The realized treated outcome :math:`y_{t, treated}` is sampled from a Poisson distribution
coupled with the counterfactual :math:`y_{t, cf}` via a thinning/superposition property to
maintain exact marginals while ensuring the realized effect is driven by the multiplier.

Time-axis semantics:

- `n_pre_periods`: number of periods strictly before the intervention anchor.
- One explicit intervention-anchor period is included in the output.
- `n_post_periods`: number of periods strictly after the intervention anchor.
- `time_start`: offset for the first `calendar_time` period relative to
  `calendar_start` (default `calendar_start="2000-01"` and `time_start=1`).
- `treated_time`: explicit 0/1 treatment-assignment indicator in returned
  data (`1` only for treated-unit rows at/after the first treated period;
  `0` otherwise).
- `PanelDataSCM` is built with required fields only:
  `df`, `y`, `unit_col`, `time_col`, `treated_time`.
- When `return_panel_data=True`, all contract metadata is derived from the
  final `treated_time` path. Because this scenario keeps one explicit anchor
  period in the panel, contract-level pre periods are
  `n_pre_periods + 1` and post periods are `n_post_periods`.
- With this function's default arguments, the explicit values are:
  `n_pre_periods=180`, `n_post_periods=4`, `calendar_start='2000-01'`,
  `time_start=1`, first treated period at `Period('2003-02', 'M')`,
  and intervention anchor at `Period('2003-01', 'M')`.

</details>

##### `leave_one_donor_out_sensitivity`

```python
leave_one_donor_out_sensitivity(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None
) -> pd.DataFrame
```

Re-fit ASCM dropping each donor once and report leave-one-donor-out sensitivity.

##### `model`

**Classes:**

- [**AugmentedSyntheticControl**](#causalis.scenarios.synthetic_control.model.AugmentedSyntheticControl) – Ridge-augmented synthetic control with simplex anchor and aggregate-first inference.

###### `ASCM`

```python
ASCM = AugmentedSyntheticControl
```

###### `AugmentedSyntheticControl`

```python
AugmentedSyntheticControl(
    *,
    lambda_aug: float = 1.0,
    lambda_sc: float = 1e-06,
    max_iter: int = 2000,
    tol: float = 1e-09,
    enforce_sum_to_one_augmented: bool = True,
    alpha: float = 0.05,
    conformal_grid_size: int = 401,
    conformal_grid_min: float | None = None,
    conformal_grid_max: float | None = None,
    conformal_grid_scale_mult: float = 6.0,
    average_att_n_folds: int = 3,
    compute_average_att_ttest: bool = True,
    compute_pointwise_conformal: bool = False
) -> None
```

Ridge-augmented synthetic control with simplex anchor and aggregate-first inference.

<details class="note" open markdown="1">
<summary>Notes</summary>

Average ATT t-test inference is the default post-treatment inference layer.
Pointwise conformal intervals/p-values are optional and can be enabled for
dynamic path uncertainty quantification.

</details>

**Functions:**

- [**estimate**](#causalis.scenarios.synthetic_control.model.AugmentedSyntheticControl.estimate) – Return dynamic-path estimate object.
- [**fit**](#causalis.scenarios.synthetic_control.model.AugmentedSyntheticControl.fit) – Fit ASCM and compute inference outputs.

**Parameters:**

- **lambda_aug** (<code>[float](#float)</code>) – Ridge regularization for augmented weights.
- **lambda_sc** (<code>[float](#float)</code>) – Numerical regularization for simplex SCM weights.
- **max_iter** (<code>[int](#int)</code>) – Maximum iterations for constrained optimization routines.
- **tol** (<code>[float](#float)</code>) – Optimization tolerance.
- **enforce_sum_to_one_augmented** (<code>[bool](#bool)</code>) – Enforce sum-to-one constraint on augmented weights.
- **alpha** (<code>[float](#float)</code>) – Default significance level used by `estimate()` inference.
- **conformal_grid_size** (<code>[int](#int)</code>) – Default number of grid points used in pointwise conformal inversion.
- **conformal_grid_min** (<code>[float](#float) or None</code>) – Optional default fixed lower bound for conformal grid.
- **conformal_grid_max** (<code>[float](#float) or None</code>) – Optional default fixed upper bound for conformal grid.
- **conformal_grid_scale_mult** (<code>[float](#float)</code>) – Default scale multiplier for automatic conformal grid width.
- **average_att_n_folds** (<code>[int](#int)</code>) – Default requested number of folds for average ATT t-test inference.
- **compute_average_att_ttest** (<code>[bool](#bool)</code>) – Default toggle for average ATT t-test inference in `estimate()`.
- **compute_pointwise_conformal** (<code>[bool](#bool)</code>) – Default toggle for pointwise conformal CIs/p-values in `estimate()`.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any hyperparameter is invalid.

####### `alpha`

```python
alpha = float(alpha)
```

####### `average_att_n_folds`

```python
average_att_n_folds = int(average_att_n_folds)
```

####### `compute_average_att_ttest`

```python
compute_average_att_ttest = bool(compute_average_att_ttest)
```

####### `compute_pointwise_conformal`

```python
compute_pointwise_conformal = bool(compute_pointwise_conformal)
```

####### `conformal_grid_max`

```python
conformal_grid_max = (
    None if conformal_grid_max is None else float(conformal_grid_max)
)

```

####### `conformal_grid_min`

```python
conformal_grid_min = (
    None if conformal_grid_min is None else float(conformal_grid_min)
)

```

####### `conformal_grid_scale_mult`

```python
conformal_grid_scale_mult = float(conformal_grid_scale_mult)
```

####### `conformal_grid_size`

```python
conformal_grid_size = int(conformal_grid_size)
```

####### `enforce_sum_to_one_augmented`

```python
enforce_sum_to_one_augmented = bool(enforce_sum_to_one_augmented)
```

####### `estimate`

```python
estimate(
    *,
    alpha: float | None = None,
    conformal_grid_size: int | None = None,
    conformal_grid_min: float | None = None,
    conformal_grid_max: float | None = None,
    conformal_grid_scale_mult: float | None = None,
    average_att_n_folds: int | None = None,
    compute_average_att_ttest: bool | None = None,
    compute_pointwise_conformal: bool | None = None
) -> PanelEstimate
```

Return dynamic-path estimate object.

**Parameters:**

- **alpha** (<code>[float](#float) or None</code>) – Optional per-call significance level override.
- **conformal_grid_size** (<code>[int](#int) or None</code>) – Optional per-call pointwise conformal grid size override.
- **conformal_grid_min** (<code>[float](#float) or None</code>) – Optional per-call lower conformal grid bound override.
- **conformal_grid_max** (<code>[float](#float) or None</code>) – Optional per-call upper conformal grid bound override.
- **conformal_grid_scale_mult** (<code>[float](#float) or None</code>) – Optional per-call automatic conformal grid scale override.
- **average_att_n_folds** (<code>[int](#int) or None</code>) – Optional per-call fold-count override for average ATT inference.
- **compute_average_att_ttest** (<code>[bool](#bool) or None</code>) – Optional per-call toggle for average ATT t-test inference.
- **compute_pointwise_conformal** (<code>[bool](#bool) or None</code>) – Optional per-call toggle for pointwise conformal inference.

**Returns:**

- <code>[PanelEstimate](#causalis.data_contracts.panel_estimate.PanelEstimate)</code> – Dynamic path estimates with pointwise inference fields. Aggregate
  average ATT t-test outputs are provided in `diagnostics` and are
  the default formal inference layer. If pointwise conformal is not
  computed, pointwise p-values/CIs are returned as `NaN` placeholders.

**Raises:**

- <code>[RuntimeError](#RuntimeError)</code> – If the model is not fitted.

####### `fit`

```python
fit(data: PanelDataSCM) -> 'AugmentedSyntheticControl'
```

Fit ASCM and compute inference outputs.

**Parameters:**

- **data** (<code>[PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code>) – Validated synthetic-control panel data.

**Returns:**

- <code>[AugmentedSyntheticControl](#causalis.scenarios.synthetic_control.model.AugmentedSyntheticControl)</code> – Fitted estimator instance.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If input type is invalid or panel requirements are violated.

####### `lambda_aug`

```python
lambda_aug = float(lambda_aug)
```

####### `lambda_sc`

```python
lambda_sc = float(lambda_sc)
```

####### `max_iter`

```python
max_iter = int(max_iter)
```

####### `tol`

```python
tol = float(tol)
```

##### `observed_vs_synthetic_plot`

```python
observed_vs_synthetic_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1
) -> plt.Figure
```

Plot observed treated path against augmented/SC synthetic paths.

##### `outcome_panel_plot`

**Functions:**

- [**outcome_panel_plot**](#causalis.scenarios.synthetic_control.outcome_panel_plot.outcome_panel_plot) – Plot SCM panel outcomes over time.

###### `outcome_panel_plot`

```python
outcome_panel_plot(
    paneldata: PanelDataSCM,
    *,
    show_donor_units: bool = True,
    donor_max_lines: Optional[int] = 20,
    show_donor_mean: bool = True,
    donor_alpha: float = 0.35,
    donor_linewidth: float = 1.2,
    shade_post_period: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot SCM panel outcomes over time.

The figure shows treated-unit outcomes over time, optional donor-unit paths,
optional donor mean, and the intervention boundary.

**Parameters:**

- **paneldata** (<code>[PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code>) – Validated long-format panel contract.
- **show_donor_units** (<code>[bool](#bool)</code>) – If True, draw donor trajectories.
- **donor_max_lines** (<code>[int](#int) or None</code>) – Maximum number of donor-unit lines to draw. `None` draws all donors.
- **show_donor_mean** (<code>[bool](#bool)</code>) – If True, draw the donor-pool mean outcome path.
- **donor_alpha** (<code>[float](#float)</code>) – Opacity for donor-unit lines.
- **donor_linewidth** (<code>[float](#float)</code>) – Line width for donor-unit lines.
- **shade_post_period** (<code>[bool](#bool)</code>) – If True, lightly shade the post-treatment region.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Optional path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saved raster outputs.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

##### `placebo_in_space_table`

```python
placebo_in_space_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None
) -> pd.DataFrame
```

Build Abadie-style placebo-in-space RMSPE ratio table.

For donor-as-treated placebo fits, the actual treated unit is excluded
from the donor pool to avoid post-treatment contamination.

##### `placebo_in_time_table`

```python
placebo_in_time_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
    pseudo_post_horizon: int | None = None
) -> pd.DataFrame
```

Build pre-treatment-only placebo-in-time falsification table.

##### `refutation`

**Modules:**

- [**diagnostic_plots**](#causalis.scenarios.synthetic_control.refutation.diagnostic_plots) –
- [**donors_diagnostics**](#causalis.scenarios.synthetic_control.refutation.donors_diagnostics) –
- [**outcome_panel_plot**](#causalis.scenarios.synthetic_control.refutation.outcome_panel_plot) –
- [**placebo**](#causalis.scenarios.synthetic_control.refutation.placebo) –
- [**scm_diagnostics**](#causalis.scenarios.synthetic_control.refutation.scm_diagnostics) –
- [**sensitivity**](#causalis.scenarios.synthetic_control.refutation.sensitivity) –

**Functions:**

- [**gap_over_time_plot**](#causalis.scenarios.synthetic_control.refutation.gap_over_time_plot) – Plot observed-minus-synthetic gap over time with intervention boundary.
- [**leave_one_donor_out_sensitivity**](#causalis.scenarios.synthetic_control.refutation.leave_one_donor_out_sensitivity) – Re-fit ASCM dropping each donor once and report leave-one-donor-out sensitivity.
- [**observed_vs_synthetic_plot**](#causalis.scenarios.synthetic_control.refutation.observed_vs_synthetic_plot) – Plot observed treated path against augmented/SC synthetic paths.
- [**placebo_in_space_table**](#causalis.scenarios.synthetic_control.refutation.placebo_in_space_table) – Build Abadie-style placebo-in-space RMSPE ratio table.
- [**placebo_in_time_table**](#causalis.scenarios.synthetic_control.refutation.placebo_in_time_table) – Build pre-treatment-only placebo-in-time falsification table.
- [**run_placebo_tests**](#causalis.scenarios.synthetic_control.refutation.run_placebo_tests) – Run placebo-in-space and placebo-in-time robustness tests.
- [**run_scm_diagnostics**](#causalis.scenarios.synthetic_control.refutation.run_scm_diagnostics) – Run compact SCM diagnostics and return thresholded checks.
- [**run_scm_feasibility**](#causalis.scenarios.synthetic_control.refutation.run_scm_feasibility) – Return core SCM feasibility tables from panel data only (EDA phase).

###### `diagnostic_plots`

**Functions:**

- [**gap_over_time_plot**](#causalis.scenarios.synthetic_control.refutation.diagnostic_plots.gap_over_time_plot) – Plot observed-minus-synthetic gap over time with intervention boundary.
- [**observed_vs_synthetic_plot**](#causalis.scenarios.synthetic_control.refutation.diagnostic_plots.observed_vs_synthetic_plot) – Plot observed treated path against augmented/SC synthetic paths.

####### `gap_over_time_plot`

```python
gap_over_time_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1
) -> plt.Figure
```

Plot observed-minus-synthetic gap over time with intervention boundary.

####### `observed_vs_synthetic_plot`

```python
observed_vs_synthetic_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1
) -> plt.Figure
```

Plot observed treated path against augmented/SC synthetic paths.

###### `donors_diagnostics`

**Functions:**

- [**donors_diagnostics**](#causalis.scenarios.synthetic_control.refutation.donors_diagnostics.donors_diagnostics) – Build one donor-level diagnostics table for SCM feasibility checks.
- [**run_scm_feasibility**](#causalis.scenarios.synthetic_control.refutation.donors_diagnostics.run_scm_feasibility) – Return core SCM feasibility tables from panel data only (EDA phase).

####### `donors_diagnostics`

```python
donors_diagnostics(panel_data: PanelDataSCM) -> pd.DataFrame
```

Build one donor-level diagnostics table for SCM feasibility checks.

####### `run_scm_feasibility`

```python
run_scm_feasibility(paneldata: PanelDataSCM) -> Dict[str, pd.DataFrame]
```

Return core SCM feasibility tables from panel data only (EDA phase).

###### `gap_over_time_plot`

```python
gap_over_time_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1
) -> plt.Figure
```

Plot observed-minus-synthetic gap over time with intervention boundary.

###### `leave_one_donor_out_sensitivity`

```python
leave_one_donor_out_sensitivity(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None
) -> pd.DataFrame
```

Re-fit ASCM dropping each donor once and report leave-one-donor-out sensitivity.

###### `observed_vs_synthetic_plot`

```python
observed_vs_synthetic_plot(
    estimate: PanelEstimate,
    *,
    show_sc: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1
) -> plt.Figure
```

Plot observed treated path against augmented/SC synthetic paths.

###### `outcome_panel_plot`

**Functions:**

- [**outcome_panel_plot**](#causalis.scenarios.synthetic_control.refutation.outcome_panel_plot.outcome_panel_plot) – Plot SCM panel outcomes over time.

####### `outcome_panel_plot`

```python
outcome_panel_plot(
    paneldata: PanelDataSCM,
    *,
    show_donor_units: bool = True,
    donor_max_lines: Optional[int] = 20,
    show_donor_mean: bool = True,
    donor_alpha: float = 0.35,
    donor_linewidth: float = 1.2,
    shade_post_period: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.5),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot SCM panel outcomes over time.

The figure shows treated-unit outcomes over time, optional donor-unit paths,
optional donor mean, and the intervention boundary.

**Parameters:**

- **paneldata** (<code>[PanelDataSCM](#causalis.data_contracts.panel_data_scm.PanelDataSCM)</code>) – Validated long-format panel contract.
- **show_donor_units** (<code>[bool](#bool)</code>) – If True, draw donor trajectories.
- **donor_max_lines** (<code>[int](#int) or None</code>) – Maximum number of donor-unit lines to draw. `None` draws all donors.
- **show_donor_mean** (<code>[bool](#bool)</code>) – If True, draw the donor-pool mean outcome path.
- **donor_alpha** (<code>[float](#float)</code>) – Opacity for donor-unit lines.
- **donor_linewidth** (<code>[float](#float)</code>) – Line width for donor-unit lines.
- **shade_post_period** (<code>[bool](#bool)</code>) – If True, lightly shade the post-treatment region.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Optional path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saved raster outputs.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

###### `placebo`

**Functions:**

- [**placebo_in_space_table**](#causalis.scenarios.synthetic_control.refutation.placebo.placebo_in_space_table) – Build Abadie-style placebo-in-space RMSPE ratio table.
- [**placebo_in_time_table**](#causalis.scenarios.synthetic_control.refutation.placebo.placebo_in_time_table) – Build pre-treatment-only placebo-in-time falsification table.
- [**run_placebo_tests**](#causalis.scenarios.synthetic_control.refutation.placebo.run_placebo_tests) – Run placebo-in-space and placebo-in-time robustness tests.

####### `placebo_in_space_table`

```python
placebo_in_space_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None
) -> pd.DataFrame
```

Build Abadie-style placebo-in-space RMSPE ratio table.

For donor-as-treated placebo fits, the actual treated unit is excluded
from the donor pool to avoid post-treatment contamination.

####### `placebo_in_time_table`

```python
placebo_in_time_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
    pseudo_post_horizon: int | None = None
) -> pd.DataFrame
```

Build pre-treatment-only placebo-in-time falsification table.

####### `run_placebo_tests`

```python
run_placebo_tests(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
    pseudo_post_horizon: int | None = None
) -> Dict[str, pd.DataFrame]
```

Run placebo-in-space and placebo-in-time robustness tests.

###### `placebo_in_space_table`

```python
placebo_in_space_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None
) -> pd.DataFrame
```

Build Abadie-style placebo-in-space RMSPE ratio table.

For donor-as-treated placebo fits, the actual treated unit is excluded
from the donor pool to avoid post-treatment contamination.

###### `placebo_in_time_table`

```python
placebo_in_time_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
    pseudo_post_horizon: int | None = None
) -> pd.DataFrame
```

Build pre-treatment-only placebo-in-time falsification table.

###### `run_placebo_tests`

```python
run_placebo_tests(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
    pseudo_post_horizon: int | None = None
) -> Dict[str, pd.DataFrame]
```

Run placebo-in-space and placebo-in-time robustness tests.

###### `run_scm_diagnostics`

```python
run_scm_diagnostics(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    output_dir: str | Path | None = None,
    filename_prefix: str = "scm_diagnostics",
    pre_tail_k: int = 3,
    dpi: int = 220
) -> pd.DataFrame
```

Run compact SCM diagnostics and return thresholded checks.

###### `run_scm_feasibility`

```python
run_scm_feasibility(paneldata: PanelDataSCM) -> Dict[str, pd.DataFrame]
```

Return core SCM feasibility tables from panel data only (EDA phase).

###### `scm_diagnostics`

**Functions:**

- [**run_scm_diagnostics**](#causalis.scenarios.synthetic_control.refutation.scm_diagnostics.run_scm_diagnostics) – Run compact SCM diagnostics and return thresholded checks.

####### `run_scm_diagnostics`

```python
run_scm_diagnostics(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    output_dir: str | Path | None = None,
    filename_prefix: str = "scm_diagnostics",
    pre_tail_k: int = 3,
    dpi: int = 220
) -> pd.DataFrame
```

Run compact SCM diagnostics and return thresholded checks.

###### `sensitivity`

**Functions:**

- [**leave_one_donor_out_sensitivity**](#causalis.scenarios.synthetic_control.refutation.sensitivity.leave_one_donor_out_sensitivity) – Re-fit ASCM dropping each donor once and report leave-one-donor-out sensitivity.

####### `leave_one_donor_out_sensitivity`

```python
leave_one_donor_out_sensitivity(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None
) -> pd.DataFrame
```

Re-fit ASCM dropping each donor once and report leave-one-donor-out sensitivity.

##### `run_placebo_tests`

```python
run_placebo_tests(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
    pseudo_post_horizon: int | None = None
) -> Dict[str, pd.DataFrame]
```

Run placebo-in-space and placebo-in-time robustness tests.

##### `run_scm_diagnostics`

```python
run_scm_diagnostics(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    output_dir: str | Path | None = None,
    filename_prefix: str = "scm_diagnostics",
    pre_tail_k: int = 3,
    dpi: int = 220
) -> pd.DataFrame
```

Run compact SCM diagnostics and return thresholded checks.

##### `run_scm_feasibility`

```python
run_scm_feasibility(paneldata: PanelDataSCM) -> Dict[str, pd.DataFrame]
```

Return core SCM feasibility tables from panel data only (EDA phase).

#### `unconfoundedness`

**Modules:**

- [**dgp**](#causalis.scenarios.unconfoundedness.dgp) –
- [**gate**](#causalis.scenarios.unconfoundedness.gate) – Group Average Treatment Effect (GATE) estimation utilities.
- [**model**](#causalis.scenarios.unconfoundedness.model) – IRM estimator consuming CausalData.
- [**refutation**](#causalis.scenarios.unconfoundedness.refutation) – Refutation and robustness utilities for Causalis.

**Classes:**

- [**IRM**](#causalis.scenarios.unconfoundedness.IRM) – Interactive Regression Model (IRM) with cross-fitting using CausalData.

##### `IRM`

```python
IRM(
    data: Optional[CausalData] = None,
    ml_g: Any = None,
    ml_m: Any = None,
    *,
    n_folds: int = 5,
    n_rep: int = 1,
    normalize_ipw: bool = False,
    trimming_rule: str = "truncate",
    trimming_threshold: float = 0.01,
    weights: Optional[np.ndarray | Dict[str, Any]] = None,
    relative_baseline_min: float = 1e-08,
    random_state: Optional[int] = None,
    n_jobs: int = 1
) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model (IRM) with cross-fitting using CausalData.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Data container with outcome, binary treatment (0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
- **n_folds** (<code>[int](#int)</code>) – Number of cross-fitting folds.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions of sample splitting. Currently only 1 is supported.
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score. Applied to ATE only.
  For ATTE, normalization is ignored to preserve the canonical ATTE EIF.
- **trimming_rule** (<code>'truncate'</code>) – Trimming approach for propensity scores.
- **trimming_threshold** (<code>[float](#float)</code>) – Threshold for trimming if rule is "truncate".
- **weights** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray) or [Dict](#typing.Dict)\]</code>) – Optional weights.
- If array of shape (n,), used as ATE weights (w). Assumed E[w|X] = w.
- If dict, can contain 'weights' (w) and 'weights_bar' (E[w|X]).
- For ATTE, computed internally (w=D/P(D=1), w_bar=m(X)/P(D=1)).
  Note: If weights depend on treatment or outcome, E[w|X] must be provided for correct sensitivity analysis.
- **relative_baseline_min** (<code>[float](#float)</code>) – Minimum absolute baseline value used for relative effects. If |mu_c| is below this
  threshold, relative estimates are set to NaN with a warning.
- **random_state** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Random seed for fold creation.
- **n_jobs** (<code>[int](#int)</code>) – Number of parallel jobs for fold-level cross-fitting.
  Use `-1` to use all available CPUs.
  Practical guidance:
- Start with `n_jobs=1` for stable, low-contention defaults.
- Increase to `n_jobs=2/4/-1` when cross-fitting is the bottleneck.
- If nuisance learners are already multithreaded (e.g. CatBoost with
  `thread_count=-1`), keep `n_jobs=1` or set learner threads to `1`
  to avoid CPU oversubscription.
- On shared machines, prefer a bounded value (for example `2` or `4`)
  instead of `-1`.

**Functions:**

- [**confint**](#causalis.scenarios.unconfoundedness.IRM.confint) – Compute confidence intervals for the estimated coefficient.
- [**estimate**](#causalis.scenarios.unconfoundedness.IRM.estimate) – Compute treatment effects using stored nuisance predictions.
- [**fit**](#causalis.scenarios.unconfoundedness.IRM.fit) – Fit nuisance models via cross-fitting.
- [**gate**](#causalis.scenarios.unconfoundedness.IRM.gate) – Convenience wrapper for `estimate(score="GATE", ...)`.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.IRM.sensitivity_analysis) – Compute a sensitivity analysis following Chernozhukov et al. (2022).

###### `coef`

```python
coef: np.ndarray
```

Return the estimated coefficient.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The estimated coefficient.

###### `confint`

```python
confint(alpha: float = 0.05) -> pd.DataFrame
```

Compute confidence intervals for the estimated coefficient.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Significance level.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with confidence intervals.

###### `data`

```python
data = data
```

###### `diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary containing 'm_hat', 'g0_hat', 'g1_hat', and 'folds'.

###### `estimate`

```python
estimate(
    score: str = "ATE",
    alpha: float = 0.05,
    diagnostic_data: bool = True,
    groups: Optional[pd.DataFrame | pd.Series] = None,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
) -> CausalEstimate | GateEstimate
```

Compute treatment effects using stored nuisance predictions.

**Parameters:**

- **score** (<code>('ATE', 'ATTE', 'GATE')</code>) – Target estimand.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.
- **groups** (<code>[Optional](#typing.Optional)\[[DataFrame](#pandas.DataFrame) | [Series](#pandas.Series)\]</code>) – Group labels/indicators for `score="GATE"`.
  If None, fallback to `self.data.gate_groups` when present.
- **cov_type** (<code>('HC0', 'HC1', 'HC2', 'HC3')</code>) – Robust covariance type for `score="GATE"` inference.
- **cov_kwds** (<code>[Optional](#typing.Optional)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]</code>) – Additional covariance keyword arguments passed to statsmodels for `score="GATE"`.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate) or [GateEstimate](#causalis.data_contracts.gate_estimate.GateEstimate)</code> – Result container for the estimated effect.

###### `fit`

```python
fit(data: Optional[CausalData] = None) -> 'IRM'
```

Fit nuisance models via cross-fitting.

**Parameters:**

- **data** (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.dgp.causaldata.CausalData)\]</code>) – CausalData container. If None, uses self.data.

**Returns:**

- **self** (<code>[IRM](#causalis.scenarios.unconfoundedness.model.IRM)</code>) – Fitted estimator.

###### `gate`

```python
gate(
    groups: pd.DataFrame | pd.Series,
    alpha: float = 0.05,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
    diagnostic_data: bool = True,
) -> GateEstimate
```

Convenience wrapper for `estimate(score="GATE", ...)`.

###### `ml_g`

```python
ml_g = ml_g
```

###### `ml_m`

```python
ml_m = ml_m
```

###### `n_folds`

```python
n_folds = int(n_folds)
```

###### `n_jobs`

```python
n_jobs = int(n_jobs)
```

###### `n_rep`

```python
n_rep = int(n_rep)
```

###### `normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

###### `normalize_ipw_effective_`

```python
normalize_ipw_effective_ = bool(normalize_ipw)
```

###### `orth_signal`

```python
orth_signal: np.ndarray
```

Return the cross-fitted orthogonal signal (psi_b).

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The orthogonal signal.

###### `pvalues`

```python
pvalues: np.ndarray
```

Return the p-values for the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The p-values.

###### `random_state`

```python
random_state = random_state
```

###### `relative_baseline_min`

```python
relative_baseline_min = float(relative_baseline_min)
```

###### `score`

```python
score = 'ATE'
```

###### `se`

```python
se: np.ndarray
```

Return the standard error of the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The standard error.

###### `sensitivity_analysis`

```python
sensitivity_analysis(
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
) -> "IRM"
```

Compute a sensitivity analysis following Chernozhukov et al. (2022).

**Parameters:**

- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome equation (R^2 form, R_Y^2; converted to odds form internally).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment equation (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation between unobserved components.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level for CI bounds.

###### `summary`

```python
summary: pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The results summary.

###### `trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

###### `trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

###### `weights`

```python
weights = weights
```

##### `dgp`

**Functions:**

- [**generate_obs_hte_26**](#causalis.scenarios.unconfoundedness.dgp.generate_obs_hte_26) – Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
- [**generate_obs_hte_26_rich**](#causalis.scenarios.unconfoundedness.dgp.generate_obs_hte_26_rich) – Observational dataset with richer confounding, nonlinear outcome model,
- [**generate_obs_hte_binary_26**](#causalis.scenarios.unconfoundedness.dgp.generate_obs_hte_binary_26) – Observational binary-outcome dataset with nonlinear confounding and
- [**obs_linear_26_dataset**](#causalis.scenarios.unconfoundedness.dgp.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.

###### `generate_obs_hte_26`

```python
generate_obs_hte_26(
    n: int = 10000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, CausalData]
```

Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
and a heterogeneous (nonlinear) treatment effect tau(X).
Based on the scenario in notebooks/cases/dml_atte.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

###### `generate_obs_hte_26_rich`

```python
generate_obs_hte_26_rich(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, CausalData]
```

Observational dataset with richer confounding, nonlinear outcome model,
nonlinear treatment assignment, and heterogeneous treatment effects.
Adds additional realistic covariates and dependencies to mimic real data.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

###### `generate_obs_hte_binary_26`

```python
generate_obs_hte_binary_26(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, CausalData]
```

Observational binary-outcome dataset with nonlinear confounding and
heterogeneous treatment effects.

This scenario follows the structure of `generate_obs_hte_26_rich`, but uses
a binary outcome model and a modified confounder set.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

###### `obs_linear_26_dataset`

```python
obs_linear_26_dataset(
    n: int = 10000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
)
```

A pre-configured observational linear dataset with 5 standard confounders.
Based on the scenario in docs/cases/dml_ate.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

##### `gate`

Group Average Treatment Effect (GATE) estimation utilities.

**Functions:**

- [**estimate_gate_from_irm**](#causalis.scenarios.unconfoundedness.gate.estimate_gate_from_irm) – Estimate strict GATEs from a fitted IRM via groupwise closed-form robust inference.

###### `estimate_gate_from_irm`

```python
estimate_gate_from_irm(
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series],
    alpha: float = 0.05,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
    diagnostic_data: bool = True,
) -> GateEstimate
```

Estimate strict GATEs from a fitted IRM via groupwise closed-form robust inference.

Groups are assumed to be pre-specified, pre-treatment, mutually exclusive, and exhaustive.

##### `model`

IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE/GATE scores.

**Classes:**

- [**IRM**](#causalis.scenarios.unconfoundedness.model.IRM) – Interactive Regression Model (IRM) with cross-fitting using CausalData.

###### `HAS_CATBOOST`

```python
HAS_CATBOOST = True
```

###### `IRM`

```python
IRM(
    data: Optional[CausalData] = None,
    ml_g: Any = None,
    ml_m: Any = None,
    *,
    n_folds: int = 5,
    n_rep: int = 1,
    normalize_ipw: bool = False,
    trimming_rule: str = "truncate",
    trimming_threshold: float = 0.01,
    weights: Optional[np.ndarray | Dict[str, Any]] = None,
    relative_baseline_min: float = 1e-08,
    random_state: Optional[int] = None,
    n_jobs: int = 1
) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model (IRM) with cross-fitting using CausalData.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Data container with outcome, binary treatment (0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
- **n_folds** (<code>[int](#int)</code>) – Number of cross-fitting folds.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions of sample splitting. Currently only 1 is supported.
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score. Applied to ATE only.
  For ATTE, normalization is ignored to preserve the canonical ATTE EIF.
- **trimming_rule** (<code>'truncate'</code>) – Trimming approach for propensity scores.
- **trimming_threshold** (<code>[float](#float)</code>) – Threshold for trimming if rule is "truncate".
- **weights** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray) or [Dict](#typing.Dict)\]</code>) – Optional weights.
- If array of shape (n,), used as ATE weights (w). Assumed E[w|X] = w.
- If dict, can contain 'weights' (w) and 'weights_bar' (E[w|X]).
- For ATTE, computed internally (w=D/P(D=1), w_bar=m(X)/P(D=1)).
  Note: If weights depend on treatment or outcome, E[w|X] must be provided for correct sensitivity analysis.
- **relative_baseline_min** (<code>[float](#float)</code>) – Minimum absolute baseline value used for relative effects. If |mu_c| is below this
  threshold, relative estimates are set to NaN with a warning.
- **random_state** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Random seed for fold creation.
- **n_jobs** (<code>[int](#int)</code>) – Number of parallel jobs for fold-level cross-fitting.
  Use `-1` to use all available CPUs.
  Practical guidance:
- Start with `n_jobs=1` for stable, low-contention defaults.
- Increase to `n_jobs=2/4/-1` when cross-fitting is the bottleneck.
- If nuisance learners are already multithreaded (e.g. CatBoost with
  `thread_count=-1`), keep `n_jobs=1` or set learner threads to `1`
  to avoid CPU oversubscription.
- On shared machines, prefer a bounded value (for example `2` or `4`)
  instead of `-1`.

**Functions:**

- [**confint**](#causalis.scenarios.unconfoundedness.model.IRM.confint) – Compute confidence intervals for the estimated coefficient.
- [**estimate**](#causalis.scenarios.unconfoundedness.model.IRM.estimate) – Compute treatment effects using stored nuisance predictions.
- [**fit**](#causalis.scenarios.unconfoundedness.model.IRM.fit) – Fit nuisance models via cross-fitting.
- [**gate**](#causalis.scenarios.unconfoundedness.model.IRM.gate) – Convenience wrapper for `estimate(score="GATE", ...)`.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.model.IRM.sensitivity_analysis) – Compute a sensitivity analysis following Chernozhukov et al. (2022).

####### `coef`

```python
coef: np.ndarray
```

Return the estimated coefficient.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The estimated coefficient.

####### `confint`

```python
confint(alpha: float = 0.05) -> pd.DataFrame
```

Compute confidence intervals for the estimated coefficient.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Significance level.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with confidence intervals.

####### `data`

```python
data = data
```

####### `diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary containing 'm_hat', 'g0_hat', 'g1_hat', and 'folds'.

####### `estimate`

```python
estimate(
    score: str = "ATE",
    alpha: float = 0.05,
    diagnostic_data: bool = True,
    groups: Optional[pd.DataFrame | pd.Series] = None,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
) -> CausalEstimate | GateEstimate
```

Compute treatment effects using stored nuisance predictions.

**Parameters:**

- **score** (<code>('ATE', 'ATTE', 'GATE')</code>) – Target estimand.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.
- **groups** (<code>[Optional](#typing.Optional)\[[DataFrame](#pandas.DataFrame) | [Series](#pandas.Series)\]</code>) – Group labels/indicators for `score="GATE"`.
  If None, fallback to `self.data.gate_groups` when present.
- **cov_type** (<code>('HC0', 'HC1', 'HC2', 'HC3')</code>) – Robust covariance type for `score="GATE"` inference.
- **cov_kwds** (<code>[Optional](#typing.Optional)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]</code>) – Additional covariance keyword arguments passed to statsmodels for `score="GATE"`.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate) or [GateEstimate](#causalis.data_contracts.gate_estimate.GateEstimate)</code> – Result container for the estimated effect.

####### `fit`

```python
fit(data: Optional[CausalData] = None) -> 'IRM'
```

Fit nuisance models via cross-fitting.

**Parameters:**

- **data** (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.dgp.causaldata.CausalData)\]</code>) – CausalData container. If None, uses self.data.

**Returns:**

- **self** (<code>[IRM](#causalis.scenarios.unconfoundedness.model.IRM)</code>) – Fitted estimator.

####### `gate`

```python
gate(
    groups: pd.DataFrame | pd.Series,
    alpha: float = 0.05,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
    diagnostic_data: bool = True,
) -> GateEstimate
```

Convenience wrapper for `estimate(score="GATE", ...)`.

####### `ml_g`

```python
ml_g = ml_g
```

####### `ml_m`

```python
ml_m = ml_m
```

####### `n_folds`

```python
n_folds = int(n_folds)
```

####### `n_jobs`

```python
n_jobs = int(n_jobs)
```

####### `n_rep`

```python
n_rep = int(n_rep)
```

####### `normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

####### `normalize_ipw_effective_`

```python
normalize_ipw_effective_ = bool(normalize_ipw)
```

####### `orth_signal`

```python
orth_signal: np.ndarray
```

Return the cross-fitted orthogonal signal (psi_b).

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The orthogonal signal.

####### `pvalues`

```python
pvalues: np.ndarray
```

Return the p-values for the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The p-values.

####### `random_state`

```python
random_state = random_state
```

####### `relative_baseline_min`

```python
relative_baseline_min = float(relative_baseline_min)
```

####### `score`

```python
score = 'ATE'
```

####### `se`

```python
se: np.ndarray
```

Return the standard error of the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The standard error.

####### `sensitivity_analysis`

```python
sensitivity_analysis(
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
) -> "IRM"
```

Compute a sensitivity analysis following Chernozhukov et al. (2022).

**Parameters:**

- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome equation (R^2 form, R_Y^2; converted to odds form internally).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment equation (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation between unobserved components.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level for CI bounds.

####### `summary`

```python
summary: pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The results summary.

####### `trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

####### `trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

####### `weights`

```python
weights = weights
```

##### `refutation`

Refutation and robustness utilities for Causalis.

Importing this package exposes the public functions from all refutation
submodules (overlap, score, unconfoundedness) so you can access
commonly used helpers directly via `causalis.refutation`.

**Modules:**

- [**overlap**](#causalis.scenarios.unconfoundedness.refutation.overlap) –
- [**score**](#causalis.scenarios.unconfoundedness.refutation.score) –
- [**unconfoundedness**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness) –

**Functions:**

- [**get_sensitivity_summary**](#causalis.scenarios.unconfoundedness.refutation.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**interpret_sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.interpret_sensitivity_analysis) – Run sensitivity analysis and return a structured interpretation.
- [**plot_influence_instability**](#causalis.scenarios.unconfoundedness.refutation.plot_influence_instability) – Plot instability diagnostics for per-unit score (EIF moment).
- [**plot_m_overlap**](#causalis.scenarios.unconfoundedness.refutation.plot_m_overlap) – Overlap plot for m(x)=P(D=1|X) with high-res rendering.
- [**plot_propensity_reliability**](#causalis.scenarios.unconfoundedness.refutation.plot_propensity_reliability) – Plot a propensity calibration reliability diagram.
- [**plot_residual_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.plot_residual_diagnostics) – Plot residual diagnostics for nuisance models.
- [**run_overlap_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_overlap_diagnostics) – Run overlap diagnostics from `CausalData` and `CausalEstimate`.
- [**run_score_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_score_diagnostics) – Run score diagnostics from `CausalData` and `CausalEstimate`.
- [**run_unconfoundedness_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_unconfoundedness_diagnostics) – Run unconfoundedness diagnostics from `CausalData` and `CausalEstimate`.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.sensitivity_analysis) – Compute bias-aware components and cache them.
- [**sensitivity_benchmark**](#causalis.scenarios.unconfoundedness.refutation.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model

###### `get_sensitivity_summary`

```python
get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any, *, label: Optional[str] = None
) -> Optional[str]
```

Render a single, unified bias-aware summary string.

If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **label** (<code>[str](#str)</code>) – The label for the estimand.

**Returns:**

- <code>[Optional](#typing.Optional)\[[str](#str)\]</code> – Formatted summary string or None if extraction fails.

###### `interpret_sensitivity_analysis`

```python
interpret_sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False
) -> Dict[str, Any]
```

Run sensitivity analysis and return a structured interpretation.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome residual confounding strength.
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment residual confounding strength.
- **rho** (<code>[float](#float)</code>) – Correlation parameter for unobserved confounding.
- **H0** (<code>[float](#float)</code>) – Null hypothesis used for significance checks.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr in the quadratic sensitivity combination.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary with:
  - raw: the output of `sensitivity_analysis(...)`
  - interpretation: machine-readable interpretation fields
  - summary: compact human-readable interpretation

###### `overlap`

**Modules:**

- [**overlap_plot**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_plot) –
- [**overlap_validation**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation) – Overlap diagnostics focused on positivity and propensity calibration.
- [**reliability_plot**](#causalis.scenarios.unconfoundedness.refutation.overlap.reliability_plot) – Reliability diagram for propensity calibration diagnostics.

**Functions:**

- [**plot_m_overlap**](#causalis.scenarios.unconfoundedness.refutation.overlap.plot_m_overlap) – Overlap plot for m(x)=P(D=1|X) with high-res rendering.
- [**plot_propensity_reliability**](#causalis.scenarios.unconfoundedness.refutation.overlap.plot_propensity_reliability) – Plot a propensity calibration reliability diagram.
- [**run_overlap_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.overlap.run_overlap_diagnostics) – Run overlap diagnostics from `CausalData` and `CausalEstimate`.

####### `overlap_plot`

**Functions:**

- [**plot_m_overlap**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_plot.plot_m_overlap) – Overlap plot for m(x)=P(D=1|X) with high-res rendering.

######## `plot_m_overlap`

```python
plot_m_overlap(
    diag: Union[UnconfoundednessDiagnosticData, CausalEstimate, dict],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
) -> plt.Figure
```

Overlap plot for m(x)=P(D=1|X) with high-res rendering.

- x in [0,1]
- Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
- Uses Matplotlib default colors unless color_t/color_c are provided

**Parameters:**

- **diag** (<code>[UnconfoundednessDiagnosticData](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData) or [CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Diagnostic data directly, or an estimate containing diagnostic_data with m_hat and d.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to clip for KDE range.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Histogram bins.
- **kde** (<code>[bool](#bool)</code>) – Whether to show KDE.
- **shade_overlap** (<code>[bool](#bool)</code>) – Whether to shade the overlap area.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.
- **color_t** (<code>[color](#color)</code>) – Color for treated group.
- **color_c** (<code>[color](#color)</code>) – Color for control group.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `overlap_validation`

Overlap diagnostics focused on positivity and propensity calibration.

**Functions:**

- [**run_overlap_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.run_overlap_diagnostics) – Run overlap diagnostics from `CausalData` and `CausalEstimate`.

######## `run_overlap_diagnostics`

```python
run_overlap_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    n_bins: int = 10,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05
) -> Dict[str, Any]
```

Run overlap diagnostics from `CausalData` and `CausalEstimate`.

####### `plot_m_overlap`

```python
plot_m_overlap(
    diag: Union[UnconfoundednessDiagnosticData, CausalEstimate, dict],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
) -> plt.Figure
```

Overlap plot for m(x)=P(D=1|X) with high-res rendering.

- x in [0,1]
- Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
- Uses Matplotlib default colors unless color_t/color_c are provided

**Parameters:**

- **diag** (<code>[UnconfoundednessDiagnosticData](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData) or [CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Diagnostic data directly, or an estimate containing diagnostic_data with m_hat and d.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to clip for KDE range.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Histogram bins.
- **kde** (<code>[bool](#bool)</code>) – Whether to show KDE.
- **shade_overlap** (<code>[bool](#bool)</code>) – Whether to shade the overlap area.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.
- **color_t** (<code>[color](#color)</code>) – Color for treated group.
- **color_c** (<code>[color](#color)</code>) – Color for control group.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `plot_propensity_reliability`

```python
plot_propensity_reliability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    n_bins: int = 10,
    show_recalibration: bool = True,
    annotate_metrics: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7.2, 6.2),
    dpi: int = 220,
    font_scale: float = 1.1,
    point_color: Optional[Any] = None,
    diagonal_color: Any = "0.35",
    recalibration_color: Any = "C1",
    min_marker_size: float = 35.0,
    marker_size_scale: float = 250.0,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot a propensity calibration reliability diagram.

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`; optionally `m_hat_raw`, `d`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for treatment `d` when not stored in diagnostic data.
- **n_bins** (<code>[int](#int)</code>) – Number of calibration bins used to build the reliability table.
- **show_recalibration** (<code>[bool](#bool)</code>) – Overlay logistic recalibration curve
  `sigmoid(alpha + beta * logit(p))` when parameters are available.
- **annotate_metrics** (<code>[bool](#bool)</code>) – Annotate ECE and logistic recalibration parameters on the figure.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **point_color** (<code>[color](#color)</code>) – Marker color for binned reliability points.
- **diagonal_color** (<code>[color](#color)</code>) – Color for the perfect calibration diagonal.
- **recalibration_color** (<code>[color](#color)</code>) – Color for the logistic recalibration curve.
- **min_marker_size** (<code>[float](#float)</code>) – Base marker area for non-empty bins.
- **marker_size_scale** (<code>[float](#float)</code>) – Additional marker area scaled by bin count share.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `reliability_plot`

Reliability diagram for propensity calibration diagnostics.

**Functions:**

- [**plot_propensity_reliability**](#causalis.scenarios.unconfoundedness.refutation.overlap.reliability_plot.plot_propensity_reliability) – Plot a propensity calibration reliability diagram.

######## `plot_propensity_reliability`

```python
plot_propensity_reliability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    n_bins: int = 10,
    show_recalibration: bool = True,
    annotate_metrics: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7.2, 6.2),
    dpi: int = 220,
    font_scale: float = 1.1,
    point_color: Optional[Any] = None,
    diagonal_color: Any = "0.35",
    recalibration_color: Any = "C1",
    min_marker_size: float = 35.0,
    marker_size_scale: float = 250.0,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot a propensity calibration reliability diagram.

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`; optionally `m_hat_raw`, `d`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for treatment `d` when not stored in diagnostic data.
- **n_bins** (<code>[int](#int)</code>) – Number of calibration bins used to build the reliability table.
- **show_recalibration** (<code>[bool](#bool)</code>) – Overlay logistic recalibration curve
  `sigmoid(alpha + beta * logit(p))` when parameters are available.
- **annotate_metrics** (<code>[bool](#bool)</code>) – Annotate ECE and logistic recalibration parameters on the figure.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **point_color** (<code>[color](#color)</code>) – Marker color for binned reliability points.
- **diagonal_color** (<code>[color](#color)</code>) – Color for the perfect calibration diagonal.
- **recalibration_color** (<code>[color](#color)</code>) – Color for the logistic recalibration curve.
- **min_marker_size** (<code>[float](#float)</code>) – Base marker area for non-empty bins.
- **marker_size_scale** (<code>[float](#float)</code>) – Additional marker area scaled by bin count share.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `run_overlap_diagnostics`

```python
run_overlap_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    n_bins: int = 10,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05
) -> Dict[str, Any]
```

Run overlap diagnostics from `CausalData` and `CausalEstimate`.

###### `plot_influence_instability`

```python
plot_influence_instability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    trimming_threshold: Optional[float] = None,
    use_estimator_psi: bool = True,
    include_ipw: bool = True,
    bins: Any = "fd",
    log_hist: bool = False,
    scatter_log_y: bool = True,
    top_k: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot instability diagnostics for per-unit score (EIF moment).

<details class="panels" open markdown="1">
<summary>Panels</summary>

1. Histogram of `|psi_i|` (optional log-x scale).
1. Scatter of `|psi_i|` versus clipped propensity `m_i`.
1. (optional) Histogram of IPW terms.
1. (optional) ESS ratio bars for treated/control weights.

</details>

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `y`, `d`, `g1_hat`, `psi`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for `y` and `d` if not stored in diagnostic data.
- **trimming_threshold** (<code>[float](#float)</code>) – Propensity clipping threshold. If omitted, uses diagnostic/model defaults.
- **use_estimator_psi** (<code>[bool](#bool)</code>) – Use estimator-provided `diagnostic_data.psi` when available; otherwise reconstruct score.
- **include_ipw** (<code>[bool](#bool)</code>) – Add IPW-term histogram and ESS ratio bar panels.
- **bins** (<code>[Any](#typing.Any)</code>) – Histogram bins for non-log histograms.
- **log_hist** (<code>[bool](#bool)</code>) – Use log-scaled x-axis bins for `|psi_i|` histogram when possible.
- **scatter_log_y** (<code>[bool](#bool)</code>) – Plot `|psi_i|` on log scale in the scatter panel.
- **top_k** (<code>[int](#int)</code>) – Highlight top-k largest `|psi_i|` in the scatter panel.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size. Defaults to `(12, 8)` with IPW panels, `(11, 4.6)` otherwise.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

###### `plot_m_overlap`

```python
plot_m_overlap(
    diag: Union[UnconfoundednessDiagnosticData, CausalEstimate, dict],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
) -> plt.Figure
```

Overlap plot for m(x)=P(D=1|X) with high-res rendering.

- x in [0,1]
- Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
- Uses Matplotlib default colors unless color_t/color_c are provided

**Parameters:**

- **diag** (<code>[UnconfoundednessDiagnosticData](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData) or [CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Diagnostic data directly, or an estimate containing diagnostic_data with m_hat and d.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to clip for KDE range.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Histogram bins.
- **kde** (<code>[bool](#bool)</code>) – Whether to show KDE.
- **shade_overlap** (<code>[bool](#bool)</code>) – Whether to shade the overlap area.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.
- **color_t** (<code>[color](#color)</code>) – Color for treated group.
- **color_c** (<code>[color](#color)</code>) – Color for control group.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

###### `plot_propensity_reliability`

```python
plot_propensity_reliability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    n_bins: int = 10,
    show_recalibration: bool = True,
    annotate_metrics: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7.2, 6.2),
    dpi: int = 220,
    font_scale: float = 1.1,
    point_color: Optional[Any] = None,
    diagonal_color: Any = "0.35",
    recalibration_color: Any = "C1",
    min_marker_size: float = 35.0,
    marker_size_scale: float = 250.0,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot a propensity calibration reliability diagram.

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`; optionally `m_hat_raw`, `d`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for treatment `d` when not stored in diagnostic data.
- **n_bins** (<code>[int](#int)</code>) – Number of calibration bins used to build the reliability table.
- **show_recalibration** (<code>[bool](#bool)</code>) – Overlay logistic recalibration curve
  `sigmoid(alpha + beta * logit(p))` when parameters are available.
- **annotate_metrics** (<code>[bool](#bool)</code>) – Annotate ECE and logistic recalibration parameters on the figure.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **point_color** (<code>[color](#color)</code>) – Marker color for binned reliability points.
- **diagonal_color** (<code>[color](#color)</code>) – Color for the perfect calibration diagonal.
- **recalibration_color** (<code>[color](#color)</code>) – Color for the logistic recalibration curve.
- **min_marker_size** (<code>[float](#float)</code>) – Base marker area for non-empty bins.
- **marker_size_scale** (<code>[float](#float)</code>) – Additional marker area scaled by bin count share.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

###### `plot_residual_diagnostics`

```python
plot_residual_diagnostics(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    clip_propensity: float = 1e-06,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot residual diagnostics for nuisance models.

<details class="panels" open markdown="1">
<summary>Panels</summary>

1. Treated-only: `u1 = y - g1` vs `g1`.
1. Control-only: `u0 = y - g0` vs `g0`.
1. Binned calibration error: `E[d - m | m in bin]` vs binned `m`.

</details>

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `g1_hat`, `y`, `d`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for `y` and `d` when missing in diagnostic data.
- **clip_propensity** (<code>[float](#float)</code>) – Clipping epsilon for propensity values in the treatment-residual panel.
- **n_bins** (<code>[int](#int)</code>) – Number of quantile bins for the binned-mean trend overlays.
- **marker_size** (<code>[float](#float)</code>) – Scatter marker size.
- **alpha** (<code>[float](#float)</code>) – Scatter opacity.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

###### `run_overlap_diagnostics`

```python
run_overlap_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    n_bins: int = 10,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05
) -> Dict[str, Any]
```

Run overlap diagnostics from `CausalData` and `CausalEstimate`.

###### `run_score_diagnostics`

```python
run_score_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run score diagnostics from `CausalData` and `CausalEstimate`.

###### `run_unconfoundedness_diagnostics`

```python
run_unconfoundedness_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run unconfoundedness diagnostics from `CausalData` and `CausalEstimate`.

###### `score`

**Modules:**

- [**influence_plot**](#causalis.scenarios.unconfoundedness.refutation.score.influence_plot) – Influence/instability plots for score-based diagnostics.
- [**residual_plots**](#causalis.scenarios.unconfoundedness.refutation.score.residual_plots) – Residual diagnostic plots for nuisance models g0/g1 and m.
- [**score_validation**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation) – Score diagnostics focused on orthogonality and EIF stability.

**Functions:**

- [**plot_influence_instability**](#causalis.scenarios.unconfoundedness.refutation.score.plot_influence_instability) – Plot instability diagnostics for per-unit score (EIF moment).
- [**plot_residual_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.score.plot_residual_diagnostics) – Plot residual diagnostics for nuisance models.
- [**run_score_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.score.run_score_diagnostics) – Run score diagnostics from `CausalData` and `CausalEstimate`.

####### `influence_plot`

Influence/instability plots for score-based diagnostics.

**Functions:**

- [**plot_influence_instability**](#causalis.scenarios.unconfoundedness.refutation.score.influence_plot.plot_influence_instability) – Plot instability diagnostics for per-unit score (EIF moment).

######## `plot_influence_instability`

```python
plot_influence_instability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    trimming_threshold: Optional[float] = None,
    use_estimator_psi: bool = True,
    include_ipw: bool = True,
    bins: Any = "fd",
    log_hist: bool = False,
    scatter_log_y: bool = True,
    top_k: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot instability diagnostics for per-unit score (EIF moment).

<details class="panels" open markdown="1">
<summary>Panels</summary>

1. Histogram of `|psi_i|` (optional log-x scale).
1. Scatter of `|psi_i|` versus clipped propensity `m_i`.
1. (optional) Histogram of IPW terms.
1. (optional) ESS ratio bars for treated/control weights.

</details>

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `y`, `d`, `g1_hat`, `psi`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for `y` and `d` if not stored in diagnostic data.
- **trimming_threshold** (<code>[float](#float)</code>) – Propensity clipping threshold. If omitted, uses diagnostic/model defaults.
- **use_estimator_psi** (<code>[bool](#bool)</code>) – Use estimator-provided `diagnostic_data.psi` when available; otherwise reconstruct score.
- **include_ipw** (<code>[bool](#bool)</code>) – Add IPW-term histogram and ESS ratio bar panels.
- **bins** (<code>[Any](#typing.Any)</code>) – Histogram bins for non-log histograms.
- **log_hist** (<code>[bool](#bool)</code>) – Use log-scaled x-axis bins for `|psi_i|` histogram when possible.
- **scatter_log_y** (<code>[bool](#bool)</code>) – Plot `|psi_i|` on log scale in the scatter panel.
- **top_k** (<code>[int](#int)</code>) – Highlight top-k largest `|psi_i|` in the scatter panel.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size. Defaults to `(12, 8)` with IPW panels, `(11, 4.6)` otherwise.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `plot_influence_instability`

```python
plot_influence_instability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    trimming_threshold: Optional[float] = None,
    use_estimator_psi: bool = True,
    include_ipw: bool = True,
    bins: Any = "fd",
    log_hist: bool = False,
    scatter_log_y: bool = True,
    top_k: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot instability diagnostics for per-unit score (EIF moment).

<details class="panels" open markdown="1">
<summary>Panels</summary>

1. Histogram of `|psi_i|` (optional log-x scale).
1. Scatter of `|psi_i|` versus clipped propensity `m_i`.
1. (optional) Histogram of IPW terms.
1. (optional) ESS ratio bars for treated/control weights.

</details>

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `y`, `d`, `g1_hat`, `psi`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for `y` and `d` if not stored in diagnostic data.
- **trimming_threshold** (<code>[float](#float)</code>) – Propensity clipping threshold. If omitted, uses diagnostic/model defaults.
- **use_estimator_psi** (<code>[bool](#bool)</code>) – Use estimator-provided `diagnostic_data.psi` when available; otherwise reconstruct score.
- **include_ipw** (<code>[bool](#bool)</code>) – Add IPW-term histogram and ESS ratio bar panels.
- **bins** (<code>[Any](#typing.Any)</code>) – Histogram bins for non-log histograms.
- **log_hist** (<code>[bool](#bool)</code>) – Use log-scaled x-axis bins for `|psi_i|` histogram when possible.
- **scatter_log_y** (<code>[bool](#bool)</code>) – Plot `|psi_i|` on log scale in the scatter panel.
- **top_k** (<code>[int](#int)</code>) – Highlight top-k largest `|psi_i|` in the scatter panel.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size. Defaults to `(12, 8)` with IPW panels, `(11, 4.6)` otherwise.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `plot_residual_diagnostics`

```python
plot_residual_diagnostics(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    clip_propensity: float = 1e-06,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot residual diagnostics for nuisance models.

<details class="panels" open markdown="1">
<summary>Panels</summary>

1. Treated-only: `u1 = y - g1` vs `g1`.
1. Control-only: `u0 = y - g0` vs `g0`.
1. Binned calibration error: `E[d - m | m in bin]` vs binned `m`.

</details>

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `g1_hat`, `y`, `d`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for `y` and `d` when missing in diagnostic data.
- **clip_propensity** (<code>[float](#float)</code>) – Clipping epsilon for propensity values in the treatment-residual panel.
- **n_bins** (<code>[int](#int)</code>) – Number of quantile bins for the binned-mean trend overlays.
- **marker_size** (<code>[float](#float)</code>) – Scatter marker size.
- **alpha** (<code>[float](#float)</code>) – Scatter opacity.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `residual_plots`

Residual diagnostic plots for nuisance models g0/g1 and m.

**Functions:**

- [**plot_residual_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.score.residual_plots.plot_residual_diagnostics) – Plot residual diagnostics for nuisance models.

######## `plot_residual_diagnostics`

```python
plot_residual_diagnostics(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    clip_propensity: float = 1e-06,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.1,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False
) -> plt.Figure
```

Plot residual diagnostics for nuisance models.

<details class="panels" open markdown="1">
<summary>Panels</summary>

1. Treated-only: `u1 = y - g1` vs `g1`.
1. Control-only: `u0 = y - g0` vs `g0`.
1. Binned calibration error: `E[d - m | m in bin]` vs binned `m`.

</details>

**Parameters:**

- **estimate** (<code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code>) – Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `g1_hat`, `y`, `d`).
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Optional fallback source for `y` and `d` when missing in diagnostic data.
- **clip_propensity** (<code>[float](#float)</code>) – Clipping epsilon for propensity values in the treatment-residual panel.
- **n_bins** (<code>[int](#int)</code>) – Number of quantile bins for the binned-mean trend overlays.
- **marker_size** (<code>[float](#float)</code>) – Scatter marker size.
- **alpha** (<code>[float](#float)</code>) – Scatter opacity.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `run_score_diagnostics`

```python
run_score_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run score diagnostics from `CausalData` and `CausalEstimate`.

####### `score_validation`

Score diagnostics focused on orthogonality and EIF stability.

**Functions:**

- [**run_score_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.run_score_diagnostics) – Run score diagnostics from `CausalData` and `CausalEstimate`.

######## `run_score_diagnostics`

```python
run_score_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run score diagnostics from `CausalData` and `CausalEstimate`.

###### `sensitivity_analysis`

```python
sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False
) -> Dict[str, Any]
```

Compute bias-aware components and cache them.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (R^2 form, R_Y^2; converted to odds form internally).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr in the quadratic combination of sensitivity components.
  If True and m_alpha/rr are available, the bias bound is computed via the
  per-unit quadratic form and RV/RVa are not reported.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results:
  - theta, se, alpha, z
  - sampling_ci
  - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
  - bias_aware_ci = faithful CI for the bounds
  - max_bias and components (sigma2, nu2)
  - params (r2_y, r2_d, rho, use_signed_rr)

###### `sensitivity_benchmark`

```python
sensitivity_benchmark(
    effect_estimation: Dict[str, Any],
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing r2_y, r2_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Legacy name for additional keyword arguments passed to `IRM.estimate(...)`
  on the short model. If `score` is omitted, ATE/ATTE is inferred from
  the supplied estimate/model, and defaults to ATE.
  If `diagnostic_data` is omitted, it defaults to `False` for faster benchmarking.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- r2_y, r2_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

###### `unconfoundedness`

**Modules:**

- [**sensitivity**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity) – Sensitivity functions refactored into a dedicated module.
- [**unconfoundedness_validation**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation) – Unconfoundedness diagnostics focused on covariate balance (SMD).

**Functions:**

- [**compute_bias_aware_ci**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.compute_bias_aware_ci) – Compute bias-aware confidence intervals.
- [**get_sensitivity_summary**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**run_unconfoundedness_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.run_unconfoundedness_diagnostics) – Run unconfoundedness diagnostics from `CausalData` and `CausalEstimate`.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity_analysis) – Compute bias-aware components and cache them.
- [**sensitivity_benchmark**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model

####### `compute_bias_aware_ci`

```python
compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False
) -> Dict[str, Any]
```

Compute bias-aware confidence intervals.

Returns a dict with:

- theta, se, alpha, z
- sampling_ci
- theta_bounds_cofounding = [theta_lower, theta_upper] = theta ± bound_width
- bias_aware_ci = \[theta - (bound_width + z*se), theta + (bound_width + z*se)\]
- max_bias_base, max_bias, bound_width and components (sigma2, nu2)

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (R^2 form, R_Y^2).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr in the quadratic combination of sensitivity components.
  If True and m_alpha/rr are available, the bias bound is computed via the
  per-unit quadratic form and RV/RVa are not reported.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results.

####### `get_sensitivity_summary`

```python
get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any, *, label: Optional[str] = None
) -> Optional[str]
```

Render a single, unified bias-aware summary string.

If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **label** (<code>[str](#str)</code>) – The label for the estimand.

**Returns:**

- <code>[Optional](#typing.Optional)\[[str](#str)\]</code> – Formatted summary string or None if extraction fails.

####### `run_unconfoundedness_diagnostics`

```python
run_unconfoundedness_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run unconfoundedness diagnostics from `CausalData` and `CausalEstimate`.

####### `sensitivity`

Sensitivity functions refactored into a dedicated module.

This module centralizes bias-aware sensitivity helpers and the public
entry points used by refutation utilities for unconfoundedness.

**Functions:**

- [**get_sensitivity_summary**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**interpret_sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity.interpret_sensitivity_analysis) – Run sensitivity analysis and return a structured interpretation.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity.sensitivity_analysis) – Compute bias-aware components and cache them.
- [**sensitivity_benchmark**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model

######## `get_sensitivity_summary`

```python
get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any, *, label: Optional[str] = None
) -> Optional[str]
```

Render a single, unified bias-aware summary string.

If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **label** (<code>[str](#str)</code>) – The label for the estimand.

**Returns:**

- <code>[Optional](#typing.Optional)\[[str](#str)\]</code> – Formatted summary string or None if extraction fails.

######## `interpret_sensitivity_analysis`

```python
interpret_sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False
) -> Dict[str, Any]
```

Run sensitivity analysis and return a structured interpretation.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome residual confounding strength.
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment residual confounding strength.
- **rho** (<code>[float](#float)</code>) – Correlation parameter for unobserved confounding.
- **H0** (<code>[float](#float)</code>) – Null hypothesis used for significance checks.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr in the quadratic sensitivity combination.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary with:
  - raw: the output of `sensitivity_analysis(...)`
  - interpretation: machine-readable interpretation fields
  - summary: compact human-readable interpretation

######## `sensitivity_analysis`

```python
sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False
) -> Dict[str, Any]
```

Compute bias-aware components and cache them.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (R^2 form, R_Y^2; converted to odds form internally).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr in the quadratic combination of sensitivity components.
  If True and m_alpha/rr are available, the bias bound is computed via the
  per-unit quadratic form and RV/RVa are not reported.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results:
  - theta, se, alpha, z
  - sampling_ci
  - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
  - bias_aware_ci = faithful CI for the bounds
  - max_bias and components (sigma2, nu2)
  - params (r2_y, r2_d, rho, use_signed_rr)

######## `sensitivity_benchmark`

```python
sensitivity_benchmark(
    effect_estimation: Dict[str, Any],
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing r2_y, r2_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Legacy name for additional keyword arguments passed to `IRM.estimate(...)`
  on the short model. If `score` is omitted, ATE/ATTE is inferred from
  the supplied estimate/model, and defaults to ATE.
  If `diagnostic_data` is omitted, it defaults to `False` for faster benchmarking.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- r2_y, r2_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

####### `sensitivity_analysis`

```python
sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False
) -> Dict[str, Any]
```

Compute bias-aware components and cache them.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **r2_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (R^2 form, R_Y^2; converted to odds form internally).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr in the quadratic combination of sensitivity components.
  If True and m_alpha/rr are available, the bias bound is computed via the
  per-unit quadratic form and RV/RVa are not reported.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results:
  - theta, se, alpha, z
  - sampling_ci
  - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
  - bias_aware_ci = faithful CI for the bounds
  - max_bias and components (sigma2, nu2)
  - params (r2_y, r2_d, rho, use_signed_rr)

####### `sensitivity_benchmark`

```python
sensitivity_benchmark(
    effect_estimation: Dict[str, Any],
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing r2_y, r2_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Legacy name for additional keyword arguments passed to `IRM.estimate(...)`
  on the short model. If `score` is omitted, ATE/ATTE is inferred from
  the supplied estimate/model, and defaults to ATE.
  If `diagnostic_data` is omitted, it defaults to `False` for faster benchmarking.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- r2_y, r2_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

####### `unconfoundedness_validation`

Unconfoundedness diagnostics focused on covariate balance (SMD).

**Functions:**

- [**run_unconfoundedness_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation.run_unconfoundedness_diagnostics) – Run unconfoundedness diagnostics from `CausalData` and `CausalEstimate`.

######## `run_unconfoundedness_diagnostics`

```python
run_unconfoundedness_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
    return_summary: bool = True
) -> Dict[str, Any]
```

Run unconfoundedness diagnostics from `CausalData` and `CausalEstimate`.

### `causalis.shared`

**Modules:**

- [**confounders_balance**](#causalis.shared.confounders_balance) –
- [**outcome_outliers**](#causalis.shared.outcome_outliers) –
- [**outcome_plots**](#causalis.shared.outcome_plots) –
- [**outcome_stats**](#causalis.shared.outcome_stats) – Outcome shared grouped by treatment for CausalData.
- [**rct_design**](#causalis.shared.rct_design) – Design module for experimental rct_design utilities.
- [**srm**](#causalis.shared.srm) – Sample Ratio Mismatch (SRM) utilities for randomized experiments.
- [**sutva_validation**](#causalis.shared.sutva_validation) – SUTVA validation helper.

**Classes:**

- [**SRMResult**](#causalis.shared.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**check_srm**](#causalis.shared.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.
- [**outcome_plot_boxplot**](#causalis.shared.outcome_plot_boxplot) – Prettified boxplot of the outcome by treatment.
- [**outcome_plot_dist**](#causalis.shared.outcome_plot_dist) – Plot the distribution of the outcome for each treatment on a single, pretty plot.
- [**print_sutva_questions**](#causalis.shared.print_sutva_questions) – Print the SUTVA validation questions.

#### `QUESTIONS`

```python
QUESTIONS: Iterable[str] = (
    "1.) Are your clients independent (i). Outcome of ones do not depend on others?",
    "2.) Are all clients have full window to measure metrics?",
    "3.) Do you measure confounders before treatment and outcome after?",
    "4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?",
)

```

#### `SRMResult`

```python
SRMResult(
    chi2: float,
    p_value: float,
    expected: Dict[Hashable, float],
    observed: Dict[Hashable, int],
    alpha: float,
    is_srm: bool,
    warning: str | None = None,
) -> None
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.shared.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**p_value**](#causalis.shared.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test, rounded to 5 decimals.
- [**expected**](#causalis.shared.SRMResult.expected) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.shared.SRMResult.observed) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.shared.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.shared.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (chi-square p-value < alpha), False otherwise.
- [**warning**](#causalis.shared.SRMResult.warning) (<code>[str](#str) or None</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

##### `alpha`

```python
alpha: float
```

##### `chi2`

```python
chi2: float
```

##### `expected`

```python
expected: Dict[Hashable, float]
```

##### `is_srm`

```python
is_srm: bool
```

##### `observed`

```python
observed: Dict[Hashable, int]
```

##### `p_value`

```python
p_value: float
```

##### `warning`

```python
warning: str | None = None
```

#### `check_srm`

```python
check_srm(
    assignments: Union[
        Iterable[Hashable], pd.Series, CausalData, Mapping[Hashable, Number]
    ],
    target_allocation: Dict[Hashable, Number],
    alpha: float = 0.001,
    min_expected: float = 5.0,
    strict_variants: bool = True,
) -> SRMResult
```

Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData) or [Mapping](#collections.abc.Mapping)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Observed variant assignments. If iterable or Series, elements are labels per
  unit (user_id, session_id, etc.). If CausalData is provided, the treatment
  column is used. If a mapping is provided, it is treated as
  `{variant: observed_count}` with non-negative integer counts.
- **target_allocation** (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping `{variant: p}` describing intended allocation as probabilities.
- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Target allocation probabilities must sum to 1 within numerical tolerance.
- `is_srm` is computed using the unrounded p-value; the returned
  `p_value` is rounded to 5 decimals.
- Missing assignments are dropped and reported via `warning`.
- Requires SciPy for p-value computation.

</details>

**Examples:**

```pycon
>>> assignments = ["control"] * 50 + ["treatment"] * 50
>>> check_srm(assignments, {"control": 0.5, "treatment": 0.5}, alpha=1e-3)
SRMResult(status=no SRM, p_value=1.00000, chi2=0.0000)
```

```pycon
>>> counts = {"control": 70, "treatment": 30}
>>> check_srm(counts, {"control": 0.5, "treatment": 0.5})
SRMResult(status=SRM DETECTED, p_value=0.00006, chi2=16.0000)
```

#### `confounders_balance`

**Functions:**

- [**confounders_balance**](#causalis.shared.confounders_balance.confounders_balance) – Compute balance diagnostics for confounders between two treatment groups.

##### `confounders_balance`

```python
confounders_balance(
    data: CausalData | MultiCausalData,
    treatment_d_0: Optional[str] = None,
    treatment_d_1: Optional[str] = None,
) -> pd.DataFrame
```

Compute balance diagnostics for confounders between two treatment groups.

Produces a DataFrame containing expanded confounder columns (after one-hot
encoding categorical variables if present) with:

- confounders: name of the confounder
- mean_d_0: mean value for control group (t=0)
- mean_d_1: mean value for treated group (t=1)
- abs_diff: abs(mean_d_1 - mean_d_0)
- smd: standardized mean difference (Cohen's d using pooled std)
- ks_pvalue: p-value for the KS test (rounded to 5 decimal places, non-scientific)

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – The causal dataset containing the dataframe and confounders.
- **treatment_d_0** (<code>[str](#str)</code>) – For MultiCausalData, name of the first treatment column to compare.
  Mapped to output column `mean_d_0`.
- **treatment_d_1** (<code>[str](#str)</code>) – For MultiCausalData, name of the second treatment column to compare.
  Mapped to output column `mean_d_1`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Balance table sorted by |smd| (descending).

#### `outcome_outliers`

**Functions:**

- [**outcome_outliers**](#causalis.shared.outcome_outliers.outcome_outliers) – Detect outcome outliers per treatment group using IQR or z-score rules.

##### `outcome_outliers`

```python
outcome_outliers(
    data: CausalData | MultiCausalData,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    *,
    method: Literal["iqr", "zscore"] = "iqr",
    iqr_k: float = 1.5,
    z_thresh: float = 3.0,
    tail: Literal["both", "lower", "upper"] = "both",
    return_rows: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
```

Detect outcome outliers per treatment group using IQR or z-score rules.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – Causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. For MultiCausalData, if not provided, one-hot
  treatment columns are converted to assigned treatment labels.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to `data.outcome`.
- **method** (<code>('iqr', 'zscore')</code>) – Outlier detection rule.
- **iqr_k** (<code>[float](#float)</code>) – Multiplier for the IQR rule.
- **z_thresh** (<code>[float](#float)</code>) – Z-score threshold for the z-score rule.
- **tail** (<code>('both', 'lower', 'upper')</code>) – Which tail(s) to flag as outliers.
- **return_rows** (<code>[bool](#bool)</code>) – If True, also return the rows flagged as outliers (subset of `data.df`).

**Returns:**

- **summary** (<code>[DataFrame](#pandas.DataFrame)</code>) – Per-treatment summary with counts, rates, bounds, and flags.
- **outliers** (<code>[DataFrame](#pandas.DataFrame)</code>) – Only returned when `return_rows=True`. Subset of `data.df` containing
  flagged outlier rows.

<details class="note" open markdown="1">
<summary>Notes</summary>

Bounds are computed within each treatment group.

</details>

#### `outcome_plot_boxplot`

```python
outcome_plot_boxplot(
    data: Union[CausalData, MultiCausalData],
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    showfliers: bool = True,
    patch_artist: bool = True,
    palette: Optional[Union[list, dict]] = None,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure
```

Prettified boxplot of the outcome by treatment.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI figure, scalable fonts
- Soft modern color styling (default Matplotlib palette)
- Optional outliers, gentle transparency
- Optional save to PNG/SVG/PDF

</details>

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. For MultiCausalData, if not provided, one-hot
  treatment columns are converted to assigned treatment labels.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches (width, height).
- **dpi** (<code>[int](#int)</code>) – Dots per inch for the figure.
- **font_scale** (<code>[float](#float)</code>) – Scaling factor for all font sizes in the plot.
- **showfliers** (<code>[bool](#bool)</code>) – Whether to show outliers (fliers).
- **patch_artist** (<code>[bool](#bool)</code>) – Whether to fill boxes with color.
- **palette** (<code>[list](#list) or [dict](#dict)</code>) – Color palette for treatments (list in treatment order or dict {treatment: color}).
- **save** (<code>[str](#str)</code>) – Path to save the figure (e.g., "boxplot.png").
- **save_dpi** (<code>[int](#int)</code>) – DPI for the saved figure. Defaults to 300 for raster formats.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save the figure with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure object.

#### `outcome_plot_dist`

```python
outcome_plot_dist(
    data: Union[CausalData, MultiCausalData],
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    bins: Union[str, int] = "fd",
    density: bool = True,
    alpha: float = 0.45,
    sharex: bool = True,
    kde: bool = True,
    clip: Optional[Tuple[float, float]] = (0.01, 0.99),
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    palette: Optional[Union[list, dict]] = None,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure
```

Plot the distribution of the outcome for each treatment on a single, pretty plot.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI canvas + scalable fonts
- Default Matplotlib colors; KDE & mean lines match their histogram colors
- Numeric outcomes: shared x-range (optional), optional KDE, quantile clipping
- Categorical outcomes: normalized grouped bars by treatment
- Binary outcomes: proportion bars with percent labels (no KDE)
- Optional hi-res export (PNG/SVG/PDF)

</details>

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. For MultiCausalData, if not provided, one-hot
  treatment columns are converted to assigned treatment labels.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Number of bins for histograms (e.g., "fd", "auto", or an integer).
- **density** (<code>[bool](#bool)</code>) – Whether to normalize histograms to form a density.
- **alpha** (<code>[float](#float)</code>) – Transparency for overlaid histograms and bars.
- **sharex** (<code>[bool](#bool)</code>) – If True, use the same x-limits across treatments for numeric outcomes.
- **kde** (<code>[bool](#bool)</code>) – Whether to overlay a smooth density (KDE) for numeric outcomes.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to trim tails for nicer view of numeric outcomes.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches (width, height).
- **dpi** (<code>[int](#int)</code>) – Dots per inch for the figure.
- **font_scale** (<code>[float](#float)</code>) – Scaling factor for all font sizes in the plot.
- **palette** (<code>[list](#list) or [dict](#dict)</code>) – Color palette for treatments (list in treatment order or dict {treatment: color}).
- **save** (<code>[str](#str)</code>) – Path to save the figure (e.g., "outcome.png").
- **save_dpi** (<code>[int](#int)</code>) – DPI for the saved figure. Defaults to 300 for raster formats.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save the figure with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure object.

#### `outcome_plots`

**Functions:**

- [**outcome_plot_boxplot**](#causalis.shared.outcome_plots.outcome_plot_boxplot) – Prettified boxplot of the outcome by treatment.
- [**outcome_plot_dist**](#causalis.shared.outcome_plots.outcome_plot_dist) – Plot the distribution of the outcome for each treatment on a single, pretty plot.
- [**outcome_plots**](#causalis.shared.outcome_plots.outcome_plots) – Plot the distribution of the outcome for every treatment on one plot,

##### `outcome_plot_boxplot`

```python
outcome_plot_boxplot(
    data: Union[CausalData, MultiCausalData],
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    showfliers: bool = True,
    patch_artist: bool = True,
    palette: Optional[Union[list, dict]] = None,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure
```

Prettified boxplot of the outcome by treatment.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI figure, scalable fonts
- Soft modern color styling (default Matplotlib palette)
- Optional outliers, gentle transparency
- Optional save to PNG/SVG/PDF

</details>

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. For MultiCausalData, if not provided, one-hot
  treatment columns are converted to assigned treatment labels.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches (width, height).
- **dpi** (<code>[int](#int)</code>) – Dots per inch for the figure.
- **font_scale** (<code>[float](#float)</code>) – Scaling factor for all font sizes in the plot.
- **showfliers** (<code>[bool](#bool)</code>) – Whether to show outliers (fliers).
- **patch_artist** (<code>[bool](#bool)</code>) – Whether to fill boxes with color.
- **palette** (<code>[list](#list) or [dict](#dict)</code>) – Color palette for treatments (list in treatment order or dict {treatment: color}).
- **save** (<code>[str](#str)</code>) – Path to save the figure (e.g., "boxplot.png").
- **save_dpi** (<code>[int](#int)</code>) – DPI for the saved figure. Defaults to 300 for raster formats.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save the figure with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure object.

##### `outcome_plot_dist`

```python
outcome_plot_dist(
    data: Union[CausalData, MultiCausalData],
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    bins: Union[str, int] = "fd",
    density: bool = True,
    alpha: float = 0.45,
    sharex: bool = True,
    kde: bool = True,
    clip: Optional[Tuple[float, float]] = (0.01, 0.99),
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    palette: Optional[Union[list, dict]] = None,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure
```

Plot the distribution of the outcome for each treatment on a single, pretty plot.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI canvas + scalable fonts
- Default Matplotlib colors; KDE & mean lines match their histogram colors
- Numeric outcomes: shared x-range (optional), optional KDE, quantile clipping
- Categorical outcomes: normalized grouped bars by treatment
- Binary outcomes: proportion bars with percent labels (no KDE)
- Optional hi-res export (PNG/SVG/PDF)

</details>

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. For MultiCausalData, if not provided, one-hot
  treatment columns are converted to assigned treatment labels.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Number of bins for histograms (e.g., "fd", "auto", or an integer).
- **density** (<code>[bool](#bool)</code>) – Whether to normalize histograms to form a density.
- **alpha** (<code>[float](#float)</code>) – Transparency for overlaid histograms and bars.
- **sharex** (<code>[bool](#bool)</code>) – If True, use the same x-limits across treatments for numeric outcomes.
- **kde** (<code>[bool](#bool)</code>) – Whether to overlay a smooth density (KDE) for numeric outcomes.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to trim tails for nicer view of numeric outcomes.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches (width, height).
- **dpi** (<code>[int](#int)</code>) – Dots per inch for the figure.
- **font_scale** (<code>[float](#float)</code>) – Scaling factor for all font sizes in the plot.
- **palette** (<code>[list](#list) or [dict](#dict)</code>) – Color palette for treatments (list in treatment order or dict {treatment: color}).
- **save** (<code>[str](#str)</code>) – Path to save the figure (e.g., "outcome.png").
- **save_dpi** (<code>[int](#int)</code>) – DPI for the saved figure. Defaults to 300 for raster formats.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save the figure with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure object.

##### `outcome_plots`

```python
outcome_plots(
    data: Union[CausalData, MultiCausalData],
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    bins: int = 30,
    density: bool = True,
    alpha: float = 0.5,
    figsize: Tuple[float, float] = (7, 4),
    sharex: bool = True,
    palette: Optional[Union[list, dict]] = None,
) -> Tuple[plt.Figure, plt.Figure]
```

Plot the distribution of the outcome for every treatment on one plot,
and also produce a boxplot by treatment to visualize outliers.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. Defaults to the one in `data_contracts`.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **bins** (<code>[int](#int)</code>) – Number of bins for histograms when the outcome is numeric.
- **density** (<code>[bool](#bool)</code>) – Whether to normalize histograms to form a density.
- **alpha** (<code>[float](#float)</code>) – Transparency for overlaid histograms.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size for the plots (width, height).
- **sharex** (<code>[bool](#bool)</code>) – If True and the outcome is numeric, use the same x-limits across treatments.
- **palette** (<code>[list](#list) or [dict](#dict)</code>) – Color palette for treatments (list in treatment order or dict {treatment: color}).

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[Figure](#matplotlib.figure.Figure), [Figure](#matplotlib.figure.Figure)\]</code> – (fig_distribution, fig_boxplot)

#### `outcome_stats`

Outcome shared grouped by treatment for CausalData.

**Functions:**

- [**outcome_stats**](#causalis.shared.outcome_stats.outcome_stats) – Comprehensive outcome shared grouped by treatment.

##### `outcome_stats`

```python
outcome_stats(data: CausalData | MultiCausalData) -> pd.DataFrame
```

Comprehensive outcome shared grouped by treatment.

Returns a DataFrame with detailed outcome shared for each treatment group,
including count, mean, std, min, various percentiles, and max.
This function provides comprehensive outcome analysis and returns
data_contracts in a clean DataFrame format suitable for reporting.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData) or [MultiCausalData](#causalis.dgp.multicausaldata.MultiCausalData)</code>) – The causal dataset containing treatment and outcome variables.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with treatment groups as index and the following columns:
- count: number of observations in each group
- mean: average outcome value
- std: standard deviation of outcome
- min: minimum outcome value
- p10: 10th percentile
- p25: 25th percentile (Q1)
- median: 50th percentile (median)
- p75: 75th percentile (Q3)
- p90: 90th percentile
- max: maximum outcome value

**Examples:**

```pycon
>>> stats = outcome_stats(causal_data)
>>> print(stats)
   treatment  count      mean       std       min       p10       p25    median       p75       p90       max
0          0   3000  5.123456  2.345678  0.123456  2.345678  3.456789  5.123456  6.789012  7.890123  9.876543
1          1   2000  6.789012  2.456789  0.234567  3.456789  4.567890  6.789012  8.901234  9.012345  10.987654
```

#### `print_sutva_questions`

```python
print_sutva_questions() -> None
```

Print the SUTVA validation questions.

Just prints questions, nothing more.

#### `rct_design`

Design module for experimental rct_design utilities.

**Modules:**

- [**mde**](#causalis.shared.rct_design.mde) – Utility functions for calculating Minimum Detectable Effect (MDE) for experimental rct_design.
- [**split**](#causalis.shared.rct_design.split) – Split (assignment) utilities for randomized controlled experiments.

**Classes:**

- [**SRMResult**](#causalis.shared.rct_design.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**assign_variants_df**](#causalis.shared.rct_design.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.
- [**calculate_mde**](#causalis.shared.rct_design.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.
- [**check_srm**](#causalis.shared.rct_design.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

##### `SRMResult`

```python
SRMResult(
    chi2: float,
    p_value: float,
    expected: Dict[Hashable, float],
    observed: Dict[Hashable, int],
    alpha: float,
    is_srm: bool,
    warning: str | None = None,
) -> None
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.shared.rct_design.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**p_value**](#causalis.shared.rct_design.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test, rounded to 5 decimals.
- [**expected**](#causalis.shared.rct_design.SRMResult.expected) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.shared.rct_design.SRMResult.observed) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.shared.rct_design.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.shared.rct_design.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (chi-square p-value < alpha), False otherwise.
- [**warning**](#causalis.shared.rct_design.SRMResult.warning) (<code>[str](#str) or None</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `alpha`

```python
alpha: float
```

###### `chi2`

```python
chi2: float
```

###### `expected`

```python
expected: Dict[Hashable, float]
```

###### `is_srm`

```python
is_srm: bool
```

###### `observed`

```python
observed: Dict[Hashable, int]
```

###### `p_value`

```python
p_value: float
```

###### `warning`

```python
warning: str | None = None
```

##### `assign_variants_df`

```python
assign_variants_df(
    df: pd.DataFrame,
    id_col: str,
    experiment_id: str,
    variants: Dict[str, float],
    *,
    salt: str = "global_ab_salt",
    layer_id: str = "default",
    variant_col: str = "variant"
) -> pd.DataFrame
```

Deterministically assign variants for each row in df based on id_col.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – Input DataFrame with an identifier column.
- **id_col** (<code>[str](#str)</code>) – Column name in df containing entity identifiers (user_id, session_id, etc.).
- **experiment_id** (<code>[str](#str)</code>) – Unique identifier for the experiment (versioned for reruns).
- **variants** (<code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code>) – Mapping from variant name to weight (coverage). Weights must be non-negative
  and their sum must be in (0, 1\]. If the sum is < 1, the remaining mass
  corresponds to "not in experiment" and the assignment will be None.
- **salt** (<code>[str](#str)</code>) – Secret string to de-correlate from other hash uses and make assignments
  non-gameable.
- **layer_id** (<code>[str](#str)</code>) – Identifier for mutual exclusivity layer or surface. In this case work like
  another random
- **variant_col** (<code>[str](#str)</code>) – Name of output column to store assigned variant labels.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of df with an extra column `variant_col`.
  Entities outside experiment coverage will have None in the variant column.

##### `calculate_mde`

```python
calculate_mde(
    sample_size: Union[int, Tuple[int, int]],
    baseline_rate: Optional[float] = None,
    variance: Optional[Union[float, Tuple[float, float]]] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    data_type: str = "conversion",
    ratio: float = 0.5,
) -> Dict[str, Any]
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data_contracts) or baseline mean (for continuous data_contracts).
  Required for conversion data_contracts.
- **variance** (<code>float or tuple of float</code>) – Variance of the data_contracts. For conversion data_contracts, this is calculated from the baseline rate if not provided.
  For continuous data_contracts, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data_contracts. Either 'conversion' for binary/conversion data_contracts or 'continuous' for continuous data_contracts.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data_contracts with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data_contracts with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

##### `check_srm`

```python
check_srm(
    assignments: Union[
        Iterable[Hashable], pd.Series, CausalData, Mapping[Hashable, Number]
    ],
    target_allocation: Dict[Hashable, Number],
    alpha: float = 0.001,
    min_expected: float = 5.0,
    strict_variants: bool = True,
) -> SRMResult
```

Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData) or [Mapping](#collections.abc.Mapping)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Observed variant assignments. If iterable or Series, elements are labels per
  unit (user_id, session_id, etc.). If CausalData is provided, the treatment
  column is used. If a mapping is provided, it is treated as
  `{variant: observed_count}` with non-negative integer counts.
- **target_allocation** (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping `{variant: p}` describing intended allocation as probabilities.
- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Target allocation probabilities must sum to 1 within numerical tolerance.
- `is_srm` is computed using the unrounded p-value; the returned
  `p_value` is rounded to 5 decimals.
- Missing assignments are dropped and reported via `warning`.
- Requires SciPy for p-value computation.

</details>

**Examples:**

```pycon
>>> assignments = ["control"] * 50 + ["treatment"] * 50
>>> check_srm(assignments, {"control": 0.5, "treatment": 0.5}, alpha=1e-3)
SRMResult(status=no SRM, p_value=1.00000, chi2=0.0000)
```

```pycon
>>> counts = {"control": 70, "treatment": 30}
>>> check_srm(counts, {"control": 0.5, "treatment": 0.5})
SRMResult(status=SRM DETECTED, p_value=0.00006, chi2=16.0000)
```

##### `mde`

Utility functions for calculating Minimum Detectable Effect (MDE) for experimental rct_design.

**Functions:**

- [**calculate_mde**](#causalis.shared.rct_design.mde.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

###### `calculate_mde`

```python
calculate_mde(
    sample_size: Union[int, Tuple[int, int]],
    baseline_rate: Optional[float] = None,
    variance: Optional[Union[float, Tuple[float, float]]] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    data_type: str = "conversion",
    ratio: float = 0.5,
) -> Dict[str, Any]
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data_contracts) or baseline mean (for continuous data_contracts).
  Required for conversion data_contracts.
- **variance** (<code>float or tuple of float</code>) – Variance of the data_contracts. For conversion data_contracts, this is calculated from the baseline rate if not provided.
  For continuous data_contracts, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data_contracts. Either 'conversion' for binary/conversion data_contracts or 'continuous' for continuous data_contracts.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data_contracts with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data_contracts with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

##### `split`

Split (assignment) utilities for randomized controlled experiments.

This module provides deterministic assignment of variants to entities based
on hashing a composite key (salt | layer_id | experiment_id | entity_id)
into the unit interval and mapping it to cumulative variant weights.

The implementation mirrors the reference notebook in docs/cases/rct_design.ipynb.

**Functions:**

- [**assign_variants_df**](#causalis.shared.rct_design.split.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.

###### `assign_variants_df`

```python
assign_variants_df(
    df: pd.DataFrame,
    id_col: str,
    experiment_id: str,
    variants: Dict[str, float],
    *,
    salt: str = "global_ab_salt",
    layer_id: str = "default",
    variant_col: str = "variant"
) -> pd.DataFrame
```

Deterministically assign variants for each row in df based on id_col.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – Input DataFrame with an identifier column.
- **id_col** (<code>[str](#str)</code>) – Column name in df containing entity identifiers (user_id, session_id, etc.).
- **experiment_id** (<code>[str](#str)</code>) – Unique identifier for the experiment (versioned for reruns).
- **variants** (<code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code>) – Mapping from variant name to weight (coverage). Weights must be non-negative
  and their sum must be in (0, 1\]. If the sum is < 1, the remaining mass
  corresponds to "not in experiment" and the assignment will be None.
- **salt** (<code>[str](#str)</code>) – Secret string to de-correlate from other hash uses and make assignments
  non-gameable.
- **layer_id** (<code>[str](#str)</code>) – Identifier for mutual exclusivity layer or surface. In this case work like
  another random
- **variant_col** (<code>[str](#str)</code>) – Name of output column to store assigned variant labels.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of df with an extra column `variant_col`.
  Entities outside experiment coverage will have None in the variant column.

#### `srm`

Sample Ratio Mismatch (SRM) utilities for randomized experiments.

This module provides a chi-square goodness-of-fit SRM check for randomized
experiments. It accepts observed assignments as labels or aggregated counts
and returns a compact result object with diagnostics.

**Classes:**

- [**SRMResult**](#causalis.shared.srm.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**check_srm**](#causalis.shared.srm.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

##### `SRMResult`

```python
SRMResult(
    chi2: float,
    p_value: float,
    expected: Dict[Hashable, float],
    observed: Dict[Hashable, int],
    alpha: float,
    is_srm: bool,
    warning: str | None = None,
) -> None
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.shared.srm.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**p_value**](#causalis.shared.srm.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test, rounded to 5 decimals.
- [**expected**](#causalis.shared.srm.SRMResult.expected) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.shared.srm.SRMResult.observed) (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.shared.srm.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.shared.srm.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (chi-square p-value < alpha), False otherwise.
- [**warning**](#causalis.shared.srm.SRMResult.warning) (<code>[str](#str) or None</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `alpha`

```python
alpha: float
```

###### `chi2`

```python
chi2: float
```

###### `expected`

```python
expected: Dict[Hashable, float]
```

###### `is_srm`

```python
is_srm: bool
```

###### `observed`

```python
observed: Dict[Hashable, int]
```

###### `p_value`

```python
p_value: float
```

###### `warning`

```python
warning: str | None = None
```

##### `check_srm`

```python
check_srm(
    assignments: Union[
        Iterable[Hashable], pd.Series, CausalData, Mapping[Hashable, Number]
    ],
    target_allocation: Dict[Hashable, Number],
    alpha: float = 0.001,
    min_expected: float = 5.0,
    strict_variants: bool = True,
) -> SRMResult
```

Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData) or [Mapping](#collections.abc.Mapping)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Observed variant assignments. If iterable or Series, elements are labels per
  unit (user_id, session_id, etc.). If CausalData is provided, the treatment
  column is used. If a mapping is provided, it is treated as
  `{variant: observed_count}` with non-negative integer counts.
- **target_allocation** (<code>[dict](#dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping `{variant: p}` describing intended allocation as probabilities.
- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Target allocation probabilities must sum to 1 within numerical tolerance.
- `is_srm` is computed using the unrounded p-value; the returned
  `p_value` is rounded to 5 decimals.
- Missing assignments are dropped and reported via `warning`.
- Requires SciPy for p-value computation.

</details>

**Examples:**

```pycon
>>> assignments = ["control"] * 50 + ["treatment"] * 50
>>> check_srm(assignments, {"control": 0.5, "treatment": 0.5}, alpha=1e-3)
SRMResult(status=no SRM, p_value=1.00000, chi2=0.0000)
```

```pycon
>>> counts = {"control": 70, "treatment": 30}
>>> check_srm(counts, {"control": 0.5, "treatment": 0.5})
SRMResult(status=SRM DETECTED, p_value=0.00006, chi2=16.0000)
```

#### `sutva_validation`

SUTVA validation helper.

This module provides a simple function to print four SUTVA-related
questions for the user to consider. It has no side effects on import.

**Functions:**

- [**print_sutva_questions**](#causalis.shared.sutva_validation.print_sutva_questions) – Print the SUTVA validation questions.

##### `QUESTIONS`

```python
QUESTIONS: Iterable[str] = (
    "1.) Are your clients independent (i). Outcome of ones do not depend on others?",
    "2.) Are all clients have full window to measure metrics?",
    "3.) Do you measure confounders before treatment and outcome after?",
    "4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?",
)

```

##### `print_sutva_questions`

```python
print_sutva_questions() -> None
```

Print the SUTVA validation questions.

Just prints questions, nothing more.
