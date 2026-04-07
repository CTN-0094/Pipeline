# Validation

**File:** `src/validate.py`

Validates the input dataset before any preprocessing or model training begins. Called early in the pipeline to catch schema issues before expensive operations run.

---

## `validate_dataset_for_model`

```python
validate_dataset_for_model(df, model_type, outcome_col, time_col=None)
```

Validates a DataFrame against the requirements of the specified endpoint type.

**Parameters**

| Name | Type | Description |
|:---|:---|:---|
| `df` | `pd.DataFrame` | Input dataset to validate |
| `model_type` | `EndpointType` or `str` | Endpoint type — can pass `"logical"`, `"integer"`, or `"survival"` as a string |
| `outcome_col` | `str` | Name of the target/outcome column |
| `time_col` | `str`, optional | Name of the duration column (required for survival endpoints only) |

**Raises**

| Error | Condition |
|:---|:---|
| `ValueError` | Outcome column not found in DataFrame |
| `ValueError` | Logical outcome contains values other than 0 and 1 |
| `ValueError` | Integer outcome column is not integer dtype |
| `ValueError` | Survival endpoint missing `time_col` argument |
| `ValueError` | Time column not found in DataFrame or is non-numeric |
| `ValueError` | Survival event indicator contains values other than 0 and 1 |
| `ValueError` | Unrecognised `model_type` string |

### Examples

=== "Logical"

    ```python
    from src.validate import validate_dataset_for_model
    from src.constants import EndpointType

    validate_dataset_for_model(
        df=df,
        model_type=EndpointType.LOGICAL,
        outcome_col="ctn0094_relapse_event"
    )
    ```

=== "Integer"

    ```python
    validate_dataset_for_model(
        df=df,
        model_type="integer",      # string form also accepted
        outcome_col="Ab_schottenfeldB_2008"
    )
    ```

=== "Survival"

    ```python
    validate_dataset_for_model(
        df=df,
        model_type=EndpointType.SURVIVAL,
        outcome_col="Ab_mokri_2016",
        time_col="Ab_mokri_2016_TTE"
    )
    ```
