# Model Training

**File:** `src/model_training.py`

Orchestrates the full train → evaluate loop across all PSM subsets for a given outcome and seed. Called once per outcome per seed in the main pipeline entry point.

---

## `train_and_evaluate_models`

```python
train_and_evaluate_models(merged_subsets, id_column, selected_outcome, processed_data_heldout)
```

Iterates over each PSM subset, instantiates the appropriate model class based on endpoint type, runs feature selection and training, then evaluates on both the internal test split and the held-out set.

**Parameters**

| Name | Type | Description |
|:---|:---|:---|
| `merged_subsets` | `list[pd.DataFrame]` | List of PSM cohorts from `create_subsets()` |
| `id_column` | `str` | Name of the unique patient ID column |
| `selected_outcome` | `dict` | Outcome config with keys `endpointType` and `columnsToUse` |
| `processed_data_heldout` | `pd.DataFrame` | Fixed held-out evaluation set |

**Returns:** `pd.DataFrame` with a MultiIndex column structure:

```
               heldout                    subset
         predictions  evaluations   predictions  evaluations
row 0    ...          {...}         ...          {...}
row 1    ...          {...}         ...          {...}
...
```

Each row corresponds to one PSM subset. `predictions` is an iterator of `(id, prediction)` pairs. `evaluations` is a dict of metric values.

### Model selection

The model class is chosen automatically from `selected_outcome["endpointType"]`:

| `endpointType` | Model class |
|:---|:---|
| `EndpointType.LOGICAL` | `LogisticModel` |
| `EndpointType.INTEGER` | `NegativeBinomialModel` |
| `EndpointType.SURVIVAL` | `CoxProportionalHazard` |

### Example

```python
from src.model_training import train_and_evaluate_models
from src.constants import EndpointType

outcome = {
    "endpointType": EndpointType.LOGICAL,
    "columnsToUse": ["ctn0094_relapse_event"]
}

results = train_and_evaluate_models(
    merged_subsets=subsets,       # list of DataFrames from create_subsets()
    id_column="who",
    selected_outcome=outcome,
    processed_data_heldout=heldout_df
)

# Access heldout evaluations for subset 0
print(results.loc[0, ("heldout", "evaluations")])
# → {"roc": 0.74, "precision": 0.68, ...}
```
