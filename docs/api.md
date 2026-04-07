# API Reference

Core modules in `src/`. All classes and functions are importable directly from the module path.

---

## `train_model` — Model Classes

### `OutcomeModel`

Base class inherited by all model types. Handles data splitting, feature extraction, Lasso feature selection, and demographic tracking.

```python
from src.train_model import OutcomeModel

model = OutcomeModel(
    data=df,
    id_column="id",
    target_column=["outcome"],
    seed=42
)
```

**Key attributes after init:**

| Attribute | Description |
|:---|:---|
| `model.X_train` | Feature matrix for training (75%) |
| `model.X_test` | Feature matrix for internal testing (25%) |
| `model.y_train` | Target values for training |
| `model.y_test` | Target values for internal testing |
| `model.selected_features` | Features retained after Lasso (populated after `selectFeatures()`) |
| `model.best_threshold` | Classification threshold (populated after `train()`) |

**Methods:**

| Method | Description |
|:---|:---|
| `selectFeatures()` | Run L1 Lasso regularization to select features |
| `train()` | Fit the model on training data |
| `evaluate(heldout_df)` | Evaluate on both internal test split and held-out set |

---

### `LogisticModel`

Binary classification using L1-regularized Logistic Regression (scikit-learn).

```python
from src.train_model import LogisticModel

model = LogisticModel(
    data=df,
    id_column="id",
    target_column=["outcome"],
    Cs=[1.0],   # regularization strengths
    seed=42
)
model.selectFeatures()
model.train()
heldout_preds, heldout_evals, subset_preds, subset_evals = model.evaluate(heldout_df)
```

**Evaluation output keys:** `roc`, `confusion_matrix`, `precision`, `recall`, `demographics`, `training_demographics`

---

### `NegativeBinomialModel`

Count regression using Negative Binomial GLM (statsmodels).

```python
from src.train_model import NegativeBinomialModel

model = NegativeBinomialModel(data=df, id_column="id", target_column=["count_outcome"])
model.selectFeatures()
model.train()
```

**Evaluation output keys:** `mse`, `rmse`, `mae`, `pearson_r`, `mcfadden_r2`, `demographics`, `training_demographics`

---

### `CoxProportionalHazard`

Survival analysis using Cox PH Fitter (lifelines).

```python
from src.train_model import CoxProportionalHazard

model = CoxProportionalHazard(
    data=df,
    id_column="id",
    target_column=["duration", "event"]
)
model.selectFeatures()
model.train()
```

**Evaluation output keys:** `concordance_index`, `demographics`, `training_demographics`

---

## `model_training` — Training Orchestration

### `train_and_evaluate_models`

Runs the full train → evaluate loop across a list of PSM subsets.

```python
from src.model_training import train_and_evaluate_models
from src.constants import EndpointType

outcome = {
    "endpointType": EndpointType.LOGICAL,
    "columnsToUse": ["ctn0094_relapse_event"]
}

results = train_and_evaluate_models(
    subsets=[df1, df2, df3],
    id_column="id",
    outcome=outcome,
    heldout_data=heldout_df
)
```

Returns a `pd.DataFrame` with a MultiIndex column structure:

| Level 0 | Level 1 | Contents |
|:---|:---|:---|
| `heldout` | `predictions` | Iterator of (id, prediction) pairs |
| `heldout` | `evaluations` | Dict of metrics on held-out set |
| `subset` | `predictions` | Iterator of (id, prediction) pairs |
| `subset` | `evaluations` | Dict of metrics on internal test split |

---

## `validate` — Dataset Validation

### `validate_dataset`

Validates an input DataFrame before preprocessing.

```python
from src.validate import validate_dataset
from src.constants import EndpointType

validate_dataset(df, outcome_col="ctn0094_relapse_event", endpoint_type=EndpointType.LOGICAL)
```

Raises `ValueError` if:

- The outcome column is missing
- Values don't match the expected type (e.g. non-binary values for logical endpoints)
- Duration column is missing or non-numeric for survival endpoints

---

## `constants` — Endpoint Types

```python
from src.constants import EndpointType

EndpointType.LOGICAL    # Binary classification
EndpointType.INTEGER    # Count regression
EndpointType.SURVIVAL   # Time-to-event
```
