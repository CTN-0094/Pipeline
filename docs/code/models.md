# Models

**File:** `src/train_model.py`

All model classes inherit from `OutcomeModel`. Each subclass implements `selectFeatures()`, `train()`, and `_evaluateOnValidation()` for its specific endpoint type.

---

## `OutcomeModel` (base class)

Handles data splitting, feature extraction, Lasso feature selection, and demographic tracking. Not used directly — instantiate one of the subclasses below.

### `__init__`

```python
OutcomeModel(data, id_column, target_column, seed=None)
```

| Parameter | Type | Description |
|:---|:---|:---|
| `data` | `pd.DataFrame` | Full input dataset |
| `id_column` | `str` | Unique patient ID column — excluded from features |
| `target_column` | `list[str]` | Target column(s) to predict |
| `seed` | `int`, optional | Random seed for reproducibility |

**Raises**

| Error | Condition |
|:---|:---|
| `ValueError` | `data` is `None` or empty |
| `TypeError` | `target_column` is not a list |
| `ValueError` | `id_column` not found in `data` |
| `ValueError` | Any column in `target_column` not found in `data` |
| `ValueError` | Duplicate IDs found in `id_column` |

**Key attributes after init**

| Attribute | Description |
|:---|:---|
| `X_train`, `X_test` | 75/25 feature split |
| `y_train`, `y_test` | Corresponding target values |
| `who_test` | Patient IDs for the test set |
| `selected_features` | All feature columns (updated after `selectFeatures()`) |
| `best_threshold` | `0.5` default (updated after `train()` for logistic) |

### `lasso_feature_selection`

```python
lasso_feature_selection(model_type='classification', alpha=0.01)
```

Fits an L1-regularized model and retains only features with non-zero coefficients. Raises `ValueError` if no features are selected (over-regularization).

| `model_type` | Underlying model |
|:---|:---|
| `"classification"` | `LogisticRegression(penalty='l1', solver='saga')` |
| `"regression"` | `Lasso(alpha=alpha)` |

---

## `LogisticModel`

Binary classification using scikit-learn's `LogisticRegression`. Use for `EndpointType.LOGICAL` outcomes.

```python
from src.train_model import LogisticModel

model = LogisticModel(data=df, id_column="who", target_column=["ctn0094_relapse_event"], seed=42)
model.selectFeatures()
model.train()
heldout_preds, heldout_evals, subset_preds, subset_evals = model.evaluate(heldout_df)
```

### `train()`

Fits L1 Lasso for feature selection, then trains a standard `LogisticRegression` on the selected features. Sets `best_threshold` to the positive class proportion in `y_test`.

### `evaluate(processed_data_heldout)`

Returns a 4-tuple: `(heldout_predictions, heldout_evaluations, subset_predictions, subset_evaluations)`.

**Evaluation dict keys**

| Key | Type | Description |
|:---|:---|:---|
| `roc` | `float` | ROC-AUC score |
| `confusion_matrix` | `np.ndarray` | 2×2 confusion matrix |
| `precision` | `float` | Precision score |
| `recall` | `float` | Recall score |
| `demographics` | `str` | Demographic breakdown of the evaluation set |
| `training_demographics` | `str` | Demographic breakdown of the training set |

**Raises** `NotFittedError` if called before `train()`.

---

## `NegativeBinomialModel`

Count regression using statsmodels' Negative Binomial GLM. Use for `EndpointType.INTEGER` outcomes.

```python
from src.train_model import NegativeBinomialModel

model = NegativeBinomialModel(data=df, id_column="who", target_column=["Ab_schottenfeldB_2008"])
model.selectFeatures()
model.train()
```

Feature selection uses Lasso regression (`model_type="regression"`, `alpha=30`).

**Evaluation dict keys**

| Key | Description |
|:---|:---|
| `mse` | Mean squared error |
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `pearson_r` | Pearson correlation coefficient |
| `mcfadden_r2` | McFadden R² (goodness-of-fit vs. null model) |
| `demographics` | Demographic breakdown of the evaluation set |
| `training_demographics` | Demographic breakdown of the training set |

---

## `CoxProportionalHazard`

Survival analysis using lifelines' `CoxPHFitter`. Use for `EndpointType.SURVIVAL` outcomes.

```python
from src.train_model import CoxProportionalHazard

model = CoxProportionalHazard(
    data=df,
    id_column="who",
    target_column=["Ab_mokri_2016_TTE", "Ab_mokri_2016"]  # [duration_col, event_col]
)
model.selectFeatures()
model.train()
```

Feature selection uses Lasso regression (`model_type="regression"`, `alpha=30`).

**Evaluation dict keys**

| Key | Description |
|:---|:---|
| `concordance_index` | C-statistic: probability the model correctly ranks a random pair |
| `demographics` | Demographic breakdown of the evaluation set |
| `training_demographics` | Demographic breakdown of the training set |

!!! note
    Predictions from `CoxProportionalHazard` are median survival times from `CoxPHFitter.predict_median()`.
