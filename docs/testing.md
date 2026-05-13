# Testing

The pipeline has **144 tests** across **10 files** in two directories. All tests run with `pytest` and complete in under 5 seconds.

```bash
pytest tests/ unitTests/ -v
```

---

## Test Structure

```
tests/                         # Behavioural unit tests (126 tests)
├── test_constants.py          # EndpointType enum values
├── test_validate.py           # Dataset schema validation
├── test_input_validation.py   # Model constructor guards
├── test_data_ingestion.py     # CSV loading & data structure
├── test_holdout_split.py      # Held-out set construction
├── test_preprocess.py         # DataPreprocessor methods
├── test_train_model.py        # OutcomeModel & LogisticModel
└── test_model_training.py     # train_and_evaluate_models orchestration

unitTests/                     # End-to-end statistical tests (18 tests)
├── test_model_fit_and_eval.py # All four model classes, real metrics
└── test_psm.py                # PSM pipeline (holdout, matching, subsets)
```

---

## `tests/` — Behavioural Unit Tests

### `test_constants.py` — 5 tests

Verifies that the `EndpointType` enum is correctly defined and usable as a string-based lookup.

| Test | What it checks |
|:---|:---|
| `test_logical_value` | `EndpointType.LOGICAL.value == 'logical'` |
| `test_integer_value` | `EndpointType.INTEGER.value == 'integer'` |
| `test_survival_value` | `EndpointType.SURVIVAL.value == 'survival'` |
| `test_enum_lookup_by_string` | `EndpointType('logical')` returns the correct member |
| `test_invalid_string_raises` | An unrecognised string raises `ValueError` |

---

### `test_validate.py` — 13 tests

Covers `validate_dataset_for_model` — the schema check that runs before preprocessing.

=== "Valid inputs"

    | Test | What it checks |
    |:---|:---|
    | `test_valid_logical` | DataFrame with binary outcome passes |
    | `test_valid_integer` | DataFrame with integer outcome passes |
    | `test_valid_survival` | DataFrame with time + event columns passes |
    | `test_accepts_endpoint_type_enum_directly` | `EndpointType.LOGICAL` accepted instead of string |
    | `test_empty_dataframe_does_not_raise` | Empty DataFrame with correct column schema passes |

=== "Rejection cases"

    | Test | What it checks |
    |:---|:---|
    | `test_missing_outcome_column` | Raises when outcome column is absent |
    | `test_invalid_logical_values` | Raises when logical column contains values other than 0/1 |
    | `test_invalid_integer_dtype` | Raises when integer column contains floats |
    | `test_survival_missing_time_column` | Raises when time column is absent |
    | `test_survival_non_numeric_time` | Raises when time column contains strings |
    | `test_survival_event_not_binary` | Raises when event column is not 0/1 |
    | `test_unsupported_model_type` | Raises for an unrecognised model type string |
    | `test_survival_no_time_col_argument_raises` | Raises when `time_col` argument is omitted for survival |

---

### `test_input_validation.py` — 8 tests

Verifies that all four model classes (`LogisticModel`, `NegativeBinomialModel`, `CoxProportionalHazard`, `BetaRegression`) reject invalid constructor inputs before any computation starts. All classes share the same guard logic inherited from `OutcomeModel`.

| Test | What it checks |
|:---|:---|
| `test_empty_dataframe_raises_error` | Empty DataFrame raises `ValueError` |
| `test_none_dataframe_raises_error` | `None` raises `ValueError` or `TypeError` |
| `test_missing_id_column_raises_error` | Non-existent ID column name raises `ValueError` |
| `test_missing_target_column_raises_error` | Non-existent target column name raises `ValueError` |
| `test_target_column_not_list_raises_error` | Passing target as a string (not list) raises `TypeError` |
| `test_duplicate_ids_raises_error` | Duplicate IDs in the data raise `ValueError` |
| `test_valid_data_does_not_raise` | Valid data constructs the model without error |
| `test_validation_applies_to_all_model_types` | All four subclasses reject an empty DataFrame |

---

### `test_data_ingestion.py` — 9 tests

Tests CSV loading mechanics and basic data structure requirements using an in-memory temporary file.

=== "CSV Loading"

    | Test | What it checks |
    |:---|:---|
    | `test_load_csv_returns_dataframe` | `pd.read_csv` returns a DataFrame |
    | `test_load_csv_correct_row_count` | All rows are loaded without truncation |
    | `test_load_csv_preserves_columns` | Column names are preserved exactly |
    | `test_load_nonexistent_file_raises_error` | Missing file raises `FileNotFoundError` |
    | `test_load_csv_numeric_types_preserved` | Numeric columns remain numeric after load |

=== "Data Structure"

    | Test | What it checks |
    |:---|:---|
    | `test_id_column_is_unique` | ID column has no duplicate values |
    | `test_no_empty_dataframe` | DataFrame contains at least one row |
    | `test_binary_outcome_values` | Binary outcome column contains only 0 and 1 |
    | `test_age_within_bounds` | Age values fall within a plausible human range |

---

### `test_holdout_split.py` — 12 tests

Tests `holdOutTestData`, which constructs a fixed demographic held-out evaluation set from the training pool.

=== "Split Mechanics"

    | Test | What it checks |
    |:---|:---|
    | `test_returns_two_dataframes` | Function returns `(train_df, holdout_df)` |
    | `test_holdout_size_matches_requested` | Held-out set has exactly `testCount` rows |
    | `test_no_id_overlap` | No patient ID appears in both train and held-out sets |
    | `test_all_data_accounted_for` | `len(train) + len(holdout) == len(original)` |

=== "Demographic Ratios"

    | Test | What it checks |
    |:---|:---|
    | `test_default_ratio_is_58_42` | Default held-out set is 58% majority, 42% minority |
    | `test_custom_ratio_respected` | Custom `percentMajority` is honoured |
    | `test_majority_value_configurable` | The value used to identify the majority group is configurable |

=== "Data Integrity"

    | Test | What it checks |
    |:---|:---|
    | `test_columns_preserved` | All original columns present in both output DataFrames |
    | `test_seed_produces_reproducible_split` | Same seed → identical held-out IDs |
    | `test_different_seeds_produce_different_splits` | Different seeds → different held-out IDs |

=== "Edge Cases"

    | Test | What it checks |
    |:---|:---|
    | `test_train_set_larger_than_holdout` | Training set is always larger than held-out set |
    | `test_small_dataset_caps_holdout_to_available` | Sampling is capped when requested count exceeds available data |

---

### `test_preprocess.py` — 41 tests

Isolates each public method of `DataPreprocessor`. Each test class covers one method; all assertions are on `preprocessor.dataframe` after the in-place call.

=== "Column Operations"

    **`TestDropColumns`** (5 tests) — `drop_columns_and_return`

    Valid columns are dropped; columns not in the drop list survive; non-existent names are silently skipped; mixed valid/invalid lists are handled correctly.

    **`TestMoveColumnToEnd`** (4 tests) — `move_column_to_end`

    Target column(s) are repositioned to the end; all columns are preserved; non-existent column names are ignored.

    **`TestRenameColumns`** (4 tests) — `rename_columns`

    `Sex` → `is_female`, `job` → `unemployed`, `is_living_stable` → `unstableliving`. Unrelated columns are unchanged.

=== "Value Transformations"

    **`TestConvertYesNoToBinary`** (5 tests) — `convert_yes_no_to_binary`

    `Yes` → `1`, `No` → `0`. NaN values are preserved. Numeric columns are not touched.

    **`TestTransformNanToZeroForBinaryColumns`** (3 tests) — `transform_nan_to_zero_for_binary_columns`

    NaNs are filled with `0` only in columns whose non-null unique values are exactly `{0, 1}`. Columns with more than two unique values are left intact.

    **`TestTransformAndRenameColumn`** (4 tests) — `transform_and_rename_column`

    Non-null values become `1`, nulls become `0`. Column is renamed while preserving its positional index. Used to convert `heroin_inject_days` → `rbsivheroin`.

    **`TestFillNanWithZero`** (4 tests) — `fill_nan_with_zero`

    NaNs in the named column are replaced with `0`; other columns are untouched; a missing column name is a no-op.

    **`TestConvertUdsToBinary`** (3 tests) — `convert_uds_to_binary`

    `UDS_*` columns are binarised (count > 0 → 1, else → 0). Non-UDS columns are untouched.

=== "Feature Engineering"

    **`TestProcessTLFBColumns`** (3 tests) — `process_tlfb_columns`

    Unspecified `TLFB_*` columns are row-summed into `TLFB_Other` and removed. Specified columns are retained. Tests verify both the computed sum and removal of source columns.

    **`TestTransformDataWithNanHandling`** (6 tests) — `transform_data_with_nan_handling`

    Verifies the full categorical encoding pass: `Sex` (female=1), `education` (1–3), `marital` (2–4, NaN→1), `race` (White=1, Black=2, Other=3, Refused=0, NaN=−1), `XTRT`, `RaceEth`, `pain`, `job`, `is_living_stable`. Missing columns are skipped without raising.

---

### `test_train_model.py` — 26 tests

Tests `OutcomeModel` (base class) and `LogisticModel` step by step through initialisation, feature selection, training, and evaluation.

=== "Train/Test Split"

    **`TestTrainTestSplit`** (5 tests)

    Verifies the internal 75/25 split: both sets are non-empty, the ratio is within tolerance, there is no index overlap, all data is accounted for, and the same seed always produces the same partition.

=== "Feature Handling"

    **`TestFeatureHandling`** (3 tests)

    Confirms that the ID column and target column are excluded from the feature matrix `X`, and that all other columns are preserved as features.

=== "Demographic Counting"

    **`TestDemographicCounting`** (2 tests)

    `_countDemographic` returns a formatted string (e.g. `"120 NHW, 50 NHB"`) containing numeric counts. Used for bias-tracking in evaluation outputs.

=== "Feature Selection"

    **`TestLogisticModelFeatureSelection`** (4 tests)

    After `selectFeatures()`: the `selected_features` list is non-empty, every feature exists in `X.columns`, ID and target are excluded, and repeated calls produce the same result (deterministic Lasso).

=== "Training"

    **`TestLogisticModelTraining`** (5 tests)

    After `train()`: the `model` attribute is populated, `best_threshold` is a probability in [0, 1] equal to the positive-class proportion in the internal test set, and `predict_proba` returns a valid `(n, 2)` probability array.

=== "Evaluation"

    **`TestLogisticModelEvaluation`** (7 tests)

    `evaluate()` returns a 4-tuple `(heldout_preds, heldout_evals, subset_preds, subset_evals)`. Both evaluation dicts contain `roc`, `confusion_matrix`, `precision`, `recall`, `demographics`, and `training_demographics`. ROC-AUC is a float in [0, 1]. Calling `evaluate()` before `train()` raises `NotFittedError`.

---

### `test_model_training.py` — 12 tests

Tests the `train_and_evaluate_models` orchestration function. All model classes are **mocked** so these tests cover routing logic and result shape, not statistical correctness.

=== "Model Selection"

    **`TestModelSelection`** (4 tests)

    `LOGICAL` → `LogisticModel`, `INTEGER` → `NegativeBinomialModel`, `SURVIVAL` → `CoxProportionalHazard`. Also verifies the model is constructed with the correct positional arguments `(data, id_column, target_columns)`.

=== "Results Structure"

    **`TestResultsStructure`** (4 tests)

    The return value is a `DataFrame` with a two-level `MultiIndex` on columns: level-0 in `{heldout, subset}`, level-1 in `{predictions, evaluations}`, giving exactly four columns. An empty subset list returns an empty frame with the same MultiIndex structure.

=== "Subset Processing"

    **`TestSubsetProcessing`** (4 tests)

    One result row is produced per subset. `selectFeatures`, `train`, and `evaluate` are each called exactly once per subset. `evaluate` always receives the shared held-out DataFrame, not subset-specific data.

---

## `unitTests/` — End-to-End Statistical Tests

These tests run the full `selectFeatures → train → evaluate` pipeline on synthetic data and assert on concrete metric values.

### `test_model_fit_and_eval.py` — 15 tests

=== "Logistic (LOGICAL)"

    | Test | What it checks |
    |:---|:---|
    | `test_logistic_model_init` | Feature matrix shape, ID/target separation, 75/25 split sizes |
    | `test_logistic_model_select_features_one_feature` | Lasso correctly selects the single correlated feature |
    | `test_logistic_model_select_features_multiple_features` | Lasso selects all three correlated features |
    | `test_logistic_model_train` | Model is fitted and exposes `predict_proba` |
    | `test_logistic_model_evaluation` | ROC-AUC, confusion matrix, precision, recall, and demographic strings match expected values |

=== "Negative Binomial (INTEGER)"

    | Test | What it checks |
    |:---|:---|
    | `test_negative_binomial_model_select_features` | Lasso selects `feature2` and `feature3` |
    | `test_negative_binomial_model_train` | Model is fitted and exposes `predict` |
    | `test_negative_binomial_model_evaluation` | MSE, RMSE, MAE, McFadden R², and demographic strings match expected values |

=== "Cox Proportional Hazard (SURVIVAL)"

    | Test | What it checks |
    |:---|:---|
    | `test_cox_proportional_hazard_select_features` | Lasso selects `feature2` and `feature3` |
    | `test_cox_proportional_hazard_train` | Model is fitted and exposes `predict_median` |
    | `test_cox_proportional_hazard_evaluation` | Concordance index and demographic strings match expected values |

=== "Beta Regression ([0,1] continuous)"

    | Test | What it checks |
    |:---|:---|
    | `test_beta_regression_select_features` | Lasso selects `feature2` and `feature3` |
    | `test_beta_regression_train` | Model is fitted and exposes `predict` |
    | `test_beta_regression_evaluation` | MSE, RMSE, MAE, Pearson r, and demographic strings match expected values |

---

### `test_psm.py` — 4 tests

| Test | What it checks |
|:---|:---|
| `test_holdOutTestData` | Returns train/held-out pair, held-out ≤ 100 rows, no ID overlap, majority/minority counts within bounds |
| `test_propensityScoreMatch` | Python merge/reshape logic (R call monkeypatched); output is a list of DataFrames each containing the ID column |
| `test_propensityScoreMatchLogic` | Real R MatchIt call on a 9-row dataset; returns 3 match groups of 3 rows each (1 minority + 2 controls) |
| `test_create_subsets` | Produces the correct number of subsets with the correct schema |

!!! note "R dependency"
    `test_propensityScoreMatchLogic` requires a working R installation with `MatchIt` installed. All other tests run without R.
