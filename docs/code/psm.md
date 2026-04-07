# PSM & Cohort Construction

**File:** `src/create_demodf_knn.py`

Handles stratified held-out set creation and propensity score matching (PSM) to construct balanced training cohorts. PSM is performed using R's `MatchIt` package via `rpy2`.

---

## `holdOutTestData`

```python
holdOutTestData(df, id_column, testCount=100, columnToSplit='RaceEth',
                majorityValue=1, percentMajority=58, seed=42)
```

Stratifies and separates a held-out evaluation set from the main dataset before PSM runs. The held-out set is fixed for the duration of the experiment and is never used in training.

**Parameters**

| Name | Type | Default | Description |
|:---|:---|:---:|:---|
| `df` | `pd.DataFrame` | — | Full input dataset |
| `id_column` | `str` | — | Name of the unique patient ID column |
| `testCount` | `int` | `100` | Total number of held-out samples |
| `columnToSplit` | `str` | `"RaceEth"` | Column defining majority/minority groups |
| `majorityValue` | `int` | `1` | Value in `columnToSplit` representing the majority group |
| `percentMajority` | `int` | `58` | Percentage of held-out set drawn from the majority group |
| `seed` | `int` | `42` | Random seed for reproducibility |

**Returns:** `(train_df, test_df)` — training pool and held-out set as DataFrames.

---

## `propensityScoreMatch`

```python
propensityScoreMatch(df, idColumn, columnToSplit='RaceEth', majorityValue=1,
                     columnsToMatch=['age', 'is_female'], sampleSize=500)
```

Runs PSM on the training pool to produce a matched set of minority and majority participants. Each minority participant is matched to 2 majority participants on `columnsToMatch`.

**Parameters**

| Name | Type | Default | Description |
|:---|:---|:---:|:---|
| `df` | `pd.DataFrame` | — | Training pool (after held-out removal) |
| `idColumn` | `str` | — | Unique patient ID column |
| `columnToSplit` | `str` | `"RaceEth"` | Column defining majority/minority groups |
| `majorityValue` | `int` | `1` | Value representing the majority group |
| `columnsToMatch` | `list` | `["age", "is_female"]` | Features to match on during PSM |
| `sampleSize` | `int` | `500` | Number of minority participants to match |

**Returns:** List of 3 DataFrames — `[minority_df, majority_df_1, majority_df_2]` — one minority group and two matched majority groups.

!!! note "R dependency"
    This function calls R's `MatchIt` package using `rpy2`. If `MatchIt` is not installed, the pipeline will prompt you to install it automatically on first run.

---

## `create_subsets`

```python
create_subsets(dfs, splits=11, sampleSize=500)
```

Combines the PSM-matched DataFrames into a series of training cohorts at varying majority/minority ratios. Produces `splits` cohorts spanning from 100% minority to 100% majority composition.

**Parameters**

| Name | Type | Default | Description |
|:---|:---|:---:|:---|
| `dfs` | `list[pd.DataFrame]` | — | Output of `propensityScoreMatch` — 3 DataFrames |
| `splits` | `int` | `11` | Number of cohorts to construct |
| `sampleSize` | `int` | `500` | Size of each cohort |

**Returns:** List of `splits` DataFrames, one per demographic ratio.

### Ratio progression (default 11 splits)

| Subset | Minority % | Majority % |
|:---:|:---:|:---:|
| 1 | 0% | 100% |
| 2 | 10% | 90% |
| 3 | 20% | 80% |
| ... | ... | ... |
| 6 | 50% | 50% |
| ... | ... | ... |
| 11 | 100% | 0% |
