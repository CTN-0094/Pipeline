# Preprocessing

**File:** `src/preprocess.py`

The `DataPreprocessor` class handles all feature engineering and data cleaning steps before PSM and model training. It operates in-place on an internal DataFrame and exposes each transformation as a separate method.

---

## `DataPreprocessor`

```python
from src.preprocess import DataPreprocessor

pre = DataPreprocessor(dataframe=df)
```

| Parameter | Type | Description |
|:---|:---|:---|
| `dataframe` | `pd.DataFrame` | Input dataset to preprocess |

The preprocessor stores the DataFrame as `self.dataframe`. All methods modify it in place.

---

### `drop_columns_and_return`

```python
pre.drop_columns_and_return(columns_to_drop)
```

Removes specified columns from the DataFrame. Silently skips columns that don't exist.

| Parameter | Type | Description |
|:---|:---|:---|
| `columns_to_drop` | `list[str]` | Column names to remove |

---

### `convert_yes_no_to_binary`

```python
pre.convert_yes_no_to_binary()
```

Finds all columns containing only `"Yes"` / `"No"` values (plus NaN) and converts them to `1` / `0` integers. Useful for encoding survey-style columns automatically.

---

### `transform_nan_to_zero_for_binary_columns`

```python
pre.transform_nan_to_zero_for_binary_columns()
```

Fills `NaN` values with `0` in binary columns (columns containing only 0, 1, and NaN). Treats missingness as absence of the event.

---

### `one_hot_encode_raceeth`

```python
pre.one_hot_encode_raceeth()
```

Adds a 0/1 indicator column for each non-reference `RaceEth` group: `RaceEth_NHB`, `RaceEth_Hisp`, `RaceEth_Other` and `RaceEth_Missing`. NHW is the reference group. The column set is fixed, so every dataset gets the same columns even when a group is absent. The raw `RaceEth` code is kept for PSM and demographic summaries, but the models never use it as a feature. Run this after `transform_data_with_nan_handling`, which maps the labels to codes.

---

### Usage pattern

Methods are typically chained in sequence during pipeline setup:

```python
pre = DataPreprocessor(df)
pre.convert_yes_no_to_binary()
pre.transform_nan_to_zero_for_binary_columns()
pre.drop_columns_and_return(["pain_when", "is_smoker", "per_day"])

cleaned_df = pre.dataframe
```
