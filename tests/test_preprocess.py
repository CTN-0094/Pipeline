"""
Unit tests for DataPreprocessor (src/preprocess.py).

Each test class isolates one public method of DataPreprocessor. Tests operate
on small in-memory DataFrames and assert on the mutated state of
preprocessor.dataframe after the method call, since all methods modify the
DataFrame in place rather than returning a new one.

Test classes:
    TestDropColumns                  — drop_columns_and_return
    TestConvertYesNoToBinary         — convert_yes_no_to_binary
    TestProcessTLFBColumns           — process_tlfb_columns
    TestMoveColumnToEnd              — move_column_to_end
    TestRenameColumns                — rename_columns
    TestTransformNanToZeroForBinaryColumns — transform_nan_to_zero_for_binary_columns
    TestTransformAndRenameColumn     — transform_and_rename_column
    TestFillNanWithZero              — fill_nan_with_zero
    TestTransformDataWithNanHandling — transform_data_with_nan_handling
    TestConvertUdsToBinary           — convert_uds_to_binary
"""

import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import DataPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "age": [25, 30, 45],
        "score": [10.0, 20.0, 30.0],
        "label": [0, 1, 0],
    })


@pytest.fixture
def yes_no_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "smoker": ["Yes", "No", "Yes", np.nan],
        "drinker": ["No", "No", "Yes", "Yes"],
        "age": [25, 30, 35, 40],
    })


@pytest.fixture
def tlfb_df():
    return pd.DataFrame({
        "id": [1, 2],
        "TLFB_Alcohol_Count": [3, 1],
        "TLFB_Cocaine_Count": [0, 2],
        "TLFB_Heroin_Count": [1, 0],
        "TLFB_Other_Drug": [5, 3],
        "TLFB_Misc": [2, 1],
    })


@pytest.fixture
def binary_nan_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "binary_col": [1.0, 0.0, np.nan, 1.0],
        "multi_col": [1.0, 2.0, np.nan, 3.0],
        "no_nan_col": [0.0, 1.0, 0.0, 1.0],
    })


@pytest.fixture
def rename_df():
    return pd.DataFrame({
        "Sex": ["male", "female"],
        "job": ["Full Time", "Unemployed"],
        "is_living_stable": [1, 0],
        "age": [30, 45],
    })


@pytest.fixture
def transform_rename_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "heroin_inject_days": [np.nan, 5.0, 10.0],
        "age": [25, 30, 35],
    })


# ---------------------------------------------------------------------------
# drop_columns_and_return
# ---------------------------------------------------------------------------

class TestDropColumns:
    """Columns in the drop list are removed; columns not in the list survive; missing names are silently skipped."""

    def test_drops_valid_columns(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.drop_columns_and_return(["score"])
        assert "score" not in pre.dataframe.columns

    def test_retains_other_columns(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.drop_columns_and_return(["score"])
        assert "id" in pre.dataframe.columns
        assert "age" in pre.dataframe.columns
        assert "label" in pre.dataframe.columns

    def test_drops_multiple_columns(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.drop_columns_and_return(["score", "age"])
        assert "score" not in pre.dataframe.columns
        assert "age" not in pre.dataframe.columns

    def test_ignores_nonexistent_columns(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.drop_columns_and_return(["nonexistent"])
        assert list(pre.dataframe.columns) == ["id", "age", "score", "label"]

    def test_mixed_valid_and_invalid(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.drop_columns_and_return(["score", "nonexistent"])
        assert "score" not in pre.dataframe.columns
        assert "id" in pre.dataframe.columns


# ---------------------------------------------------------------------------
# convert_yes_no_to_binary
# ---------------------------------------------------------------------------

class TestConvertYesNoToBinary:
    """Yes→1, No→0 for columns whose only non-null values are 'Yes'/'No'; NaNs are preserved; numeric columns are untouched."""

    def test_converts_yes_to_1(self, yes_no_df):
        pre = DataPreprocessor(yes_no_df.copy())
        pre.convert_yes_no_to_binary()
        assert pre.dataframe.loc[0, "smoker"] == 1
        assert pre.dataframe.loc[0, "drinker"] == 0

    def test_converts_no_to_0(self, yes_no_df):
        pre = DataPreprocessor(yes_no_df.copy())
        pre.convert_yes_no_to_binary()
        assert pre.dataframe.loc[1, "smoker"] == 0

    def test_preserves_nan(self, yes_no_df):
        pre = DataPreprocessor(yes_no_df.copy())
        pre.convert_yes_no_to_binary()
        assert pd.isna(pre.dataframe.loc[3, "smoker"])

    def test_does_not_touch_numeric_column(self, yes_no_df):
        pre = DataPreprocessor(yes_no_df.copy())
        original_ages = yes_no_df["age"].tolist()
        pre.convert_yes_no_to_binary()
        assert pre.dataframe["age"].tolist() == original_ages

    def test_both_columns_converted(self, yes_no_df):
        pre = DataPreprocessor(yes_no_df.copy())
        pre.convert_yes_no_to_binary()
        assert pre.dataframe["smoker"].dtype in [np.float64, np.int64, object]
        assert set(pre.dataframe["drinker"].dropna().unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# process_tlfb_columns
# ---------------------------------------------------------------------------

class TestProcessTLFBColumns:
    """Unspecified TLFB_* columns are row-summed into TLFB_Other and removed; specified columns are kept as-is."""

    def test_other_tlfb_columns_summed_into_tlfb_other(self, tlfb_df):
        pre = DataPreprocessor(tlfb_df.copy())
        specified = ["TLFB_Alcohol_Count", "TLFB_Cocaine_Count", "TLFB_Heroin_Count"]
        pre.process_tlfb_columns(specified)
        assert "TLFB_Other" in pre.dataframe.columns
        assert pre.dataframe.loc[0, "TLFB_Other"] == 7   # 5 + 2
        assert pre.dataframe.loc[1, "TLFB_Other"] == 4   # 3 + 1

    def test_unspecified_tlfb_columns_removed(self, tlfb_df):
        pre = DataPreprocessor(tlfb_df.copy())
        specified = ["TLFB_Alcohol_Count", "TLFB_Cocaine_Count", "TLFB_Heroin_Count"]
        pre.process_tlfb_columns(specified)
        assert "TLFB_Other_Drug" not in pre.dataframe.columns
        assert "TLFB_Misc" not in pre.dataframe.columns

    def test_specified_columns_retained(self, tlfb_df):
        pre = DataPreprocessor(tlfb_df.copy())
        specified = ["TLFB_Alcohol_Count", "TLFB_Cocaine_Count", "TLFB_Heroin_Count"]
        pre.process_tlfb_columns(specified)
        for col in specified:
            assert col in pre.dataframe.columns


# ---------------------------------------------------------------------------
# move_column_to_end
# ---------------------------------------------------------------------------

class TestMoveColumnToEnd:
    """Target column(s) are repositioned to the end of the DataFrame; all other columns and row count are unchanged."""

    def test_moves_single_column_to_end(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.move_column_to_end(["age"])
        assert pre.dataframe.columns[-1] == "age"

    def test_moves_multiple_columns_to_end(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.move_column_to_end(["id", "age"])
        assert set(pre.dataframe.columns[-2:]) == {"id", "age"}

    def test_nonexistent_column_is_ignored(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.move_column_to_end(["nonexistent"])
        assert list(pre.dataframe.columns) == ["id", "age", "score", "label"]

    def test_all_columns_preserved(self, simple_df):
        pre = DataPreprocessor(simple_df.copy())
        pre.move_column_to_end(["label"])
        assert set(pre.dataframe.columns) == {"id", "age", "score", "label"}


# ---------------------------------------------------------------------------
# rename_columns
# ---------------------------------------------------------------------------

class TestRenameColumns:
    """Sex→is_female, job→unemployed, is_living_stable→unstableliving; unrelated columns are unchanged."""

    def test_sex_renamed_to_is_female(self, rename_df):
        pre = DataPreprocessor(rename_df.copy())
        pre.rename_columns()
        assert "is_female" in pre.dataframe.columns
        assert "Sex" not in pre.dataframe.columns

    def test_job_renamed_to_unemployed(self, rename_df):
        pre = DataPreprocessor(rename_df.copy())
        pre.rename_columns()
        assert "unemployed" in pre.dataframe.columns
        assert "job" not in pre.dataframe.columns

    def test_is_living_stable_renamed(self, rename_df):
        pre = DataPreprocessor(rename_df.copy())
        pre.rename_columns()
        assert "unstableliving" in pre.dataframe.columns
        assert "is_living_stable" not in pre.dataframe.columns

    def test_other_columns_unchanged(self, rename_df):
        pre = DataPreprocessor(rename_df.copy())
        pre.rename_columns()
        assert "age" in pre.dataframe.columns


# ---------------------------------------------------------------------------
# transform_nan_to_zero_for_binary_columns
# ---------------------------------------------------------------------------

class TestTransformNanToZeroForBinaryColumns:
    """NaNs are filled with 0 only in columns whose non-null unique values are exactly {0, 1}; multi-value columns with NaNs are left intact."""

    def test_nan_in_binary_column_filled_with_zero(self, binary_nan_df):
        pre = DataPreprocessor(binary_nan_df.copy())
        pre.transform_nan_to_zero_for_binary_columns()
        assert pre.dataframe["binary_col"].isna().sum() == 0
        assert pre.dataframe.loc[2, "binary_col"] == 0.0

    def test_non_binary_column_with_nan_not_modified(self, binary_nan_df):
        pre = DataPreprocessor(binary_nan_df.copy())
        pre.transform_nan_to_zero_for_binary_columns()
        assert pd.isna(pre.dataframe.loc[2, "multi_col"])

    def test_column_without_nan_unchanged(self, binary_nan_df):
        pre = DataPreprocessor(binary_nan_df.copy())
        original = binary_nan_df["no_nan_col"].tolist()
        pre.transform_nan_to_zero_for_binary_columns()
        assert pre.dataframe["no_nan_col"].tolist() == original


# ---------------------------------------------------------------------------
# transform_and_rename_column
# ---------------------------------------------------------------------------

class TestTransformAndRenameColumn:
    """Non-null values become 1, NULLs become 0; column is renamed in place preserving its positional index."""

    def test_non_null_values_become_1(self, transform_rename_df):
        pre = DataPreprocessor(transform_rename_df.copy())
        pre.transform_and_rename_column("heroin_inject_days", "rbsivheroin")
        assert pre.dataframe.loc[1, "rbsivheroin"] == 1
        assert pre.dataframe.loc[2, "rbsivheroin"] == 1

    def test_null_values_become_0(self, transform_rename_df):
        pre = DataPreprocessor(transform_rename_df.copy())
        pre.transform_and_rename_column("heroin_inject_days", "rbsivheroin")
        assert pre.dataframe.loc[0, "rbsivheroin"] == 0

    def test_column_is_renamed(self, transform_rename_df):
        pre = DataPreprocessor(transform_rename_df.copy())
        pre.transform_and_rename_column("heroin_inject_days", "rbsivheroin")
        assert "rbsivheroin" in pre.dataframe.columns
        assert "heroin_inject_days" not in pre.dataframe.columns

    def test_column_position_preserved(self, transform_rename_df):
        pre = DataPreprocessor(transform_rename_df.copy())
        original_pos = list(transform_rename_df.columns).index("heroin_inject_days")
        pre.transform_and_rename_column("heroin_inject_days", "rbsivheroin")
        new_pos = list(pre.dataframe.columns).index("rbsivheroin")
        assert original_pos == new_pos


# ---------------------------------------------------------------------------
# fill_nan_with_zero
# ---------------------------------------------------------------------------

class TestFillNanWithZero:
    """NaNs in the named column are replaced with 0; all other columns are untouched; a missing column name is a no-op (no raise)."""

    def test_nan_filled_with_zero(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        pre = DataPreprocessor(df.copy())
        pre.fill_nan_with_zero("a")
        assert pre.dataframe["a"].isna().sum() == 0
        assert pre.dataframe.loc[1, "a"] == 0.0

    def test_other_columns_untouched(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        pre = DataPreprocessor(df.copy())
        pre.fill_nan_with_zero("a")
        assert pre.dataframe["b"].isna().sum() == 2

    def test_nonexistent_column_does_not_raise(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        pre = DataPreprocessor(df.copy())
        pre.fill_nan_with_zero("nonexistent")  # should print warning, not raise

    def test_column_without_nan_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pre = DataPreprocessor(df.copy())
        pre.fill_nan_with_zero("a")
        assert pre.dataframe["a"].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# transform_data_with_nan_handling
# (modifies self.dataframe in place, no return value)
# ---------------------------------------------------------------------------

class TestTransformDataWithNanHandling:
    """
    Verifies the full categorical encoding pass: Sex (female=1), education (1-3),
    marital status (2-4, NaN→1), race (White=1, Black=2, Other=3, Refused=0, NaN=-1),
    XTRT, RaceEth, pain, job, and is_living_stable. Missing columns are silently skipped.
    """

    def test_sex_female_becomes_1(self):
        df = pd.DataFrame({"Sex": ["female", "male", "male"]})
        pre = DataPreprocessor(df.copy())
        pre.transform_data_with_nan_handling()
        assert pre.dataframe["Sex"].tolist() == [1, 0, 0]

    def test_education_mapping(self):
        df = pd.DataFrame({"education": ["Less than HS", "HS/GED", "More than HS", np.nan]})
        pre = DataPreprocessor(df.copy())
        pre.transform_data_with_nan_handling()
        assert pre.dataframe["education"].tolist() == [1.0, 2.0, 3.0, 0.0]

    def test_marital_mapping(self):
        df = pd.DataFrame({"marital": ["Never married", "Married or Partnered", "Separated/Divorced/Widowed", np.nan]})
        pre = DataPreprocessor(df.copy())
        pre.transform_data_with_nan_handling()
        assert pre.dataframe["marital"].tolist() == [2.0, 3.0, 4.0, 1.0]

    def test_race_mapping(self):
        df = pd.DataFrame({"race": ["White", "Black", "Other", "Refused/missing", np.nan]})
        pre = DataPreprocessor(df.copy())
        pre.transform_data_with_nan_handling()
        assert pre.dataframe["race"].tolist() == [1.0, 2.0, 3.0, 0.0, -1.0]

    def test_missing_columns_skipped_without_error(self):
        df = pd.DataFrame({"unrelated": [1, 2, 3]})
        pre = DataPreprocessor(df.copy())
        pre.transform_data_with_nan_handling()  # should not raise
        assert list(pre.dataframe.columns) == ["unrelated"]

    def test_modifies_dataframe_in_place(self):
        df = pd.DataFrame({"Sex": ["female", "male"]})
        pre = DataPreprocessor(df.copy())
        pre.transform_data_with_nan_handling()
        assert pre.dataframe["Sex"].tolist() == [1, 0]


# ---------------------------------------------------------------------------
# convert_uds_to_binary
# ---------------------------------------------------------------------------

class TestConvertUdsToBinary:
    """UDS_* columns are binarised (count > 0 → 1, else 0); non-UDS columns are untouched."""

    def test_uds_above_zero_becomes_1(self):
        df = pd.DataFrame({"UDS_Alcohol_Count": [0, 3, 1], "age": [25, 30, 35]})
        pre = DataPreprocessor(df.copy())
        pre.convert_uds_to_binary()
        assert pre.dataframe["UDS_Alcohol_Count"].tolist() == [0, 1, 1]

    def test_uds_zero_stays_zero(self):
        df = pd.DataFrame({"UDS_Cocaine_Count": [0, 0, 5]})
        pre = DataPreprocessor(df.copy())
        pre.convert_uds_to_binary()
        assert pre.dataframe.loc[0, "UDS_Cocaine_Count"] == 0
        assert pre.dataframe.loc[2, "UDS_Cocaine_Count"] == 1

    def test_non_uds_columns_untouched(self):
        df = pd.DataFrame({"UDS_Alcohol_Count": [2, 0], "age": [30, 40]})
        pre = DataPreprocessor(df.copy())
        pre.convert_uds_to_binary()
        assert pre.dataframe["age"].tolist() == [30, 40]
