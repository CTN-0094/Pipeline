"""
Integration-level tests for preprocess_data() in src/preprocess_pipeline.py.

preprocess_data() is the orchestrator that chains all DataPreprocessor methods
in a fixed order and returns the fully processed DataFrame. Unlike the unit
tests in test_preprocess.py (which isolate individual methods), these tests
verify the contract of the assembled pipeline:

    - Columns slated for removal are absent from the output
    - heroin_inject_days is transformed and renamed to rbsivheroin
    - UDS count columns are binarised
    - Yes/No string columns are converted to numeric
    - The function returns a non-empty DataFrame without raising
    - The outcome column is present in the output

A minimal fixture DataFrame is used; columns not present are silently skipped
by the individual DataPreprocessor methods, so only the behaviours under test
need to be supplied.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess_pipeline import preprocess_data


OUTCOME_COL = "ctn0094_relapse_event"

FIRST_DROP_COLS = [
    "pain_when", "is_smoker", "per_day", "max", "amount",
    "depression", "anxiety", "schizophrenia",
    "cocaine_inject_days", "speedball_inject_days",
    "opioid_inject_days", "speed_inject_days",
    "UDS_Alcohol_Count", "UDS_Mdma/Hallucinogen_Count",
]

SECOND_DROP_COLS = [
    "rbs_iv_days", "race", "RBS_cocaine_Days", "RBS_heroin_Days",
    "RBS_opioid_Days", "RBS_speed_Days", "RBS_speedball_Days",
]


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame that exercises every stage of preprocess_data().

    Includes the columns required by calculate_behavioral_columns (msm_npt,
    Sex, txx_prt), the columns targeted by transform_and_rename_column and
    fill_nan_with_zero, a UDS count column, a Yes/No column, and enough of the
    first/second drop-list columns to verify removal.
    """
    n = 6
    return pd.DataFrame({
        # --- identity / outcome ---
        "who": range(n),
        OUTCOME_COL: [0, 1, 0, 1, 0, 1],

        # --- Yes/No column that should be binarised ---
        "iv_drug_use": ["Yes", "No", "Yes", "No", "Yes", np.nan],

        # --- required by calculate_behavioral_columns ---
        "msm_npt": [0.0, 1.0, np.nan, 0.0, 2.0, 0.0],
        "Sex": ["male", "female", "male", "male", "female", "male"],
        "txx_prt": [1, 3, np.nan, 1, 2, 0],

        # --- targeted by transform_and_rename_column ---
        "heroin_inject_days": [np.nan, 5.0, 0.0, np.nan, 3.0, 0.0],

        # --- targeted by fill_nan_with_zero ---
        "ftnd": [np.nan, 3.0, 1.0, np.nan, 5.0, 2.0],

        # --- UDS count column → should become binary ---
        "UDS_Heroin_Count": [0, 3, 1, 0, 2, 0],

        # --- first drop-list samples ---
        "pain_when": ["always", "never", "often", "sometimes", "always", "never"],
        "depression": [1, 0, 1, 0, 1, 0],

        # --- second drop-list samples ---
        "rbs_iv_days": [0, 1, 0, 0, 2, 0],
        "race": ["White", "Black", "White", "Other", "White", "Black"],
    })


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

class TestPreprocessDataReturnsValidDataFrame:
    """preprocess_data must return a non-empty DataFrame with the same row count."""

    def test_returns_dataframe(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert len(result) == len(minimal_df)

    def test_result_is_not_empty(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert not result.empty

    def test_outcome_column_present_in_output(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert OUTCOME_COL in result.columns


# ---------------------------------------------------------------------------
# Column removal
# ---------------------------------------------------------------------------

class TestDroppedColumns:
    """Columns from both drop passes must be absent from the output."""

    def test_first_drop_pass_columns_removed(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        for col in ["pain_when", "depression"]:
            assert col not in result.columns, f"Expected '{col}' to be dropped"

    def test_second_drop_pass_columns_removed(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        for col in ["rbs_iv_days", "race"]:
            assert col not in result.columns, f"Expected '{col}' to be dropped"

    def test_intermediate_behavioral_drop_cols_removed(self, minimal_df: pd.DataFrame) -> None:
        # msm_npt, msm_frq, txx_prt are dropped in the second column pass
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        for col in ["msm_npt", "txx_prt"]:
            assert col not in result.columns, f"Expected '{col}' to be dropped"


# ---------------------------------------------------------------------------
# Column transformations
# ---------------------------------------------------------------------------

class TestColumnTransformations:
    """Verify key per-column transformations applied by the orchestrator."""

    def test_heroin_inject_days_renamed_to_rbsivheroin(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert "rbsivheroin" in result.columns
        assert "heroin_inject_days" not in result.columns

    def test_heroin_inject_null_becomes_0(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        # Rows 0 and 3 had NaN heroin_inject_days → should become 0
        assert result["rbsivheroin"].iloc[0] == 0
        assert result["rbsivheroin"].iloc[3] == 0

    def test_heroin_inject_nonzero_becomes_1(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        # Rows 1 and 4 had non-null, non-zero values → should become 1
        assert result["rbsivheroin"].iloc[1] == 1
        assert result["rbsivheroin"].iloc[4] == 1

    def test_ftnd_nan_filled_with_zero(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert "ftnd" in result.columns
        assert result["ftnd"].isna().sum() == 0

    def test_uds_count_column_binarised(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert "UDS_Heroin_Count" in result.columns
        unique_vals = set(result["UDS_Heroin_Count"].dropna().unique())
        assert unique_vals.issubset({0, 1}), f"Expected only 0/1, got {unique_vals}"

    def test_yes_no_column_converted_to_numeric(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert "iv_drug_use" in result.columns
        non_null = result["iv_drug_use"].dropna()
        assert set(non_null.unique()).issubset({0, 1}), (
            f"Expected only 0/1, got {set(non_null.unique())}"
        )

    def test_behavioral_columns_added(self, minimal_df: pd.DataFrame) -> None:
        result = preprocess_data(minimal_df.copy(), OUTCOME_COL)
        assert "Homosexual_Behavior" in result.columns
        assert "Non_monogamous_Relationships" in result.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """preprocess_data should be robust to missing optional columns."""

    def test_runs_without_drop_list_columns(self) -> None:
        """A DataFrame that lacks all drop-list columns should not raise."""
        df = pd.DataFrame({
            "who": [1, 2, 3],
            OUTCOME_COL: [0, 1, 0],
            "msm_npt": [0.0, 1.0, 0.0],
            "Sex": ["male", "female", "male"],
            "txx_prt": [1, 2, 0],
        })
        result = preprocess_data(df.copy(), OUTCOME_COL)
        assert isinstance(result, pd.DataFrame)

    def test_runs_without_tlfb_columns(self) -> None:
        """A DataFrame with no TLFB columns should not raise."""
        df = pd.DataFrame({
            "who": [1, 2],
            OUTCOME_COL: [0, 1],
            "msm_npt": [0.0, 1.0],
            "Sex": ["male", "female"],
            "txx_prt": [1, 2],
        })
        result = preprocess_data(df.copy(), OUTCOME_COL)
        assert not result.empty

    def test_all_nan_heroin_column(self) -> None:
        """All-NaN heroin_inject_days should become all-zero rbsivheroin."""
        df = pd.DataFrame({
            "who": [1, 2, 3],
            OUTCOME_COL: [0, 1, 0],
            "msm_npt": [0.0, 0.0, 0.0],
            "Sex": ["male", "female", "male"],
            "txx_prt": [1, 1, 1],
            "heroin_inject_days": [np.nan, np.nan, np.nan],
        })
        result = preprocess_data(df.copy(), OUTCOME_COL)
        assert "rbsivheroin" in result.columns
        assert (result["rbsivheroin"] == 0).all()
