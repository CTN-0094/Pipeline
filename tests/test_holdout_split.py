"""
Unit tests for holdout data splitting.

This module tests the holdOutTestData function which creates
a fixed diverse holdout set for model evaluation. The holdout
maintains a specific demographic ratio (default 58% majority,
42% minority) to enable fair bias comparisons across models.

Test Categories:
    - Split Mechanics: Verifies train/holdout separation
    - Demographic Ratios: Ensures correct majority/minority proportions
    - Data Integrity: Confirms no data leakage or loss
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.create_demodf_knn import holdOutTestData


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_demographic_data():
    """
    Create sample data with demographic groups for holdout testing.

    Generates a DataFrame with:
        - id: Unique identifier for each row
        - RaceEth: Demographic group (1=majority ~60%, 2,3=minority ~40%)
        - age: Random ages
        - outcome: Binary outcome

    Returns:
        pd.DataFrame: Sample dataset with 500 rows
    """
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'id': range(1, n + 1),
        'RaceEth': np.random.choice([1, 2, 3], n, p=[0.6, 0.25, 0.15]),
        'age': np.random.randint(18, 65, n),
        'outcome': np.random.choice([0, 1], n),
    })


# =============================================================================
# SPLIT MECHANICS TESTS
# =============================================================================

class TestSplitMechanics:
    """
    Tests for basic train/holdout splitting mechanics.

    These tests verify that the function correctly separates
    data into training and holdout sets.
    """

    def test_returns_two_dataframes(self, sample_demographic_data):
        """
        Verify function returns both train and holdout DataFrames.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id'
        )

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(holdout_df, pd.DataFrame)

    def test_holdout_size_matches_requested(self, sample_demographic_data):
        """
        Verify holdout set has the requested number of samples.

        Default testCount is 100, so holdout should have 100 rows.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            testCount=100
        )

        assert len(holdout_df) == 100

    def test_no_id_overlap(self, sample_demographic_data):
        """
        Verify no IDs appear in both train and holdout sets.

        Data leakage between sets would invalidate evaluation,
        so the same ID must not appear in both sets.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id'
        )

        train_ids = set(train_df['id'])
        holdout_ids = set(holdout_df['id'])

        # No overlap between train and holdout IDs
        assert len(train_ids & holdout_ids) == 0

    def test_all_data_accounted_for(self, sample_demographic_data):
        """
        Verify all original data appears in either train or holdout.

        No data should be lost during the split.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id'
        )

        total = len(train_df) + len(holdout_df)
        assert total == len(sample_demographic_data)


# =============================================================================
# DEMOGRAPHIC RATIO TESTS
# =============================================================================

class TestDemographicRatios:
    """
    Tests for demographic ratio enforcement.

    The holdout set must maintain specific majority/minority ratios
    to enable fair bias comparisons across differently-trained models.
    """

    def test_default_ratio_is_58_42(self, sample_demographic_data):
        """
        Verify default holdout ratio is 58% majority, 42% minority.

        This ratio represents a diverse evaluation set that tests
        model performance across demographic groups.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            testCount=100,
            percentMajority=58
        )

        majority_count = (holdout_df['RaceEth'] == 1).sum()
        minority_count = (holdout_df['RaceEth'] != 1).sum()

        assert majority_count == 58
        assert minority_count == 42

    def test_custom_ratio_respected(self, sample_demographic_data):
        """
        Verify custom majority percentage is respected.

        Users should be able to specify different demographic ratios
        for the holdout set.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            testCount=100,
            percentMajority=70  # 70% majority, 30% minority
        )

        majority_count = (holdout_df['RaceEth'] == 1).sum()
        minority_count = (holdout_df['RaceEth'] != 1).sum()

        assert majority_count == 70
        assert minority_count == 30

    def test_majority_value_configurable(self, sample_demographic_data):
        """
        Verify the majority group value can be configured.

        Different datasets may use different values to indicate
        the majority demographic group.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            testCount=100,
            columnToSplit='RaceEth',
            majorityValue=1,  # Group 1 is majority
            percentMajority=58
        )

        # Majority (value=1) should be 58%
        majority_count = (holdout_df['RaceEth'] == 1).sum()
        assert majority_count == 58


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

class TestDataIntegrity:
    """
    Tests for data integrity during splitting.

    These tests ensure columns and data values are preserved
    correctly during the holdout split process.
    """

    def test_columns_preserved(self, sample_demographic_data):
        """
        Verify all columns are preserved in both output DataFrames.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id'
        )

        original_cols = set(sample_demographic_data.columns)
        train_cols = set(train_df.columns)
        holdout_cols = set(holdout_df.columns)

        assert train_cols == original_cols
        assert holdout_cols == original_cols

    def test_seed_produces_reproducible_split(self, sample_demographic_data):
        """
        Verify same seed produces identical holdout sets.

        Reproducibility is critical for scientific experiments.
        """
        _, holdout1 = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            seed=42
        )
        _, holdout2 = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            seed=42
        )

        # Same seed should produce same holdout IDs
        assert list(holdout1['id'].sort_values()) == list(holdout2['id'].sort_values())

    def test_different_seeds_produce_different_splits(self, sample_demographic_data):
        """
        Verify different seeds produce different holdout sets.
        """
        _, holdout1 = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            seed=42
        )
        _, holdout2 = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            seed=99
        )

        # Different seeds should (very likely) produce different holdouts
        assert list(holdout1['id'].sort_values()) != list(holdout2['id'].sort_values())


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.

    These tests verify the function handles unusual inputs gracefully,
    such as small datasets or requested counts that exceed available samples.
    """

    def test_train_set_larger_than_holdout(self, sample_demographic_data):
        """
        Verify the train set is always larger than the holdout set.

        For any reasonable testCount, the remaining training data
        should be larger than the holdout set.
        """
        train_df, holdout_df = holdOutTestData(
            sample_demographic_data,
            id_column='id',
            testCount=100
        )

        assert len(train_df) > len(holdout_df)

    def test_small_dataset_caps_holdout_to_available(self):
        """
        Verify sampling is capped when requested count exceeds available data.

        If there are fewer majority samples than requested, the function
        should use min() to avoid a ValueError rather than crashing.
        """
        np.random.seed(0)
        small_df = pd.DataFrame({
            'id': range(1, 31),
            'RaceEth': [1] * 10 + [2] * 20,  # only 10 majority samples
            'age': np.random.randint(18, 65, 30),
            'outcome': np.random.choice([0, 1], 30),
        })

        # Request 20 majority but only 10 exist — should not raise
        train_df, holdout_df = holdOutTestData(
            small_df,
            id_column='id',
            testCount=30,
            percentMajority=58  # wants 17 majority, only 10 available
        )

        majority_in_holdout = (holdout_df['RaceEth'] == 1).sum()
        assert majority_in_holdout <= 10  # capped at available


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
