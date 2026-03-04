"""
Unit tests for model training components.

This module tests the core functionality of the OutcomeModel class
which handles train/test splitting, feature extraction, and
demographic tracking for the bias detection pipeline.

Test Categories:
    - Train/Test Split: Verifies correct data partitioning
    - Feature Handling: Ensures proper feature extraction
    - Demographic Counting: Tests demographic tracking for bias analysis
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model import OutcomeModel, LogisticModel


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_model_data():
    """
    Create sample data for model testing.

    Generates a DataFrame with:
        - id: Unique identifier for each row
        - age: Random ages between 18-65
        - income: Random income values
        - score: Random float scores
        - RaceEth: Demographic groups (1=majority, 2,3=minority)
        - outcome: Binary outcome variable (0 or 1)

    Returns:
        pd.DataFrame: Sample dataset with 200 rows
    """
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'id': range(1, n + 1),
        'age': np.random.randint(18, 65, n),
        'income': np.random.randint(20000, 100000, n),
        'score': np.random.rand(n),
        'RaceEth': np.random.choice([1, 2, 3], n, p=[0.6, 0.25, 0.15]),
        'outcome': np.random.choice([0, 1], n),
    })


# =============================================================================
# TRAIN/TEST SPLIT TESTS
# =============================================================================

class TestTrainTestSplit:
    """
    Tests for train/test splitting functionality.

    The OutcomeModel uses sklearn's train_test_split to partition data.
    These tests verify the split is done correctly and reproducibly.
    """

    def test_split_creates_train_and_test(self, sample_model_data):
        """
        Verify that initializing OutcomeModel creates both train and test sets.

        The model should automatically split data on initialization,
        populating X_train and X_test attributes.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        # Both sets should exist and have data
        assert model.X_train is not None
        assert model.X_test is not None
        assert len(model.X_train) > 0
        assert len(model.X_test) > 0

    def test_split_ratio_is_75_25(self, sample_model_data):
        """
        Verify the default train/test split ratio is 75%/25%.

        The OutcomeModel uses test_size=0.25, meaning 75% of data
        goes to training and 25% to testing.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        total = len(sample_model_data)
        train_ratio = len(model.X_train) / total

        # Allow small variance due to rounding
        assert 0.74 <= train_ratio <= 0.76

    def test_split_no_data_leakage(self, sample_model_data):
        """
        Verify there is no overlap between train and test sets.

        Data leakage would invalidate model evaluation, so we must
        ensure no row appears in both train and test sets.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        train_idx = set(model.X_train.index)
        test_idx = set(model.X_test.index)

        # Intersection should be empty (no shared indices)
        assert len(train_idx & test_idx) == 0

    def test_split_preserves_all_data(self, sample_model_data):
        """
        Verify all original data is accounted for after split.

        No data should be lost - the sum of train and test set sizes
        should equal the original dataset size.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        total_split = len(model.X_train) + len(model.X_test)
        assert total_split == len(sample_model_data)

    def test_seed_produces_reproducible_split(self, sample_model_data):
        """
        Verify that using the same seed produces identical splits.

        Reproducibility is critical for scientific experiments.
        The same seed should always produce the same train/test partition.
        """
        model1 = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )
        model2 = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        # Same seed = same indices in training set
        assert list(model1.X_train.index) == list(model2.X_train.index)


# =============================================================================
# FEATURE HANDLING TESTS
# =============================================================================

class TestFeatureHandling:
    """
    Tests for feature extraction and handling.

    The OutcomeModel should correctly separate features (X) from
    the target variable (y) and exclude ID columns from features.
    """

    def test_id_column_excluded_from_features(self, sample_model_data):
        """
        Verify the ID column is not included in the feature set.

        ID columns are identifiers, not predictive features.
        Including them would cause data leakage and invalid models.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        assert 'id' not in model.X.columns

    def test_target_excluded_from_features(self, sample_model_data):
        """
        Verify the target column is not included in the feature set.

        The target (outcome) variable must be separate from features
        to avoid data leakage during training.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        assert 'outcome' not in model.X.columns

    def test_features_preserved(self, sample_model_data):
        """
        Verify that non-ID, non-target columns are preserved as features.

        All columns except ID and target should be available
        for the model to use as predictive features.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        expected_features = ['age', 'income', 'score', 'RaceEth']
        for feat in expected_features:
            assert feat in model.X.columns


# =============================================================================
# DEMOGRAPHIC COUNTING TESTS
# =============================================================================

class TestDemographicCounting:
    """
    Tests for demographic counting functionality.

    The _countDemographic method tracks the demographic composition
    of training data, which is essential for bias analysis.
    """

    def test_count_demographic_returns_string(self, sample_model_data):
        """
        Verify demographic count returns a formatted string.

        The output is used in logging and CSV exports,
        so it must be a string format like "120 1, 50 2, 30 3".
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        result = model._countDemographic(model.X_train)
        assert isinstance(result, str)

    def test_count_demographic_contains_counts(self, sample_model_data):
        """
        Verify demographic string contains actual count numbers.

        The string should contain numeric counts for each
        demographic group present in the data.
        """
        model = OutcomeModel(
            data=sample_model_data,
            id_column='id',
            target_column=['outcome'],
            seed=42
        )

        result = model._countDemographic(model.X_train)

        # String should contain digits (the counts)
        assert any(char.isdigit() for char in result)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v"])
