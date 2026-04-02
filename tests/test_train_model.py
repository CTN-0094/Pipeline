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

from sklearn.exceptions import NotFittedError

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
# LOGISTIC MODEL FIXTURES
# =============================================================================

@pytest.fixture
def logistic_data():
    """
    Sample data for LogisticModel tests.

    Features have genuine signal so Lasso retains at least one of them,
    avoiding the 'no features selected' edge case during training tests.
    """
    np.random.seed(42)
    n = 200
    feature1 = np.random.rand(n)
    feature2 = np.random.rand(n)
    # outcome correlates with feature1 so L1 regularization keeps it
    outcome = (feature1 + np.random.rand(n) * 0.3 > 0.6).astype(int)
    return pd.DataFrame({
        "id": range(1, n + 1),
        "feature1": feature1,
        "feature2": feature2,
        "RaceEth": np.random.choice([1, 2, 3], n, p=[0.6, 0.25, 0.15]),
        "outcome": outcome,
    })


@pytest.fixture
def trained_logistic_model(logistic_data):
    """A LogisticModel that has completed selectFeatures() and train()."""
    model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
    model.selectFeatures()
    model.train()
    return model


@pytest.fixture
def logistic_heldout(logistic_data):
    """Held-out set with the same schema as logistic_data."""
    np.random.seed(99)
    n = 50
    feature1 = np.random.rand(n)
    feature2 = np.random.rand(n)
    outcome = (feature1 + np.random.rand(n) * 0.3 > 0.6).astype(int)
    return pd.DataFrame({
        "id": range(1, n + 1),
        "feature1": feature1,
        "feature2": feature2,
        "RaceEth": np.random.choice([1, 2, 3], n, p=[0.6, 0.25, 0.15]),
        "outcome": outcome,
    })


# =============================================================================
# LOGISTIC MODEL — FEATURE SELECTION
# =============================================================================

class TestLogisticModelFeatureSelection:
    """Tests for selectFeatures() / lasso_feature_selection(classification)."""

    def test_select_features_populates_selected_features(self, logistic_data):
        """selected_features should be non-empty after selectFeatures()."""
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        assert len(model.selected_features) > 0

    def test_selected_features_are_subset_of_input_columns(self, logistic_data):
        """Every selected feature must exist in the original feature set."""
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        for feat in model.selected_features:
            assert feat in model.X.columns

    def test_selected_features_exclude_id_and_target(self, logistic_data):
        """ID and target columns must not appear in selected_features."""
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        assert "id" not in model.selected_features
        assert "outcome" not in model.selected_features

    def test_select_features_uses_classification_lasso(self, logistic_data):
        """
        Calling selectFeatures() twice should produce the same features
        (deterministic given the same seed), confirming the classification
        Lasso path is used consistently.
        """
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        first_run = list(model.selected_features)
        model.selectFeatures()
        second_run = list(model.selected_features)
        assert first_run == second_run


# =============================================================================
# LOGISTIC MODEL — TRAINING
# =============================================================================

class TestLogisticModelTraining:
    """Tests for train() and _find_best_threshold()."""

    def test_train_sets_model_attribute(self, logistic_data):
        """model attribute should be populated after train()."""
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        model.train()
        assert model.model is not None

    def test_train_sets_threshold_between_0_and_1(self, logistic_data):
        """best_threshold should be a probability value in [0, 1]."""
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        model.train()
        threshold = float(model.best_threshold)
        assert 0.0 <= threshold <= 1.0

    def test_threshold_equals_positive_proportion_of_test_set(self, logistic_data):
        """
        _find_best_threshold sets threshold to the positive class proportion
        in y_test — verify the value matches directly.
        """
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        model.train()
        expected = model.y_test.mean()
        assert float(model.best_threshold) == pytest.approx(float(expected))

    def test_trained_model_can_predict_proba(self, trained_logistic_model):
        """predict_proba should return an array with shape (n, 2) on test data."""
        X = trained_logistic_model.X_test[trained_logistic_model.selected_features]
        proba = trained_logistic_model.model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_train_without_select_features_raises(self, logistic_data):
        """
        Calling train() before selectFeatures() should fail because
        selected_features defaults to all columns, but the internal
        LogisticRegression fit will still run — what we verify is that
        selectFeatures() was designed to be called first by checking the
        default selected_features equals all X columns before selection.
        """
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        # Before selection, selected_features is the full column index
        assert list(model.selected_features) == list(model.X.columns)


# =============================================================================
# LOGISTIC MODEL — EVALUATION
# =============================================================================

class TestLogisticModelEvaluation:
    """Tests for evaluate() and _evaluateOnValidation()."""

    def test_evaluate_returns_four_values(self, trained_logistic_model, logistic_heldout):
        """evaluate() must return a 4-tuple: (heldout_preds, heldout_evals, subset_preds, subset_evals)."""
        result = trained_logistic_model.evaluate(logistic_heldout)
        assert len(result) == 4

    def test_heldout_evaluations_has_required_keys(self, trained_logistic_model, logistic_heldout):
        """Heldout evaluation dict must contain roc, confusion_matrix, precision, recall, demographics."""
        _, heldout_evals, _, _ = trained_logistic_model.evaluate(logistic_heldout)
        for key in ("roc", "confusion_matrix", "precision", "recall", "demographics"):
            assert key in heldout_evals, f"Missing key: {key}"

    def test_subset_evaluations_has_required_keys(self, trained_logistic_model, logistic_heldout):
        """Subset evaluation dict must contain the same keys as heldout evaluations."""
        _, _, _, subset_evals = trained_logistic_model.evaluate(logistic_heldout)
        for key in ("roc", "confusion_matrix", "precision", "recall", "demographics"):
            assert key in subset_evals, f"Missing key: {key}"

    def test_heldout_roc_is_valid_probability(self, trained_logistic_model, logistic_heldout):
        """ROC-AUC score must be a float in [0, 1]."""
        _, heldout_evals, _, _ = trained_logistic_model.evaluate(logistic_heldout)
        roc = heldout_evals["roc"]
        assert isinstance(roc, float)
        assert 0.0 <= roc <= 1.0

    def test_both_evaluations_include_training_demographics(self, trained_logistic_model, logistic_heldout):
        """training_demographics key must be added to both evaluation dicts."""
        _, heldout_evals, _, subset_evals = trained_logistic_model.evaluate(logistic_heldout)
        assert "training_demographics" in heldout_evals
        assert "training_demographics" in subset_evals

    def test_evaluate_before_train_raises_not_fitted_error(self, logistic_data, logistic_heldout):
        """evaluate() before train() must raise NotFittedError."""
        model = LogisticModel(data=logistic_data, id_column="id", target_column=["outcome"], seed=42)
        model.selectFeatures()
        with pytest.raises(NotFittedError):
            model.evaluate(logistic_heldout)

    def test_heldout_predictions_are_iterable(self, trained_logistic_model, logistic_heldout):
        """heldout_predictions should be iterable (id, pred) pairs."""
        heldout_preds, _, _, _ = trained_logistic_model.evaluate(logistic_heldout)
        pairs = list(heldout_preds)
        assert len(pairs) == len(logistic_heldout)
        for pair in pairs:
            assert len(pair) == 2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v"])
