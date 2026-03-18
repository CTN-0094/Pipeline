import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch
from src.model_training import train_and_evaluate_models
from src.constants import EndpointType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_data():
    np.random.seed(42)
    c1 = np.random.randint(0, 2, 100)
    c2 = np.random.randint(0, 2, 100)
    c3 = np.random.randint(0, 2, 100)
    return pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": c1,
        "feature2": c2,
        "feature3": c3,
        "feature4": np.random.rand(100),
        "label": c1 & c2 & c3,
    })


@pytest.fixture
def classification_heldout():
    np.random.seed(41)
    c1 = np.random.randint(0, 2, 100)
    c2 = np.random.randint(0, 2, 100)
    c3 = np.random.randint(0, 2, 100)
    return pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": c1,
        "feature2": c2,
        "feature3": c3,
        "feature4": np.random.rand(100),
        "label": c1 & c2 & c3,
    })


@pytest.fixture
def logical_outcome():
    return {"endpointType": EndpointType.LOGICAL, "columnsToUse": ["label"]}


@pytest.fixture
def integer_outcome():
    return {"endpointType": EndpointType.INTEGER, "columnsToUse": ["label"]}


@pytest.fixture
def survival_outcome():
    return {"endpointType": EndpointType.SURVIVAL, "columnsToUse": ["labelTTE", "label"]}


def _fake_evaluate():
    """Return a 4-tuple matching the MultiIndex columns: heldout_preds, heldout_evals, subset_preds, subset_evals."""
    return (iter([]), {}, iter([]), {})


# ── Model Selection ───────────────────────────────────────────────────────────

class TestModelSelection:
    """Verify the correct model class is instantiated for each endpoint type."""

    def test_logical_endpoint_uses_logistic_model(self, classification_data, classification_heldout, logical_outcome):
        with patch("src.model_training.LogisticModel") as MockModel:
            instance = MagicMock()
            instance.evaluate.return_value = _fake_evaluate()
            MockModel.return_value = instance
            train_and_evaluate_models([classification_data], "id", logical_outcome, classification_heldout)
            MockModel.assert_called_once()

    def test_integer_endpoint_uses_negative_binomial_model(self, classification_data, classification_heldout, integer_outcome):
        with patch("src.model_training.NegativeBinomialModel") as MockModel:
            instance = MagicMock()
            instance.evaluate.return_value = _fake_evaluate()
            MockModel.return_value = instance
            train_and_evaluate_models([classification_data], "id", integer_outcome, classification_heldout)
            MockModel.assert_called_once()

    def test_survival_endpoint_uses_cox_model(self, classification_data, classification_heldout, survival_outcome):
        np.random.seed(42)
        data = classification_data.copy()
        data["labelTTE"] = np.random.randint(1, 200, 100)
        heldout = classification_heldout.copy()
        heldout["labelTTE"] = np.random.randint(1, 200, 100)
        with patch("src.model_training.CoxProportionalHazard") as MockModel:
            instance = MagicMock()
            instance.evaluate.return_value = _fake_evaluate()
            MockModel.return_value = instance
            train_and_evaluate_models([data], "id", survival_outcome, heldout)
            MockModel.assert_called_once()

    def test_model_instantiated_with_correct_args(self, classification_data, classification_heldout, logical_outcome):
        with patch("src.model_training.LogisticModel") as MockModel:
            instance = MagicMock()
            instance.evaluate.return_value = _fake_evaluate()
            MockModel.return_value = instance
            train_and_evaluate_models([classification_data], "id", logical_outcome, classification_heldout)
            MockModel.assert_called_once_with(classification_data, "id", ["label"])


# ── Results Structure ─────────────────────────────────────────────────────────

class TestResultsStructure:
    """Verify the returned DataFrame has the correct MultiIndex shape."""

    def test_results_has_multiindex_columns(self, classification_data, classification_heldout, logical_outcome):
        results = train_and_evaluate_models(
            [classification_data], "id", logical_outcome, classification_heldout
        )
        assert isinstance(results.columns, pd.MultiIndex)

    def test_results_column_levels(self, classification_data, classification_heldout, logical_outcome):
        results = train_and_evaluate_models(
            [classification_data], "id", logical_outcome, classification_heldout
        )
        assert set(results.columns.get_level_values(0)) == {"heldout", "subset"}
        assert set(results.columns.get_level_values(1)) == {"predictions", "evaluations"}

    def test_results_has_four_columns(self, classification_data, classification_heldout, logical_outcome):
        results = train_and_evaluate_models(
            [classification_data], "id", logical_outcome, classification_heldout
        )
        assert len(results.columns) == 4

    def test_empty_subsets_returns_empty_dataframe(self, classification_heldout, logical_outcome):
        results = train_and_evaluate_models([], "id", logical_outcome, classification_heldout)
        assert results.shape[0] == 0
        assert isinstance(results.columns, pd.MultiIndex)


# ── Subset Processing ─────────────────────────────────────────────────────────

class TestSubsetProcessing:
    """Verify each subset is independently trained and evaluated."""

    def test_single_subset_produces_one_result_row(self, classification_data, classification_heldout, logical_outcome):
        results = train_and_evaluate_models(
            [classification_data], "id", logical_outcome, classification_heldout
        )
        assert results.shape[0] == 1

    def test_two_subsets_produce_two_result_rows(self, classification_data, classification_heldout, logical_outcome):
        subset1 = classification_data.iloc[:60].reset_index(drop=True)
        subset2 = classification_data.iloc[40:].reset_index(drop=True)
        results = train_and_evaluate_models(
            [subset1, subset2], "id", logical_outcome, classification_heldout
        )
        assert results.shape[0] == 2

    def test_model_pipeline_called_once_per_subset(self, classification_data, classification_heldout, logical_outcome):
        subset1 = classification_data.iloc[:60].reset_index(drop=True)
        subset2 = classification_data.iloc[40:].reset_index(drop=True)
        subset3 = classification_data.reset_index(drop=True)

        with patch("src.model_training.LogisticModel") as MockModel:
            instances = [MagicMock() for _ in range(3)]
            for inst in instances:
                inst.evaluate.return_value = _fake_evaluate()
            MockModel.side_effect = instances

            train_and_evaluate_models(
                [subset1, subset2, subset3], "id", logical_outcome, classification_heldout
            )

            assert MockModel.call_count == 3
            for inst in instances:
                inst.selectFeatures.assert_called_once()
                inst.train.assert_called_once()
                inst.evaluate.assert_called_once()

    def test_evaluate_called_with_heldout_data(self, classification_data, classification_heldout, logical_outcome):
        with patch("src.model_training.LogisticModel") as MockModel:
            instance = MagicMock()
            instance.evaluate.return_value = _fake_evaluate()
            MockModel.return_value = instance

            train_and_evaluate_models(
                [classification_data], "id", logical_outcome, classification_heldout
            )

            instance.evaluate.assert_called_once_with(classification_heldout)
