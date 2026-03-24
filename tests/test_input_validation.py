import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model import LogisticModel, NegativeBinomialModel, CoxProportionalHazard, BetaRegression


@pytest.fixture
def valid_data():
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(100),
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "RaceEth": np.random.choice([0, 1], 100),
        "label": np.random.choice([0, 1], 100),
    })


# ── TestInputValidation ───────────────────────────────────────────────────────

class TestInputValidation:

    def test_empty_dataframe_raises_error(self):
        with pytest.raises(ValueError, match="cannot be None or empty"):
            LogisticModel(data=pd.DataFrame(), id_column="id", target_column=["label"])

    def test_none_dataframe_raises_error(self):
        with pytest.raises((ValueError, TypeError)):
            LogisticModel(data=None, id_column="id", target_column=["label"])

    def test_missing_id_column_raises_error(self, valid_data):
        with pytest.raises(ValueError, match="ID column 'missing_id' not found"):
            LogisticModel(data=valid_data, id_column="missing_id", target_column=["label"])

    def test_missing_target_column_raises_error(self, valid_data):
        with pytest.raises(ValueError, match="Target column 'missing_label' not found"):
            LogisticModel(data=valid_data, id_column="id", target_column=["missing_label"])

    def test_target_column_not_list_raises_error(self, valid_data):
        with pytest.raises(TypeError, match="target_column must be a list"):
            LogisticModel(data=valid_data, id_column="id", target_column="label")

    def test_duplicate_ids_raises_error(self):
        df = pd.DataFrame({
            "id": [1, 1, 2, 3],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "RaceEth": [0, 1, 0, 1],
            "label": [0, 1, 0, 1],
        })
        with pytest.raises(ValueError, match="duplicate IDs"):
            LogisticModel(data=df, id_column="id", target_column=["label"])

    def test_valid_data_does_not_raise(self, valid_data):
        model = LogisticModel(data=valid_data, id_column="id", target_column=["label"])
        assert model is not None

    def test_validation_applies_to_all_model_types(self):
        """All subclasses inherit OutcomeModel validation."""
        df = pd.DataFrame()
        for ModelClass in [LogisticModel, NegativeBinomialModel, CoxProportionalHazard, BetaRegression]:
            with pytest.raises(ValueError, match="cannot be None or empty"):
                ModelClass(data=df, id_column="id", target_column=["label"])
