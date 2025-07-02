# tests/test_validate.py

import sys
import os
import pandas as pd
import pytest

# Ensure src/ is in the path for imports
sys.path.append(os.path.abspath("src"))

from validate import validate_dataset_for_model

# ----------------------------
# ✅ VALID CASES
# ----------------------------

def test_valid_binary():
    df = pd.DataFrame({'outcome': [0, 1, 1, 0]})
    assert validate_dataset_for_model(df, "binary", "outcome") is True

def test_valid_negative_binomial():
    df = pd.DataFrame({'outcome': [0, 2, 4]})
    assert validate_dataset_for_model(df, "negative_binomial", "outcome") is True

def test_valid_beta():
    df = pd.DataFrame({'outcome': [0.01, 0.5, 0.9999]})
    assert validate_dataset_for_model(df, "beta", "outcome") is True

def test_valid_survival():
    df = pd.DataFrame({'time': [5.0, 10.0, 3.5], 'event': [0, 1, 1]})
    assert validate_dataset_for_model(df, "survival", "event", "time") is True

# ----------------------------
# ❌ INVALID CASES
# ----------------------------

def test_invalid_binary():
    df = pd.DataFrame({'outcome': [0, 2, 1]})
    with pytest.raises(ValueError, match="Binary model requires outcome to contain only 0 or 1"):
        validate_dataset_for_model(df, "binary", "outcome")

def test_invalid_negative_binomial():
    df = pd.DataFrame({'outcome': [1.5, 2.1]})
    with pytest.raises(ValueError, match="Negative Binomial model requires an integer outcome column."):
        validate_dataset_for_model(df, "negative_binomial", "outcome")

def test_invalid_beta():
    df = pd.DataFrame({'outcome': [0.0, 1.0, 0.5]})
    with pytest.raises(ValueError, match="Beta model requires values strictly between 0 and 1"):
        validate_dataset_for_model(df, "beta", "outcome")

def test_missing_time_column_for_survival():
    df = pd.DataFrame({'event': [1, 0, 1]})
    with pytest.raises(ValueError, match="Survival model requires both an outcome and a time column."):
        validate_dataset_for_model(df, "survival", "event", None)

def test_invalid_survival_event_values():
    df = pd.DataFrame({'time': [5.0, 10.0, 3.5], 'event': [0, 2, 1]})
    with pytest.raises(ValueError, match="Survival event column must contain only 0"):
        validate_dataset_for_model(df, "survival", "event", "time")

def test_invalid_survival_time_column_type():
    df = pd.DataFrame({'time': ['a', 'b', 'c'], 'event': [1, 0, 1]})
    with pytest.raises(ValueError, match="Survival model requires a numeric time-to-event column."):
        validate_dataset_for_model(df, "survival", "event", "time")
