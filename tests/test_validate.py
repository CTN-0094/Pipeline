import os
import sys

import pytest
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.validate import validate_dataset_for_model



# ----------------------------
# ✅ VALID CASES
# ----------------------------

def test_valid_logical():
    df = pd.DataFrame({'outcome': [0, 1, 1, 0]})
    validate_dataset_for_model(df, "logical", "outcome")  

def test_valid_integer():
    df = pd.DataFrame({'outcome': [0, 2, 4]})
    validate_dataset_for_model(df, "integer", "outcome") 

def test_valid_survival():
    df = pd.DataFrame({'time': [5.0, 10.0, 3.5], 'event': [0, 1, 1]})
    validate_dataset_for_model(df, "survival", "event", "time")  


# ----------------------------
# ❌ INVALID CASES
# ----------------------------

def test_missing_outcome_column():
    df = pd.DataFrame({'not_outcome': [0, 1, 1]})
    with pytest.raises(ValueError, match="Outcome column 'outcome' not found in dataset"):
        validate_dataset_for_model(df, "logical", "outcome")

def test_invalid_logical_values():
    df = pd.DataFrame({'outcome': [0, 2, 1]})
    with pytest.raises(ValueError, match="must contain only 0 and 1 values"):
        validate_dataset_for_model(df, "logical", "outcome")

def test_invalid_integer_dtype():
    df = pd.DataFrame({'outcome': [1.0, 2.5, 3.1]})
    with pytest.raises(ValueError, match="must contain only integers"):
        validate_dataset_for_model(df, "integer", "outcome")

def test_survival_missing_time_column():
    df = pd.DataFrame({'event': [0, 1, 1]})
    with pytest.raises(ValueError, match="Time column 'time' not found in dataset"):
        validate_dataset_for_model(df, "survival", "event", "time")

def test_survival_non_numeric_time():
    df = pd.DataFrame({'time': ['five', 'ten', 'three'], 'event': [0, 1, 1]})
    with pytest.raises(ValueError, match="Time column 'time' must be numeric"):
        validate_dataset_for_model(df, "survival", "event", "time")

def test_survival_event_not_binary():
    df = pd.DataFrame({'time': [1.0, 2.0, 3.0], 'event': [0, 2, 1]})
    with pytest.raises(ValueError, match="Event indicator column 'event' must contain only 0 or 1"):
        validate_dataset_for_model(df, "survival", "event", "time")

def test_unsupported_model_type():
    df = pd.DataFrame({'outcome': [0, 1, 1]})
    with pytest.raises(ValueError, match="Unsupported model type 'unsupported'"):
        validate_dataset_for_model(df, "unsupported", "outcome")
