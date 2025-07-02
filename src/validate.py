import pandas as pd
from src.constants import EndpointType

def validate_dataset_for_model(df, model_type, outcome_col, time_col=None):
    """
    Validate the dataset based on the model type.

    Parameters:
    - df: The pandas DataFrame to validate.
    - model_type: The EndpointType enum or string ('logical', 'integer', 'survival').
    - outcome_col: The name of the outcome column.
    - time_col: The name of the time column (for survival analysis).

    Raises:
    - ValueError if the dataset does not meet expected format for the model type.
    """

    # Normalize model_type to Enum if passed as string
    if isinstance(model_type, str):
        try:
            model_type = EndpointType(model_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported model type '{model_type}'")

    # Check that outcome column exists
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in dataset")

    if model_type == EndpointType.LOGICAL:
        unique_vals = set(df[outcome_col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Binary outcome column '{outcome_col}' must contain only 0 and 1 values. Found: {unique_vals}")

    elif model_type == EndpointType.INTEGER:
        if not pd.api.types.is_integer_dtype(df[outcome_col]):
            raise ValueError(f"Outcome column '{outcome_col}' must contain only integers.")

    elif model_type == EndpointType.SURVIVAL:
        if not time_col:
            raise ValueError("Survival analysis requires both time and event columns.")
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            raise ValueError(f"Time column '{time_col}' must be numeric.")
        if not df[outcome_col].dropna().isin([0, 1]).all():
            raise ValueError(f"Event indicator column '{outcome_col}' must contain only 0 or 1.")

    else:
        raise ValueError(f"Unsupported model type '{model_type}'")
