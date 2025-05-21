import pandas as pd
import pytest
from src.data_loading import load_datasets


def test_load_datasets_returns_dataframes(tmp_path):
    """
    Test that load_datasets correctly loads two CSV files and returns two DataFrames.
    """
    # Create a temporary CSV file simulating master_data.csv
    master_path = tmp_path / "master_data.csv"
    master_path.write_text("who,age,sex\n1,25,M\n2,30,F")

    # Create a temporary CSV file simulating all_binary_selected_outcomes.csv
    outcomes_path = tmp_path / "all_binary_selected_outcomes.csv"
    outcomes_path.write_text("who,ctn0094_relapse_event\n1,0\n2,1")

    # Call the function being tested
    df1, df2 = load_datasets(str(master_path), str(outcomes_path))

    # Verify both outputs are pandas DataFrames
    assert isinstance(df1, pd.DataFrame)  # Check master file is loaded as a DataFrame
    assert isinstance(df2, pd.DataFrame)  # Check outcome file is loaded as a DataFrame

    # Check dimensions of the returned DataFrames
    assert df1.shape == (2, 3)  # Expect 2 rows, 3 columns in master data
    assert df2.shape == (2, 2)  # Expect 2 rows, 2 columns in outcome data

    # Ensure column names are loaded correctly
    assert list(df1.columns) == ["who", "age", "sex"]  # Master data columns
    assert list(df2.columns) == ["who", "ctn0094_relapse_event"]  # Outcome data columns


def test_load_datasets_file_not_found():
    """
    Test that load_datasets raises a FileNotFoundError when files do not exist.
    """
    # This should raise an error because the file paths don't exist
    with pytest.raises(FileNotFoundError):
        load_datasets("fake_master.csv", "fake_outcomes.csv")


def test_load_datasets_missing_who_column(tmp_path):
    """
    Test behavior when 'who' column is missing from both files.
    """
    # Create master CSV without the 'who' column
    master_path = tmp_path / "master.csv"
    master_path.write_text("id,age\n1,25\n2,30")

    # Create outcome CSV also without the 'who' column
    outcomes_path = tmp_path / "outcomes.csv"
    outcomes_path.write_text("id,ctn0094_relapse_event\n1,0\n2,1")

    # Load the datasets
    df1, df2 = load_datasets(master_path, outcomes_path)

    # Check that the 'who' column is not present in either file
    assert "who" not in df1.columns or "who" not in df2.columns
