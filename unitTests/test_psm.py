import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.create_demodf_knn import holdOutTestData, propensityScoreMatch, create_subsets

@pytest.fixture
def mock_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(200),
        "RaceEth": np.random.choice([0, 1], size=200, p=[0.4, 0.6]),
        "age": np.random.randint(18, 65, size=200),
        "is_female": np.random.choice([0, 1], size=200),
        "some_feature": np.random.rand(200)
    })
    return df

@pytest.fixture
def small_psm_test_data():
    # 6 majority (RaceEth=1), 3 minority (RaceEth=0), with varying age and gender
    return pd.DataFrame({
        "id": range(9),
        "RaceEth": [1, 1, 1, 1, 1, 1, 0, 0, 0],
        "age":     [30, 32, 31, 35, 34, 36, 29, 33, 30],
        "is_female": [1, 1, 1, 1, 1, 0, 1, 0, 1],
    })

def test_holdOutTestData(mock_data):
    train_df, test_df = holdOutTestData(mock_data, "id")

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(test_df) <= 100  # Limited to 100 records
    assert not any(test_df["id"].isin(train_df["id"]))  # No ID overlap
    assert test_df["RaceEth"].value_counts().get(1, 0) <= 58
    assert test_df["RaceEth"].value_counts().get(0, 0) <= 42

def test_propensityScoreMatch(monkeypatch, mock_data):
    def mock_rmatchit(df, id_column, columnsToMatch, sampleSize):
        return pd.DataFrame({
            "treated_row": [1, 2, 3],
            "control_row_0": [101, 102, 103],
            "control_row_1": [201, 202, 203],
        })

    monkeypatch.setattr("src.create_demodf_knn.PropensityScoreMatchRMatchit", mock_rmatchit)

    matched_dfs = propensityScoreMatch(
        mock_data,
        idColumn="id",
        columnToSplit='RaceEth',
        majorityValue=1,
        columnsToMatch=['age', 'is_female'],
        sampleSize=3
    )

    assert isinstance(matched_dfs, list)
    assert all(isinstance(df, pd.DataFrame) for df in matched_dfs)
    assert all("id" in df.columns for df in matched_dfs)


def test_propensityScoreMatchLogic(small_psm_test_data):
    matched = propensityScoreMatch(
        small_psm_test_data.copy(),
        idColumn="id",
        columnToSplit='RaceEth',
        majorityValue=1,
        columnsToMatch=['age', 'is_female'],
        sampleSize=3  # only 3 treated (minority) are available
    )

    # Should return a list of 3 matched DataFrames (one per minority participant)
    assert isinstance(matched, list)
    assert len(matched) == 3

    for df in matched:
        # Each match group should have 1 treated and 2 controls (3 total)
        assert df.shape[0] == 3
        assert "RaceEth" in df.columns


def test_create_subsets(mock_data):
    # Simulate 3 matched DataFrames
    df1 = mock_data.copy()
    df2 = mock_data.copy()
    df3 = mock_data.copy()

    subsets = create_subsets([df1, df2, df3], splits=5, sampleSize=100)
    assert isinstance(subsets, list)
    assert len(subsets) == 5
    for subset in subsets:
        assert isinstance(subset, pd.DataFrame)
        assert set(mock_data.columns).issubset(subset.columns)
