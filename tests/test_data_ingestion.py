"""
Unit tests for data ingestion in the pipeline.

Focuses on:
- CSV file loading
- Basic data structure validation
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Create a minimal valid dataset for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'id': range(1, n + 1),
        'age': np.random.randint(18, 65, n),
        'group': np.random.choice([1, 2, 3], n),
        'outcome': np.random.choice([0, 1], n),
    })


@pytest.fixture
def sample_csv_file(sample_data):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


# =============================================================================
# CSV LOADING TESTS
# =============================================================================

class TestCSVLoading:
    """Tests for CSV file loading."""

    def test_load_csv_returns_dataframe(self, sample_csv_file):
        """CSV loads as DataFrame."""
        df = pd.read_csv(sample_csv_file)
        assert isinstance(df, pd.DataFrame)

    def test_load_csv_correct_row_count(self, sample_csv_file):
        """All rows are loaded."""
        df = pd.read_csv(sample_csv_file)
        assert len(df) == 100

    def test_load_csv_preserves_columns(self, sample_csv_file):
        """Columns are preserved after loading."""
        df = pd.read_csv(sample_csv_file)
        expected_cols = ['id', 'age', 'group', 'outcome']
        assert list(df.columns) == expected_cols

    def test_load_nonexistent_file_raises_error(self):
        """Loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            pd.read_csv('this_file_does_not_exist.csv')

    def test_load_csv_numeric_types_preserved(self, sample_csv_file):
        """Numeric columns stay numeric."""
        df = pd.read_csv(sample_csv_file)
        assert pd.api.types.is_numeric_dtype(df['age'])
        assert pd.api.types.is_numeric_dtype(df['outcome'])


# =============================================================================
# DATA STRUCTURE TESTS
# =============================================================================

class TestDataStructure:
    """Tests for basic data structure validation."""

    def test_id_column_is_unique(self, sample_data):
        """ID column has no duplicates."""
        assert sample_data['id'].is_unique

    def test_no_empty_dataframe(self, sample_data):
        """DataFrame is not empty."""
        assert len(sample_data) > 0

    def test_binary_outcome_values(self, sample_data):
        """Binary outcome contains only 0 and 1."""
        unique_vals = set(sample_data['outcome'].unique())
        assert unique_vals.issubset({0, 1})

    def test_age_within_bounds(self, sample_data):
        """Age values are reasonable."""
        assert sample_data['age'].min() >= 0
        assert sample_data['age'].max() <= 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
