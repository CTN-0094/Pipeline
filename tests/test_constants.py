"""
Unit tests for constants.

Verifies that EndpointType enum members exist and map to the expected
string values used throughout the pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.constants import EndpointType


def test_logical_value():
    assert EndpointType.LOGICAL.value == 'logical'

def test_integer_value():
    assert EndpointType.INTEGER.value == 'integer'

def test_survival_value():
    assert EndpointType.SURVIVAL.value == 'survival'

def test_enum_lookup_by_string():
    """String values can be used to look up enum members."""
    assert EndpointType('logical') == EndpointType.LOGICAL
    assert EndpointType('integer') == EndpointType.INTEGER
    assert EndpointType('survival') == EndpointType.SURVIVAL

def test_invalid_string_raises():
    """An unrecognised string raises a ValueError."""
    import pytest
    with pytest.raises(ValueError):
        EndpointType('unknown')


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
