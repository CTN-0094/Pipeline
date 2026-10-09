"""
Tests for the run_pipelineV2 CLI helpers.

These cover argument handling and the predictions writer, where an off-by-one
or a mistyped value produces output that looks plausible but is wrong.
"""

import csv
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_pipelineV2 import argument_handler, resolve_seed_list, save_predictions_to_csv
from src.constants import EndpointType


class TestResolveSeedList:
    """-l/--loop expands to an inclusive range of seeds."""

    def test_no_loop_runs_seed_zero(self):
        assert resolve_seed_list(None) == [0]

    def test_range_includes_both_ends(self):
        assert resolve_seed_list([5, 10]) == [5, 6, 7, 8, 9, 10]

    def test_single_value_runs_that_seed(self):
        assert resolve_seed_list([5]) == [5]

    def test_order_of_bounds_does_not_matter(self):
        assert resolve_seed_list([10, 5]) == resolve_seed_list([5, 10])


class TestArgumentHandler:
    """Parsed arguments have the types the pipeline compares them against."""

    def test_majority_is_parsed_as_int(self, monkeypatch):
        """RaceEth is numeric, so a string "1" would match no majority rows."""
        monkeypatch.setattr(sys, 'argv', ['run_pipelineV2.py', '--data', 'x.csv', '--majority', '1'])
        assert argument_handler().majority == 1

    def test_outcome_defaults_to_none(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['run_pipelineV2.py', '--data', 'x.csv'])
        assert argument_handler().outcome is None


def test_predictions_header_matches_subset_count(tmp_path):
    """The header names one column per subset, however many subsets there are."""
    num_subsets = 11
    data = pd.Series([zip(['a', 'b'], [i, i + 0.5]) for i in range(num_subsets)])
    selected_outcome = {"name": "test_outcome", "endpointType": EndpointType.LOGICAL}
    save_predictions_to_csv(data, 0, selected_outcome, str(tmp_path), 'preds')

    written = list((tmp_path / 'preds').iterdir())
    with open(written[0], newline='') as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ['who'] + [f'Subset_{i + 1}' for i in range(num_subsets)]
    assert all(len(row) == len(rows[0]) for row in rows)
