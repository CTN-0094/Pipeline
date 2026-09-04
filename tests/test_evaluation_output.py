"""
Tests for the evaluation CSV writer in run_pipelineV2.

These guard the mapping between the in-memory evaluation dicts produced by the
model classes and the columns written to disk. The metrics themselves are
covered elsewhere; what matters here is that each value lands under the header
that names it, since a transposed count is silently plausible in the output.
"""

import csv
import os
import sys

import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_pipelineV2 import save_evaluations_to_csv
from src.constants import EndpointType


# An imbalanced, deliberately asymmetric set so every cell of the matrix differs;
# a transposition cannot coincidentally still match.
Y_TRUE = [0] * 12 + [1] * 8
Y_PRED = [0] * 9 + [1] * 3 + [0] * 5 + [1] * 3


def write_logical_row(tmp_path, y_true, y_pred, name):
    """Run one LOGICAL evaluation through the writer and read the row back."""
    evaluations = {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc": roc_auc_score(y_true, y_pred),
        "demographics": "10 NHW, 10 NHB",
        "training_demographics": "8 NHW, 7 NHB",
    }
    selected_outcome = {"name": "test_outcome", "endpointType": EndpointType.LOGICAL}
    save_evaluations_to_csv([evaluations], 0, selected_outcome, str(tmp_path), name)

    directory = tmp_path / name
    written = list(directory.iterdir())
    assert len(written) == 1, f"expected exactly one CSV, got {written}"
    with open(written[0], newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]


@pytest.fixture
def written_row(tmp_path):
    """The row written for a model that predicts both classes."""
    return write_logical_row(tmp_path, Y_TRUE, Y_PRED, "evals")


@pytest.fixture
def no_positive_prediction_row(tmp_path):
    """The row written for a model that predicts the negative class throughout.

    Imbalanced outcomes can leave a model predicting no positives at all, which
    is the case that drove F1 to NaN.
    """
    return write_logical_row(tmp_path, Y_TRUE, [0] * len(Y_TRUE), "degenerate_evals")


def test_confusion_matrix_counts_land_under_their_own_headers(written_row):
    """TP/TN/FP/FN must hold what their headers say.

    The writer previously read the matrix corners positionally, which swapped TP
    with TN, then emitted the values in an order that did not match the header
    row. Three of the four counts were written under the wrong column, so every
    logical-endpoint result on disk misreported them.
    """
    tn, fp, fn, tp = confusion_matrix(Y_TRUE, Y_PRED).ravel()
    assert int(written_row["TP"]) == tp
    assert int(written_row["TN"]) == tn
    assert int(written_row["FP"]) == fp
    assert int(written_row["FN"]) == fn


def test_written_counts_reproduce_the_written_rates(written_row):
    """The counts and the rates on the same row must describe one result.

    Precision and recall come from sklearn rather than from the counts, so
    recomputing them from the written columns is an independent check that the
    columns are not transposed.
    """
    tp, fp, fn = (int(written_row[k]) for k in ("TP", "FP", "FN"))
    assert tp / (tp + fp) == pytest.approx(float(written_row["Precision"]))
    assert tp / (tp + fn) == pytest.approx(float(written_row["Recall"]))


def test_accuracy_matches_the_written_counts(written_row):
    tp, tn, fp, fn = (int(written_row[k]) for k in ("TP", "TN", "FP", "FN"))
    expected = (tp + tn) / (tp + tn + fp + fn)
    assert float(written_row["Accuracy"]) == pytest.approx(expected)
    assert np.isclose(expected, np.mean(np.array(Y_TRUE) == np.array(Y_PRED)))


def test_f1_matches_sklearn(written_row):
    """F1 is derived from precision and recall, so it must agree with sklearn."""
    assert float(written_row["F1"]) == pytest.approx(f1_score(Y_TRUE, Y_PRED))


def test_f1_is_zero_when_the_model_predicts_no_positives(no_positive_prediction_row):
    """A model that never predicts a positive scores 0, not NaN.

    Precision and recall are both 0 there, so the 2pr/(p+r) the writer uses is
    0/0. With numpy floats that yields NaN rather than raising, which reached
    the CSV as an unreadable F1 column. sklearn reports 0 for this case.
    """
    assert float(no_positive_prediction_row["Precision"]) == 0
    assert float(no_positive_prediction_row["Recall"]) == 0

    written_f1 = float(no_positive_prediction_row["F1"])
    assert not np.isnan(written_f1)
    assert written_f1 == pytest.approx(
        f1_score(Y_TRUE, [0] * len(Y_TRUE), zero_division=0)
    )
