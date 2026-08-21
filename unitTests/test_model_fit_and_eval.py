"""
Integration-style unit tests for all four model classes: select → train → evaluate.

Unlike the mocked tests in tests/test_model_training.py, these tests run the full
statistical pipeline end-to-end on synthetic data and verify concrete outputs
(feature lists, metric values, confusion matrices, demographic strings).

Models under test:
    LogisticModel           — binary outcome (LOGICAL endpoint)
    NegativeBinomialModel   — count outcome (INTEGER endpoint)
    CoxProportionalHazard   — time-to-event outcome (SURVIVAL endpoint)
    BetaRegression          — continuous [0, 1] outcome

Fixtures with 'noisy' in the name include an uncorrelated feature1 to exercise
LASSO feature selection; non-noisy fixtures have clean signal to test exact
feature lists. Heldout fixtures use a different seed (41) from training (42)
to simulate an independent evaluation set.
"""

import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
from src.train_model import LogisticModel, NegativeBinomialModel, CoxProportionalHazard, BetaRegression
from src.model_training import train_and_evaluate_models
from src.constants import EndpointType

@pytest.fixture
def sample_classification_data_one_feature():
    np.random.seed(42)
    randomValues = np.random.randint(0, 2, 100)
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": randomValues + np.random.rand(100),
        "feature2": np.random.rand(100),
        "label": randomValues
    })
    return df



@pytest.fixture
def sample_classification_data_multiple_features():
    np.random.seed(42)
    randomValuesComponent1 = np.random.randint(0, 2, 100)
    randomValuesComponent2 = np.random.randint(0, 2, 100)
    randomValuesComponent3 = np.random.randint(0, 2, 100)
    randomValues = randomValuesComponent1 & randomValuesComponent2 & randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": randomValuesComponent1,
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "label": randomValues
    })
    return df



@pytest.fixture
def sample_classification_data_multiple_features_noisy():
    np.random.seed(42)
    randomValuesComponent1 = np.random.randint(0, 2, 100)
    randomValuesComponent2 = np.random.randint(0, 2, 100)
    randomValuesComponent3 = np.random.randint(0, 2, 100)
    randomValues = randomValuesComponent1 & randomValuesComponent2 & randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.randint(0, 2, 100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "label": randomValues
    })
    return df



@pytest.fixture
def sample_classification_data_heldout_multiple_features_noisy():
    np.random.seed(41)
    randomValuesComponent1 = np.random.randint(0, 2, 100)
    randomValuesComponent2 = np.random.randint(0, 2, 100)
    randomValuesComponent3 = np.random.randint(0, 2, 100)
    randomValues = randomValuesComponent1 & randomValuesComponent2 & randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.randint(0, 2, 100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "label": randomValues
    })
    return df



@pytest.fixture
def sample_integer_data_multiple_features_noisy():
    np.random.seed(42)
    randomValuesComponent1 = np.random.randint(0, 200, 100)
    randomValuesComponent2 = np.random.randint(0, 200, 100)
    randomValuesComponent3 = np.random.randint(0, 200, 100)
    randomValues = randomValuesComponent1 + randomValuesComponent2 + randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.randint(0, 200, 100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "label": randomValues
    })
    return df



@pytest.fixture
def sample_integer_data_heldout_multiple_features_noisy():
    np.random.seed(41)
    randomValuesComponent1 = np.random.randint(0, 200, 100)
    randomValuesComponent2 = np.random.randint(0, 200, 100)
    randomValuesComponent3 = np.random.randint(0, 200, 100)
    randomValues = randomValuesComponent1 + randomValuesComponent2 + randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.randint(0, 200, 100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "label": randomValues
    })
    return df



@pytest.fixture
def sample_survival_data_multiple_features_noisy():
    np.random.seed(42)
    randomValuesComponent1 = np.random.randint(0, 200, 100)
    randomValuesComponent2 = np.random.randint(0, 200, 100)
    randomValuesComponent3 = np.random.randint(0, 200, 100)
    randomValues = randomValuesComponent1 + randomValuesComponent2 + randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.randint(0, 200, 100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "labelTTE": randomValues,
        "label": np.random.randint(0, 2, 100)
    })
    return df



@pytest.fixture
def sample_survival_data_heldout_multiple_features_noisy():
    np.random.seed(41)
    randomValuesComponent1 = np.random.randint(0, 1, 100)
    randomValuesComponent2 = np.random.randint(0, 1, 100)
    randomValuesComponent3 = np.random.randint(0, 1, 100)
    randomValues = randomValuesComponent1 + randomValuesComponent2 + randomValuesComponent3
    randomValuesBinComponent1 = np.random.randint(0, 2, 100)
    randomValuesBinComponent2 = np.random.randint(0, 2, 100)
    randomValuesBinComponent3 = np.random.randint(0, 2, 100)
    randomBinValues = randomValuesBinComponent1 & randomValuesBinComponent2 & randomValuesBinComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.randint(0, 1, 100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "labelTTE": randomValues,
        "label": randomBinValues
    })
    return df



@pytest.fixture
def sample_0to1_data_multiple_features_noisy():
    np.random.seed(42)
    randomValuesComponent1 = np.random.rand(100) / 3
    randomValuesComponent2 = np.random.rand(100) / 3
    randomValuesComponent3 = np.random.rand(100) / 3
    randomValues = randomValuesComponent1 + randomValuesComponent2 + randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.rand(100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "label": randomValues
    })
    return df



@pytest.fixture
def sample_0to1_data_heldout_multiple_features_noisy():
    np.random.seed(41)
    randomValuesComponent1 = np.random.rand(100) / 3
    randomValuesComponent2 = np.random.rand(100) / 3
    randomValuesComponent3 = np.random.rand(100) / 3
    randomValues = randomValuesComponent1 + randomValuesComponent2 + randomValuesComponent3
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(20, 60, 100),
        "RaceEth": np.random.choice([0, 1], 100),
        "feature1": np.random.rand(100),
        "feature2": randomValuesComponent2,
        "feature3": randomValuesComponent3,
        "feature4": np.random.rand(100),
        "label": randomValues
    })
    return df





def test_logistic_model_init(sample_classification_data_one_feature):
    model = LogisticModel(data=sample_classification_data_one_feature, id_column="id", target_column=["label"])
    
    #Ensure only the features are in the X matrix
    assert np.array_equal(model.X.columns, ["age", "RaceEth", "feature1", "feature2"])

    assert model.who.name == "id"
    assert np.array_equal(model.y.columns, ["label"])
    assert model.who_test.shape[0] == 25

    #Ensure data properly split
    assert model.X_train.shape[0] == 75
    assert model.X_test.shape[0] == 25
    assert model.y_train.shape[0] == 75
    assert model.y_test.shape[0] == 25



def test_logistic_model_select_features_one_feature(sample_classification_data_one_feature):
    model = LogisticModel(data=sample_classification_data_one_feature, id_column="id", target_column=["label"])
    model.selectFeatures()
    assert isinstance(model.selected_features, list)
    assert model.selected_features == ["feature1"]
    assert all(f in model.X.columns for f in model.selected_features)



def test_logistic_model_select_features_multiple_features(sample_classification_data_multiple_features):
    model = LogisticModel(data=sample_classification_data_multiple_features, id_column="id", target_column=["label"])
    model.selectFeatures()
    assert isinstance(model.selected_features, list)
    assert model.selected_features == ["feature1", "feature2", "feature3"]
    assert all(f in model.X.columns for f in model.selected_features)



def test_logistic_model_train(sample_classification_data_multiple_features):
    model = LogisticModel(data=sample_classification_data_multiple_features, id_column="id", target_column=["label"])
    model.selectFeatures()
    model.train()
    assert model.model is not None
    assert hasattr(model.model, "predict_proba")



def test_logistic_model_evaluation(sample_classification_data_multiple_features_noisy, sample_classification_data_heldout_multiple_features_noisy):
    model = LogisticModel(data=sample_classification_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    model.train()
    results = model.evaluate(sample_classification_data_heldout_multiple_features_noisy)
    assert results[1]["roc"] == 0.7797619047619048
    assert np.array_equal(results[1]["confusion_matrix"], np.array([[47, 37], [ 0, 16]]))
    assert results[1]["precision"] == 0.3018867924528302
    assert results[1]["recall"] == 1
    assert results[1]["demographics"] == '53 NHW, 47 Refused/Missing'
    assert results[1]["training_demographics"] == '40 NHW, 35 Refused/Missing'
    assert results[3]["roc"] == 0.8043478260869565
    assert np.array_equal(results[3]["confusion_matrix"], np.array([[14, 9], [ 0, 2]]))
    assert results[3]["precision"] == 0.18181818181818182
    assert results[3]["recall"] == 1
    assert results[3]["demographics"] == '15 Refused/Missing, 10 NHW'
    assert results[3]["training_demographics"] == '40 NHW, 35 Refused/Missing'



def test_negative_binomial_model_select_features(sample_integer_data_multiple_features_noisy):
    # Behavioral: LassoCV chooses the L1 strength by cross-validation (#22), so the
    # exact feature list is not pinned — assert structural properties instead.
    model = NegativeBinomialModel(data=sample_integer_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    assert isinstance(model.selected_features, list)
    assert len(model.selected_features) > 0
    assert all(f in model.X.columns for f in model.selected_features)
    assert "id" not in model.selected_features
    assert "label" not in model.selected_features

def test_negative_binomial_model_train(sample_integer_data_multiple_features_noisy):
    model = NegativeBinomialModel(data=sample_integer_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    model.train()
    assert model.model is not None
    assert hasattr(model.model, "predict")

def test_negative_binomial_model_evaluation(sample_integer_data_multiple_features_noisy, sample_integer_data_heldout_multiple_features_noisy):
    model = NegativeBinomialModel(data=sample_integer_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    model.train()
    results = model.evaluate(sample_integer_data_heldout_multiple_features_noisy)
    # Behavioral: with CV-selected features the exact metric values are not pinned.
    # Verify the evaluation contract instead (keys, non-negative errors, rmse==sqrt(mse)).
    for evals in (results[1], results[3]):
        for key in ("mse", "rmse", "mae", "pearson_r", "mcfadden_r2", "demographics", "training_demographics"):
            assert key in evals, f"Missing key: {key}"
        assert evals["mse"] >= 0
        assert evals["mae"] >= 0
        assert evals["rmse"] == pytest.approx(np.sqrt(evals["mse"]))






def test_cox_proportional_hazard_select_features(sample_survival_data_multiple_features_noisy):
    # Behavioral: LassoCV chooses the L1 strength by cross-validation (#22), so the
    # exact feature list is not pinned — assert structural properties instead.
    model = CoxProportionalHazard(data=sample_survival_data_multiple_features_noisy, id_column="id", target_column=["labelTTE", "label"])
    model.selectFeatures()
    assert isinstance(model.selected_features, list)
    assert len(model.selected_features) > 0
    assert all(f in model.X.columns for f in model.selected_features)
    assert "id" not in model.selected_features
    assert not {"labelTTE", "label"} & set(model.selected_features)

def test_cox_proportional_hazard_train(sample_survival_data_multiple_features_noisy):
    model = CoxProportionalHazard(data=sample_survival_data_multiple_features_noisy, id_column="id", target_column=["labelTTE", "label"])
    model.selectFeatures()
    model.train()
    assert model.model is not None
    assert hasattr(model.model, "predict_median")

def test_cox_proportional_hazard_evaluation(sample_survival_data_multiple_features_noisy, sample_survival_data_heldout_multiple_features_noisy):
    model = CoxProportionalHazard(data=sample_survival_data_multiple_features_noisy, id_column="id", target_column=["labelTTE", "label"])
    model.selectFeatures()
    model.train()
    results = model.evaluate(sample_survival_data_heldout_multiple_features_noisy)
    # Behavioral: with CV-selected features the exact C-index is not pinned.
    for evals in (results[1], results[3]):
        for key in ("concordance_index", "demographics", "training_demographics"):
            assert key in evals, f"Missing key: {key}"
        assert 0.0 <= evals["concordance_index"] <= 1.0



# ---------------------------------------------------------------------------
# Regression guards for issue #22.
#
# The fixtures above give the outcome as an exact sum of features, so a
# hardcoded Lasso(alpha=30) still retains signal and the bug stays invisible.
# The fixtures below reproduce the real-data condition instead: a small-variance
# count/time outcome against large-scale features, where alpha=30 shrinks every
# coefficient to zero and the endpoints become unrunnable.
# ---------------------------------------------------------------------------


def _weak_signal_frame(seed: int) -> pd.DataFrame:
    """Build a weak-signal frame: low-variance counts against large-scale features.

    ``feature2`` and ``feature3`` carry small true effects on a log-mean scale,
    mirroring the real CTN-0094 endpoints. Note that LassoCV is not a consistent
    selector and will sometimes also retain a noise column, so tests assert that
    true signal is recovered rather than that noise is excluded.
    """
    rng = np.random.default_rng(seed)
    n = 400
    feature2 = rng.normal(50, 25, n)
    feature3 = rng.normal(100, 40, n)
    linear = 0.6 + 0.006 * (feature2 - 50) + 0.004 * (feature3 - 100)
    return pd.DataFrame({
        "id": range(n),
        "age": rng.integers(20, 60, n),
        "RaceEth": rng.choice([0, 1], n),
        "feature1": rng.normal(75, 30, n),
        "feature2": feature2,
        "feature3": feature3,
        "label": rng.poisson(np.exp(linear))
    })


@pytest.fixture
def sample_integer_data_weak_signal():
    return _weak_signal_frame(seed=0)


@pytest.fixture
def sample_survival_data_weak_signal():
    df = _weak_signal_frame(seed=1)
    rng = np.random.default_rng(2)
    # Shift counts to strictly positive durations; Cox cannot take a zero time.
    df["labelTTE"] = df["label"] + 1
    df["label"] = rng.integers(0, 2, len(df))
    return df


def test_negative_binomial_weak_signal_hardcoded_alpha_selects_nothing(sample_integer_data_weak_signal):
    """The #22 failure mode: alpha=30 zeroes every coefficient on weak signal."""
    model = NegativeBinomialModel(data=sample_integer_data_weak_signal, id_column="id",
                                  target_column=["label"], seed=42)
    with pytest.raises(ValueError, match="No features were selected"):
        model.lasso_feature_selection(model_type="regression", alpha=30)


def test_negative_binomial_weak_signal_cv_recovers_features(sample_integer_data_weak_signal):
    """The fix: a CV-chosen alpha recovers the true signal and trains end-to-end."""
    model = NegativeBinomialModel(data=sample_integer_data_weak_signal, id_column="id",
                                  target_column=["label"], seed=42)
    model.selectFeatures()
    assert "feature2" in model.selected_features
    assert "feature3" in model.selected_features
    model.train()


def test_cox_weak_signal_hardcoded_alpha_selects_nothing(sample_survival_data_weak_signal):
    """The #22 failure mode on the survival endpoint."""
    model = CoxProportionalHazard(data=sample_survival_data_weak_signal, id_column="id",
                                  target_column=["labelTTE", "label"], seed=42)
    with pytest.raises(ValueError, match="No features were selected"):
        model.lasso_feature_selection(model_type="regression", alpha=30)


def test_cox_weak_signal_cv_recovers_features(sample_survival_data_weak_signal):
    """The fix: the survival endpoint selects and trains end-to-end on weak signal."""
    model = CoxProportionalHazard(data=sample_survival_data_weak_signal, id_column="id",
                                  target_column=["labelTTE", "label"], seed=42)
    model.selectFeatures()
    assert "feature2" in model.selected_features
    assert "feature3" in model.selected_features
    model.train()


def test_mcfadden_r2_is_invariant_to_evaluation_set_size(sample_integer_data_weak_signal):
    """McFadden's R^2 must reflect fit quality, not the train/eval size ratio.

    Previously the full term came from the training fit while the null was fit
    on the evaluation set, so the statistic scaled with n_train / n_eval and was
    reliably large and negative. Scoring the same model on nested evaluation
    sets drawn from one distribution must now give stable values.
    """
    model = NegativeBinomialModel(data=sample_integer_data_weak_signal, id_column="id",
                                  target_column=["label"], seed=42)
    model.selectFeatures()
    model.train()

    heldout = _weak_signal_frame(seed=3)
    scores = []
    for size in (200, 100, 50):
        subset = heldout.iloc[:size]
        _, evals = model._evaluateOnValidation(subset, subset[["label"]], subset["id"])
        scores.append(evals["mcfadden_r2"])

    assert all(abs(s) < 1.0 for s in scores), f"implausible McFadden values: {scores}"
    assert max(scores) - min(scores) < 0.5, f"McFadden tracks evaluation size: {scores}"


def test_negative_binomial_pearson_r_is_a_scalar(sample_integer_data_weak_signal):
    """pearson_r must be a single float, not a broadcast (n, n) result object."""
    model = NegativeBinomialModel(data=sample_integer_data_weak_signal, id_column="id",
                                  target_column=["label"], seed=42)
    model.selectFeatures()
    model.train()

    heldout = _weak_signal_frame(seed=4)
    _, evals = model._evaluateOnValidation(heldout, heldout[["label"]], heldout["id"])
    pearson_r = evals["pearson_r"]
    assert isinstance(pearson_r, float)
    assert not np.isnan(pearson_r)
    assert -1.0 <= pearson_r <= 1.0







def _beta_frame(seed: int, signal: float, n: int = 300) -> pd.DataFrame:
    """Build a (0, 1)-bounded outcome whose dependence on the features is tunable.

    ``signal`` scales the true effect of ``feature2``; at 0.0 the outcome is
    pure noise, so a sound pseudo-R^2 must rise with it.
    """
    rng = np.random.default_rng(seed)
    feature2 = rng.normal(0, 1, n)
    mu = 1 / (1 + np.exp(-(0.2 + signal * feature2)))
    return pd.DataFrame({
        "id": range(n),
        "age": rng.integers(20, 60, n),
        "RaceEth": rng.choice([0, 1], n),
        "feature1": rng.normal(0, 1, n),
        "feature2": feature2,
        "feature3": rng.normal(0, 1, n),
        "label": rng.beta(mu * 20, (1 - mu) * 20)
    })


def _beta_pseudo_r2(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> float:
    model = BetaRegression(data=train_df, id_column="id", target_column=["label"], seed=42)
    model.selected_features = ["feature1", "feature2", "feature3"]
    model.train()
    _, evals = model._evaluateOnValidation(eval_df, eval_df[["label"]], eval_df["id"])
    return evals["pseudo_r2"]


def test_beta_pseudo_r2_increases_with_signal():
    """A stronger true effect must score higher.

    McFadden's ``1 - ll_full / ll_null`` inverted here: a Beta log-likelihood is
    a log-density and is routinely positive, so better fits scored further below
    zero. Cox-Snell uses the log-likelihood difference and orders correctly.
    """
    scores = [_beta_pseudo_r2(_beta_frame(0, s), _beta_frame(1, s)) for s in (0.0, 0.5, 1.5)]
    assert scores == sorted(scores), f"pseudo_r2 not monotonic in signal: {scores}"
    # With no true effect the score must not be appreciably positive. It sits a
    # little below zero rather than exactly at it, because the two noise columns
    # are still fitted and so generalize slightly worse than the null — that is
    # correct out-of-sample behavior, hence only a loose lower bound.
    assert -0.5 < scores[0] < 0.15, f"no-signal case should sit near zero: {scores[0]}"
    assert scores[-1] > 0.5, f"strong-signal case should be clearly positive: {scores[-1]}"


def test_beta_pseudo_r2_is_stable_across_evaluation_set_size():
    """The score must reflect fit quality, not how big the evaluation set is."""
    train = _beta_frame(0, 0.8)
    heldout = _beta_frame(1, 0.8)
    scores = [_beta_pseudo_r2(train, heldout.iloc[:size]) for size in (300, 150, 75)]
    assert max(scores) - min(scores) < 0.2, f"pseudo_r2 tracks evaluation size: {scores}"


def test_beta_regression_select_features(sample_0to1_data_multiple_features_noisy):
    model = BetaRegression(data=sample_0to1_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    assert isinstance(model.selected_features, list)
    assert model.selected_features == ["feature2", "feature3"]
    assert all(f in model.X.columns for f in model.selected_features)

def test_beta_regression_train(sample_0to1_data_multiple_features_noisy):
    model = BetaRegression(data=sample_0to1_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    model.train()
    assert model.model is not None
    assert hasattr(model.model, "predict")

def test_beta_regression_evaluation(sample_0to1_data_multiple_features_noisy, sample_0to1_data_heldout_multiple_features_noisy):
    model = BetaRegression(data=sample_0to1_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    model.train()
    results = model.evaluate(sample_0to1_data_heldout_multiple_features_noisy)
    assert results[1]["mse"] == pytest.approx(0.008120363539, rel=1e-6)
    assert results[1]["rmse"] == pytest.approx(0.090113059760, rel=1e-6)
    assert results[1]["mae"] == pytest.approx(0.077157156966, rel=1e-6)
    assert results[1]["pearson_r"] == pytest.approx(0.857509435637, rel=1e-6)
    assert results[1]["demographics"] == '56 Refused/Missing, 44 NHW'
    assert results[1]["training_demographics"] == '46 Refused/Missing, 29 NHW'





# def test_train_and_evaluate_models_multiple_subsets(sample_classification_data_multiple_features_noisy, sample_classification_data_heldout_multiple_features_noisy):
#     subset1 = sample_classification_data_multiple_features_noisy.iloc[:50]
#     subset2 = sample_classification_data_multiple_features_noisy.iloc[50:]
#     merged_subsets = [subset1, subset2]

#     selected_outcome = {
#         "endpointType": EndpointType.LOGICAL,
#         "columnsToUse": ["label"]
#     }

#     results = train_and_evaluate_models(
#         merged_subsets,
#         id_column="id",
#         selected_outcome=selected_outcome,
#         processed_data_heldout=sample_classification_data_heldout_multiple_features_noisy
#     )

#     assert results.shape[0] == 2