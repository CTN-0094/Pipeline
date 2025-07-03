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
    assert results[1]["demographics"] == '53 1, 47 0'
    assert results[1]["training_demographics"] == '40 1, 35 0'
    assert results[3]["roc"] == 0.8043478260869565
    assert np.array_equal(results[3]["confusion_matrix"], np.array([[14, 9], [ 0, 2]]))
    assert results[3]["precision"] == 0.18181818181818182
    assert results[3]["recall"] == 1
    assert results[3]["demographics"] == '15 0, 10 1'
    assert results[3]["training_demographics"] == '40 1, 35 0'



def test_negative_binomial_model_select_features(sample_integer_data_multiple_features_noisy):
    model = NegativeBinomialModel(data=sample_integer_data_multiple_features_noisy, id_column="id", target_column=["label"])
    model.selectFeatures()
    assert isinstance(model.selected_features, list)
    assert model.selected_features == ["feature2", "feature3"]
    assert all(f in model.X.columns for f in model.selected_features)

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
    print("RESULTS\n\n\n\n\n\n", results)
    assert results[1]["mse"] == 3211.027355868404
    assert results[1]["rmse"] == 56.66592764500025
    assert results[1]["mae"] == 47.779331871074746
    #assert results[1]["pearson_r"][0] > 0
    assert results[1]["mcfadden_r2"] == 0.14728095353339066
    assert results[1]["demographics"] == '59 1, 41 0'
    assert results[1]["training_demographics"] == '40 1, 35 0'
    assert results[3]["mse"] == 3998.2214606095818
    assert results[3]["rmse"] == 63.231491051607996
    assert results[3]["mae"] == 52.37012632059721
    #assert results[1]["pearson_r"][0] > 0
    assert results[3]["mcfadden_r2"] == -2.398448108914756
    assert results[3]["demographics"] == '13 0, 12 1'
    assert results[3]["training_demographics"] == '40 1, 35 0'






def test_cox_proportional_hazard_select_features(sample_survival_data_multiple_features_noisy):
    model = CoxProportionalHazard(data=sample_survival_data_multiple_features_noisy, id_column="id", target_column=["labelTTE", "label"])
    model.selectFeatures()
    assert isinstance(model.selected_features, list)
    assert model.selected_features == ["feature2", "feature3"]
    assert all(f in model.X.columns for f in model.selected_features)

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
    assert results[1]["concordance_index"] == .5
    assert results[1]["demographics"] == '53 1, 47 0'
    assert results[1]["training_demographics"] == '39 0, 36 1'







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
    assert results[1]["mse"] == 0.008120363539425442
    assert results[1]["rmse"] == 0.09011305976064425
    assert results[1]["mae"] == 0.07715715696613251
    assert results[1]["pearson_r"] == 0.8575094356375779
    #assert results[1]["mcfadden_r2"] > 0.5
    assert results[1]["demographics"] == '56 0, 44 1'
    assert results[1]["training_demographics"] == '46 0, 29 1'





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