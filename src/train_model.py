import logging  # Import the logging module for logging messages
import random  # Import the random module for random number generation
from typing import Optional  # Import the Optional type hint for optional arguments

import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation and analysis

from joblib import dump, load  # Import joblib for saving and loading models
from sklearn.exceptions import NotFittedError  # Import NotFittedError for handling cases where a model hasn't been fitted
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression  # Import LogisticRegression models
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling
from sklearn.pipeline import Pipeline, make_pipeline  # Import Pipeline and make_pipeline for creating ML pipelines
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score  # Import metrics for model evaluation
import statsmodels.api as sm  # Import statsmodels for statistical modeling, especially Negative Binomial regression

class OutcomeModel:
    def __init__(self, data, target_column, seed=None):
        self.data = data  # Store the entire input dataset
        self.target_column = target_column  # Store the target column name
        self.X = data.drop([target_column], axis=1)  # Create the feature matrix by dropping the target column
        self.y = data[target_column]  # Extract the target variable
        self.model = None  # Initialize the model attribute to None
        self.selected_features = None  # Initialize the selected features attribute to None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(  # Split the data into training and testing sets
            self.X, self.y, test_size=0.25
        )
        self.best_threshold = 0.5  # Set the default classification threshold
        self.seed = seed  # Store the random seed for reproducibility
        if seed is not None:  # Set the random seed if provided
            np.random.seed(seed)
            random.seed(seed)

    def train(self):
        pass  # Placeholder for training implementation

    def evaluate(self):
        pass  # Placeholder for evaluation implementation

class LogisticModel(OutcomeModel):
    def __init__(self, data: pd.DataFrame, target_column: str, Cs: Optional[list] = None, required_features: Optional[list] = None, seed: Optional[int] = None):
        super().__init__(data, target_column, seed)  # Call the parent class constructor
        self.Cs = Cs if Cs is not None else [1.0]  # Set the regularization strength (C)
        self.required_features = required_features or ['who', 'is_female']  # Set the required features
        self.selected_features = None  # Initialize the selected features attribute
        self.model = None  # Initialize the model attribute

        logging.info(f"LogisticModel initialized successfully with Cs={self.Cs} and required_features={self.required_features}.")

    def feature_selection_and_model_fitting(self) -> None:
        logging.info("Starting feature selection using L1 regularization...")
        # Create a pipeline for feature scaling and logistic regression with L1 regularization
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(penalty='l1', solver='saga', C=self.Cs[0], max_iter=10000))
        ])
        pipeline.fit(self.X_train, self.y_train)  # Fit the pipeline to the training data
        logistic_model = pipeline.named_steps['logistic']  # Extract the logistic regression model

        # Check if the model coefficients are not None
        if logistic_model.coef_ is None:
            raise ValueError("The model coefficients are None. The model did not fit properly.")

        # Select features with non-zero coefficients
        self.selected_features = self.X_train.columns[logistic_model.coef_.flatten() != 0].tolist()
        if not self.selected_features:
            raise ValueError("No features were selected. Check if your L1 regularization is too strong.")

        # Ensure required features are included
        required_features = ['who', 'is_female']
        for feature in required_features:
            if feature in self.X_train.columns and feature not in self.selected_features:
                self.selected_features.append(feature)
                logging.info(f"Manually including required feature: {feature}")

        # Update the training and testing sets with the selected features
        self.X_train = self.X_train[self.selected_features]
        self.X_test = self.X_test[self.selected_features]

        # Create a new logistic regression model without L1 regularization
        self.model = LogisticRegression(max_iter=10000, C=self.Cs[0])
        self.model.fit(self.X_train, self.y_train)  # Fit the model to the training data

        if self.model is None:
            raise ValueError("The model was not properly instantiated.")

        logging.info("Feature selection and model fitting completed successfully.")

    def find_best_threshold(self):
        logging.info("Calculating the dynamic threshold based on the proportion of positive outcomes...")
        # Calculate the proportion of positive outcomes in the test set
        positive_proportion = self.y_test.mean()
        negative_proportion = 1 - positive_proportion

        # Log the ratio of 1s to 0s in the outcome variable
        logging.info(f"Ratio of 1s to 0s in the outcome variable: {positive_proportion:.2f} : {negative_proportion:.2f}")

        # Set the best threshold based on the proportion of positive outcomes
        self.best_threshold = positive_proportion

        logging.info(f"Dynamic threshold set to: {self.best_threshold}")

    def train(self):
        self.feature_selection_and_model_fitting()
        self.find_best_threshold()

    def evaluate(self) -> None:
        if self.model is None:
            raise NotFittedError("The model is not fitted. Call train() before evaluation.")

        if self.X_test is None or self.X_test.empty:
            raise ValueError("X_test is empty. Ensure the train-test split was successful.")

        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        if y_pred_proba is None:
            raise ValueError("y_pred_proba is None. Check model predictions.")

        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        confusion = confusion_matrix(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)

        logging.info(f"ROC AUC: {roc_auc}")
        logging.info(f"Confusion Matrix:\n{confusion}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")

        if y_pred_proba is None:
            raise ValueError("y_pred_proba is None. This may be caused by a failed model prediction.")

        return zip(self.X_test['who'], y_pred_proba)

class NegativeBinomialModel(OutcomeModel):
    def train(self, subset, selected_outcome):
        endog = [selected_outcome]
        exog = sm.add_constant(subset)
        self.model = sm.NegativeBinomial(endog, exog, loglike_method='nb2')
        self.model.fit()
        logging.info("AARON DEBUG: ", self.model.summary())

    def predict(self):
        return self.model.evaluate_model()