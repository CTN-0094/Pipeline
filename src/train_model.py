import logging  # For logging messages
import random  # For random number generation
from typing import Optional  # For optional type hints

import numpy as np  # For numerical operations
import pandas as pd  # For DataFrame manipulation

from joblib import dump, load  # For saving/loading models
from sklearn.exceptions import NotFittedError  # Handle unfitted model errors
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.pipeline import Pipeline  # For creating a machine learning pipeline
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score  # For model evaluation

import statsmodels.api as sm  # For statistical models (like Negative Binomial)


class OutcomeModel:
    """Base class for handling an outcome model."""
    def __init__(self, data: pd.DataFrame, target_column: str, seed: Optional[int] = None):
        """
        Initialize the OutcomeModel class.

        Parameters:
        - data (pd.DataFrame): Full input dataset.
        - target_column (str): Target variable to predict.
        - seed (int, optional): Seed for reproducibility.
        """
        self.data = data  # Store the full dataset
        self.target_column = target_column  # Target column name
        
        # Store 'who' column separately for tracking if it exists in the data
        self.who = data['who'] if 'who' in data.columns else None  
        
        # Drop 'who' and target_column from X (feature set) 
        self.X = data.drop([target_column, 'who'], axis=1, errors='ignore')  
        
        self.y = data[target_column]  # Target variable (y)
        self.model = None  # Placeholder for the trained model
        self.selected_features = None  # Placeholder for selected features
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=seed  # Split the dataset
        )
        
        # Track 'who' column for test set (after the train-test split)
        self.who_test = self.who.loc[self.X_test.index] if self.who is not None else None  
        
        self.best_threshold = 0.5  # Default classification threshold

        # Set random seed for reproducibility
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def train(self):
        """Placeholder method for training."""
        pass

    def evaluate(self):
        """Placeholder method for evaluation."""
        pass


class LogisticModel(OutcomeModel):
    """Logistic regression model with L1 regularization for feature selection."""
    def __init__(self, data: pd.DataFrame, target_column: str, Cs: Optional[list] = None, seed: Optional[int] = None):
        """
        Initialize the LogisticModel class.

        Parameters:
        - data (pd.DataFrame): Full input dataset.
        - target_column (str): Target variable to predict.
        - Cs (list, optional): List of C values for L1 regularization (default: [1.0]).
        - seed (int, optional): Seed for reproducibility.
        """
        super().__init__(data, target_column, seed)
        self.Cs = Cs if Cs is not None else [1.0]  # Default regularization strength (C)
        logging.info(f"LogisticModel initialized successfully with Cs={self.Cs}.")

    def feature_selection_and_model_fitting(self) -> None:
        """Select features using L1 regularization and fit the logistic regression model."""
        logging.info("Starting feature selection using L1 regularization...")
        
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('logistic', LogisticRegression(penalty='l1', solver='saga', C=self.Cs[0], max_iter=10000))
            ])
            pipeline.fit(self.X_train, self.y_train)  # Fit the pipeline to the training data

            logistic_model = pipeline.named_steps['logistic']

            if logistic_model.coef_ is None:
                raise ValueError("The model coefficients are None. The model did not fit properly.")

            self.selected_features = self.X_train.columns[logistic_model.coef_.flatten() != 0].tolist()

            self.X_train = self.X_train[self.selected_features]
            self.X_test = self.X_test[self.selected_features]

            self.model = LogisticRegression(max_iter=10000, C=self.Cs[0])
            self.model.fit(self.X_train, self.y_train)

            logging.info("Feature selection and model fitting completed successfully.")

        except Exception as e:
            logging.error(f"Error during feature selection and model fitting: {e}")
            raise

    def find_best_threshold(self):
        """Determine the best classification threshold based on positive outcome proportion."""
        try:
            positive_proportion = self.y_test.mean()  # Calculate the proportion of positive outcomes
            self.best_threshold = positive_proportion  # Set threshold to be the proportion of positive outcomes
            logging.info(f"Dynamic threshold set to: {self.best_threshold}")
        except Exception as e:
            logging.error(f"Error during threshold evaluation: {e}")
            raise

    def train(self):
        """Train the logistic regression model."""
        self.feature_selection_and_model_fitting()
        self.find_best_threshold()

    def evaluate(self) -> None:
        """Evaluate the logistic model."""
        try:
            if self.model is None:
                raise NotFittedError("The model is not fitted. Call train() before evaluation.")

            if self.X_test is None or self.X_test.empty:
                raise ValueError("X_test is empty. Ensure the train-test split was successful.")

            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]  
            
            if y_pred_proba is None:
                raise ValueError("y_pred_proba is None. This may be caused by a failed model prediction.")
            
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)

            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            confusion = confusion_matrix(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)

            logging.info(f"ROC AUC: {roc_auc}")
            logging.info(f"Confusion Matrix:\n{confusion}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")

            # Use self.who_test to track 'who' for test set
            return zip(self.who_test, y_pred_proba)  # Return the 'who' identifiers and prediction probabilities

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise


class NegativeBinomialModel(OutcomeModel):
    """Negative Binomial model implementation."""
    def train(self, subset, selected_outcome):
        """Train a Negative Binomial model."""
        endog = [selected_outcome]
        exog = sm.add_constant(subset)
        self.model = sm.NegativeBinomial(endog, exog, loglike_method='nb2')
        self.model.fit()
        logging.info("Negative Binomial Model trained successfully.")

    def predict(self):
        """Make predictions with the trained Negative Binomial model."""
        return self.model.evaluate_model()
