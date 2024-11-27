import logging
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    precision_score, 
    recall_score
)
from sklearn.exceptions import NotFittedError


class OutcomeModel:
    """
    A base class for predictive models. Handles common functionalities like data splitting, 
    seed setting, and placeholders for training and evaluation methods.
    """
    def __init__(self, data, target_column, seed=None):
        """
        Initializes the OutcomeModel with the dataset, target column, and random seed.

        Args:
            data (pd.DataFrame): The full dataset.
            target_column (str): The name of the target column.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.data = data  # Full dataset
        self.target_column = target_column  # Target variable name

        # Separate features (X) and target (y)
        self.X = data.drop([target_column], axis=1)  # Features
        self.y = data[target_column]  # Target

        # Placeholders for model and selected features
        self.model = None
        self.selected_features = None

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=seed
        )

        # Default threshold for classification
        self.best_threshold = 0.5

        # Set the random seed if provided
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def train(self):
        """To be implemented in subclasses: Trains the model."""
        raise NotImplementedError("The `train` method must be implemented in subclasses.")

    def evaluate(self):
        """To be implemented in subclasses: Evaluates the model."""
        raise NotImplementedError("The `evaluate` method must be implemented in subclasses.")


class LogisticModel(OutcomeModel):
    """
    A logistic regression model class inheriting from OutcomeModel. Includes 
    L1-regularized feature selection and model evaluation methods.
    """
    def __init__(self, data, target_column, Cs=[1.0], seed=None):
        """
        Initializes the LogisticModel with regularization strengths and seed.

        Args:
            data (pd.DataFrame): The dataset.
            target_column (str): The name of the target column.
            Cs (list): List of regularization strengths for L1 regularization. Defaults to [1.0].
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__(data, target_column, seed)  # Initialize OutcomeModel
        self.Cs = Cs  # Regularization strengths for feature selection

    def feature_selection_and_model_fitting(self):
        """
        Performs L1-regularized feature selection and fits the logistic regression model.
        """
        try:
            logging.info("Starting feature selection using L1 regularization...")

            # Create a pipeline with scaling and L1 regularization
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Scale features
                ('logistic', LogisticRegression(
                    penalty='l1', solver='saga', C=self.Cs[0], max_iter=10000
                ))  # Apply L1 regularization
            ])

            # Fit the pipeline to the training data
            pipeline.fit(self.X_train, self.y_train)

            # Extract non-zero coefficient features
            logistic_model = pipeline.named_steps['logistic']
            self.selected_features = self.X_train.columns[logistic_model.coef_.flatten() != 0].tolist()
            logging.info(f"Selected features: {self.selected_features}")

            # Update the training and testing sets with selected features
            self.X_train = self.X_train[self.selected_features]
            self.X_test = self.X_test[self.selected_features]

            # Fit final logistic regression model with selected features
            self.model = LogisticRegression(max_iter=10000, C=self.Cs[0])
            self.model.fit(self.X_train, self.y_train)
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error during feature selection and model fitting: {e}")
            raise

    def find_best_threshold(self):
        """
        Sets the classification threshold dynamically based on the proportion of positive outcomes in the test set.
        """
        try:
            positive_proportion = self.y_test.mean()
            self.best_threshold = positive_proportion
            logging.info(f"Best threshold set dynamically to: {self.best_threshold}")
        except Exception as e:
            logging.error(f"Error in threshold calculation: {e}")
            raise

    def train(self):
        """Runs feature selection, model fitting, and sets the best threshold."""
        self.feature_selection_and_model_fitting()
        self.find_best_threshold()

    def evaluate(self):
        """
        Evaluates the trained model on the test set and logs key metrics.
        """
        try:
            if self.model is None:
                raise NotFittedError("Model has not been trained. Call `train` before evaluation.")

            # Generate predictions and probabilities
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)

            # Calculate metrics
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            confusion = confusion_matrix(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)

            # Log metrics
            logging.info(f"ROC AUC: {roc_auc}")
            logging.info(f"Confusion Matrix:\n{confusion}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise
