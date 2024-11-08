import logging
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.exceptions import NotFittedError

class OutcomeModel:
    def __init__(self, data, target_column, seed=None):
        # Store the full dataset and the name of the target column
        self.data = data
        self.target_column = target_column

        # Separate features (X) and target (y) from the dataset
        self.X = data.drop([target_column], axis=1)  # Features
        self.y = data[target_column]  # Target variable

        # Initialize placeholders for the model and selected features
        self.model = None
        self.selected_features = None

        # Split data into training and testing sets with a 25% test size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=seed
        )

        # Set a default threshold for classification
        self.best_threshold = 0.5

        # Set the random seed if specified to ensure reproducibility
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def train(self):
        """Placeholder method for training; to be implemented in subclasses."""
        pass

    def evaluate(self):
        """Placeholder method for evaluation; to be implemented in subclasses."""
        pass

class LogisticModel(OutcomeModel):
    def __init__(self, data, target_column, Cs=[1.0], seed=None):
        # Initialize the superclass OutcomeModel with data, target_column, and seed
        super().__init__(data, target_column, seed)

        # Store the regularization strengths for L1 regularization (default is 1.0)
        self.Cs = Cs

    def feature_selection_and_model_fitting(self):
        """
        Performs feature selection using L1-regularized logistic regression on training data only,
        then fits a final logistic regression model on the selected features.
        """
        try:
            logging.info("Scaling features and applying L1-regularized logistic regression for feature selection...")

            # Create a pipeline with scaling and L1-regularized logistic regression for feature selection
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Step 1: Scale features
                ('logistic', LogisticRegression(penalty='l1', solver='saga', C=self.Cs[0], max_iter=10000))  # Step 2: L1 regularization
            ])

            # Fit the pipeline on the training data (prevents data leakage)
            pipeline.fit(self.X_train, self.y_train)

            # Extract the logistic regression model from the pipeline
            model = pipeline.named_steps['logistic']

            # Select features with non-zero coefficients (those chosen by L1 regularization)
            self.selected_features = self.X_train.columns[model.coef_.flatten() != 0].tolist()
            logging.info(f"Selected features: {self.selected_features}")

            # Update the training and test sets to include only the selected features
            self.X_train = self.X_train[self.selected_features]
            self.X_test = self.X_test[self.selected_features]

            # Fit a final logistic regression model using only the selected features
            self.model = LogisticRegression(max_iter=10000, C=self.Cs[0])
            self.model.fit(self.X_train, self.y_train)
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during feature selection and model fitting: {e}")

    def find_best_threshold(self):
        """
        Calculates and sets the classification threshold dynamically based on the proportion of positive outcomes
        in the test set.
        """
        try:
            logging.info("Calculating dynamic threshold based on positive outcome proportion in the test set...")

            # Calculate the proportion of positive outcomes in the test set
            positive_proportion = self.y_test.mean()

            # Use the proportion as the best threshold
            self.best_threshold = positive_proportion

            logging.info(f"Dynamic threshold set to: {self.best_threshold}")
        except Exception as e:
            logging.error(f"An error occurred during threshold evaluation: {e}")

    def train(self):
        """Performs feature selection, model fitting, and sets the best threshold."""
        self.feature_selection_and_model_fitting()
        self.find_best_threshold()

    def evaluate(self):
        """
        Evaluates the model on the test set using metrics like ROC AUC, confusion matrix, precision, and recall.
        Logs each metric to provide detailed performance information.
        """
        try:
            # Generate prediction probabilities for the test set
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

            # Apply the best threshold to convert probabilities into binary predictions
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)

            # Calculate various evaluation metrics
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            confusion = confusion_matrix(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)

            # Log each metric to help evaluate model performance
            logging.info(f"ROC AUC Score: {roc_auc}")
            logging.info(f"Confusion Matrix: \n{confusion}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
        except NotFittedError:
            logging.error("The model is not fitted yet. Please train the model before evaluation.")
        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}")

