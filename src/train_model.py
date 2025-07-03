import logging  # For logging messages
import random  # For random number generation
from typing import Optional  # For optional type hints

import numpy as np  # For numerical operations
import pandas as pd  # For DataFrame manipulation

from joblib import dump, load  # For saving/loading models
from sklearn.exceptions import NotFittedError  # Handle unfitted model errors
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression  # Logistic regression model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.pipeline import Pipeline  # For creating a machine learning pipeline
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score # For model evaluation
import statsmodels.api as sm  # For statistical models (like Negative Binomial)
from statsmodels.othermod.betareg import BetaModel
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.stats import pearsonr
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index



class OutcomeModel:
    """Base class for handling an outcome model."""
    def __init__(self, data: pd.DataFrame, id_column: str, target_column: list, seed: Optional[int] = None):
        """
        Initialize the OutcomeModel class.

        Parameters:
        - data (pd.DataFrame): Full input dataset.
        - target_column (list<str>): Target variable(s) to predict.
        - seed (int, optional): Seed for reproducibility.
        """
        self.data = data  # Store the full dataset
        self.target_column = target_column  # Target column name
        
        self.id_column = id_column
        # Store 'who' column separately for tracking if it exists in the data
        self.who = data[id_column] if id_column in data.columns else None  
        
        # Drop 'who' and target_column from X (feature set) 
        self.X = data.drop([id_column] + target_column, axis=1, errors='ignore')  
        
        self.y = data[target_column]  # Target variable (y)
        self.model = None  # Placeholder for the trained model
        self.selected_features = self.X.columns  # Placeholder for selected features
        
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
        
        # Initialize global variables for callback
        self.iteration_count = 0
        self.loss_values = []
        self.scaler = None

    def selectFeatures(self):
        self.lasso_feature_selection()

    def train(self):
        """Placeholder method for training."""
        pass

    def evaluate(self, processed_data_heldout) -> None:
        """Evaluate the logistic model."""
        try:
            if self.model is None:
                raise NotFittedError("The model is not fitted. Call train() before evaluation.")
            if self.X_test is None or self.X_test.empty:
                raise ValueError("X_test is empty. Ensure the train-test split was successful.")

            # Heldout Evaluation
            id = processed_data_heldout[self.id_column]
            heldout_X = processed_data_heldout
            heldout_y = processed_data_heldout[self.target_column]
            heldout_predictions, heldout_evaluations = self._evaluateOnValidation(heldout_X, heldout_y, id)
            # Subset Evaluation
            subset_predictions, subset_evaluations = self._evaluateOnValidation(self.X_test, self.y_test, self.who_test)

            # Calculate and add training demographics *new*
            training_demographics = self._countDemographic(self.X_train)
            subset_evaluations["training_demographics"] = training_demographics
            heldout_evaluations["training_demographics"] = training_demographics

            # Return all evaluations, including training demographics
            return heldout_predictions, heldout_evaluations, subset_predictions, subset_evaluations

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

    def _evaluateOnValidation(self, X, y, id):
        """Placeholder method for Evaluating Validation sets."""
        pass

    def lasso_feature_selection(self, model_type = 'classification', alpha=0.01):
        """
        Perform feature selection using Lasso regression.
        
        Parameters:
        -----------
        alpha : float, default=0.1
            The regularization strength. Higher values result in fewer features selected.
            
        Returns:
        --------
        selected_features : list
            List of column names of the selected features.
        """
        try:
            if model_type == 'regression':
                model = Lasso(alpha=alpha, random_state=42)
            elif model_type == 'classification':
                model = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=10000)
            else:
                raise ValueError("model_type must be either 'regression' or 'classification'")

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            pipeline.fit(self.X_train, self.y_train)
            

            # Extract the fitted logistic regression model from the pipeline
            model = pipeline.named_steps['model']
            
            # Check if the model learned any coefficients; raise an error if not
            if model.coef_ is None:
                raise ValueError("The model coefficients are None. The model did not fit properly.")
            coefficients = model.coef_
            if isinstance(coefficients[0], np.ndarray):
                coefficients = coefficients[0]
            # Select the features where the L1 regularization retained non-zero coefficients
            self.selected_features = self.X_train.columns[coefficients.flatten() != 0].tolist()

            # If no features were selected, raise an error to signal potential over-regularization
            if not self.selected_features:
                raise ValueError("No features were selected. Check your regularization strength.")

            # Log the number of features selected
            feature_names = self.X_train.columns
            print(f"Lasso feature selection completed. Selected {len(self.selected_features)} out of {len(feature_names)} features.")
            print(f"Features are: ", self.selected_features)
        except Exception as e:
            logging.error(f"Error during Lasso feature selection: {e}")
            raise



    def _countDemographic(self, data):
        
        demographic_counts = data['RaceEth'].value_counts().to_dict()
        dem_string = ", ".join([f"{v} {k}" for k, v in demographic_counts.items()])
        logging.info(f"demographic makeup: {dem_string}")
        return dem_string



class LogisticModel(OutcomeModel):
    """Logistic regression model with L1 regularization for feature selection."""

    def __init__(self, data: pd.DataFrame, id_column: str, target_column: str, Cs: Optional[list] = None, seed: Optional[int] = None):
        """
        Initialize the LogisticModel class.

        Parameters:
        - data (pd.DataFrame): The full dataset containing features and the target column.
        - target_column (str): The name of the target column for prediction.
        - Cs (list, optional): List of regularization strengths (C) for L1 regularization. Defaults to [1.0].
        - seed (int, optional): Random seed for reproducibility.
        """
        # Call the parent class constructor to handle dataset initialization and seeding
        super().__init__(data, id_column, target_column, seed)

        # Set the regularization strength for L1 regularization. If not provided, defaults to [1.0]
        self.Cs = Cs if Cs is not None else [1.0]

        # Log the initialization details for traceability and debugging purposes
        logging.info(f"LogisticModel initialized successfully with Cs={self.Cs}.")

    def _feature_selection_and_model_fitting(self) -> None:
        """
        Select features using L1 regularization and fit the logistic regression model.

        - L1 regularization is used here to perform feature selection by shrinking some coefficients to zero.
        """
        logging.info("Starting feature selection using L1 regularization...")

        try:
            # Train a new logistic regression model on the reduced feature set
            # Note: This logistic regression does not use L1 regularization but the selected features
            self.model = LogisticRegression(max_iter=10000, C=self.Cs[0])  
            self.model.fit(self.X_train[self.selected_features], self.y_train)  # Fit the final logistic regression model

            logging.info("Feature selection and model fitting completed successfully.")

        # Catch and log any errors that occur during the feature selection or model fitting process
        except Exception as e:
            logging.error(f"Error during feature selection and model fitting: {e}")
            raise  # Re-raise the error to ensure it propagates if not handled elsewhere


    def selectFeatures(self):
        self.lasso_feature_selection(model_type="classification")

    def _find_best_threshold(self):
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
        self._feature_selection_and_model_fitting()
        self._find_best_threshold()


    def _evaluateOnValidation(self, X, y, id):
        y_pred_proba = self.model.predict_proba(X[self.selected_features])[:, 1]
        if y_pred_proba is None: raise ValueError("y_pred_proba is None. This may be caused by a failed model prediction.")
        y_pred = (y_pred_proba >= self.best_threshold.squeeze()).astype(int)

        predictions = zip(id, y_pred)
        evaluations = {
            "roc": roc_auc_score(y, y_pred),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "demographics": self._countDemographic(X)
        }
        return predictions, evaluations
    


class NegativeBinomialModel(OutcomeModel):

    """Negative Binomial model implementation."""
    def train(self):
        # Prepare the data for statsmodels (add a constant term)
        X_with_constant = np.column_stack((np.ones(self.X_train[self.selected_features].shape[0]), self.X_train[self.selected_features]))
        self.model = sm.GLM(self.y_train, X_with_constant, family=sm.families.NegativeBinomial())
        
        self.model = self.model.fit(
            method='bfgs',
            disp=0
        )

        logging.info("NBR model fitting completed successfully.")

    def selectFeatures(self):
        self.lasso_feature_selection(model_type="regression", alpha=30)

    def predict(self):
        """Make predictions with the trained Negative Binomial model."""
        return self.model.evaluate_model() 
    
    def _evaluateOnValidation(self, X, y, id):

        X_with_constant = np.column_stack((np.ones(X[self.selected_features].shape[0]), X[self.selected_features]))

        y_pred = self.model.predict(X_with_constant)

        ll_full = self.model.llf
        X_null = np.ones((X.shape[0], 1))
        null_model = NegativeBinomial(y, X_null).fit(method='bfgs', disp=0)
        ll_null = null_model.llf
        mcfadden_r2 = 1 - (ll_full / ll_null)

        predictions = zip(id, y_pred)
        evaluations = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "pearson_r": pearsonr(y, y_pred),
            "mcfadden_r2": mcfadden_r2,
            "demographics": self._countDemographic(X)
        }
        return predictions, evaluations


#Consider adding accelerated failure time
#Kaplan Meier also working (they are not great for prediction, low bias high variance)
class CoxProportionalHazard(OutcomeModel):

    def train(self):
        try:
            self.model = CoxPHFitter()
            table = pd.concat([self.X_train[self.selected_features], self.y_train], axis=1)
            self.model.fit(table, duration_col=self.y_train.columns[0], event_col=self.y_train.columns[1])

            logging.info("CPH model fitting completed successfully.")

        # Catch and log any errors that occur during the feature selection or model fitting process
        except Exception as e:
            logging.error(f"Error during feature selection and NBR model fitting: {e}")
            raise

    def predict(self):
        """Make predictions with the trained Negative Binomial model."""
        return self.model.evaluate_model() 
    
    def _evaluateOnValidation(self, X, y, id):
        table = pd.concat([X[self.selected_features], y], axis=1)

        ci = concordance_index(table[y.columns[0]], -self.model.predict_partial_hazard(table), table[y.columns[1]])

        predictions = zip(id, self.model.predict_median(X[self.selected_features]))
        evaluations = {
            "concordance_index": ci,
            "demographics": self._countDemographic(X)
        }
        return predictions, evaluations
    
    def selectFeatures(self):
        self.lasso_feature_selection(model_type="regression", alpha=30)




class BetaRegression(OutcomeModel):


    """Negative Binomial model implementation."""
    def train(self):

        # Step 1: Add intercept
        X_with_constant = sm.add_constant(self.X_train[self.selected_features], has_constant='add')

        # Step 2: Fit beta regression
        self.model = BetaModel(endog=self.y_train, exog=X_with_constant)

        # Step 3: Fit with optimizer
        self.model = self.model.fit(method='bfgs', disp=0)

        logging.info("NBR model fitting completed successfully.")

    def selectFeatures(self):
        self.lasso_feature_selection(model_type="regression", alpha=.05)

    def predict(self):
        """Make predictions with the trained Negative Binomial model."""
        return self.model.evaluate_model() 
    
    def _evaluateOnValidation(self, X, y, id):

        X_with_constant = np.column_stack((np.ones(X[self.selected_features].shape[0]), X[self.selected_features]))

        y_pred = self.model.predict(X_with_constant)

        ll_full = self.model.llf
        X_null = np.ones((X.shape[0], 1))
        null_model = BetaModel(endog=y, exog=X_null).fit(method='bfgs', disp=0)
        ll_null = null_model.llf
        mcfadden_r2 = 1 - (ll_full / ll_null)

        y = np.ravel(y)
        y_pred = np.ravel(y_pred)

        r, p = pearsonr(y, y_pred)
        predictions = zip(id, y_pred)
        evaluations = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "pearson_r": r,
            "mcfadden_r2": mcfadden_r2,
            "demographics": self._countDemographic(X)
        }
        print(evaluations)
        return predictions, evaluations


