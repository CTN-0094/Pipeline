import logging
import random
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

# Set seed for reproducibility in numpy and random
#np.random.seed(42)
#random.seed(42)

class LogisticModel:
    def __init__(self, data, target_column, Cs=[1000], cv=5, seed=None):
        self.data = data  # Full dataset
        self.target_column = target_column  # Target variable for prediction
        self.X = data.drop([target_column], axis=1)  # Features matrix
        self.y = data[target_column]  # Target variable
        self.model = None  # Placeholder for the trained logistic regression model
        self.selected_features = None  # Placeholder for features selected by LASSO
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25
        )
        self.Cs = Cs  # Custom regularization strengths for LASSO
        self.cv = cv  # Number of cross-validation folds
        self.best_threshold = 0.5  # Default threshold
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def feature_selection_and_model_fitting(self):
        try:
            logging.info("Scaling features and applying LASSO for feature selection...")
            # Scale features and apply LASSO for feature selection within a pipeline
            lasso = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    penalty='l1', solver='saga', Cs=self.Cs, cv=self.cv, max_iter=10000
                )
            )
            lasso.fit(self.X, self.y)

            # Extract model from pipeline and identify non-zero coefficient features
            model = lasso.named_steps['logisticregressioncv']
            logging.info("TESTING THIS AARON: ", self.X.columns)
            selected_features = self.X.columns[model.coef_.flatten() != 0].tolist()

            # Explicitly include 'who' and 'is_female' in the selected features if they are not already included
            required_features = ['who', 'is_female']  # Adjust according to actual column names
            for feature in required_features:
                if feature not in selected_features and feature in self.X.columns:
                    selected_features.append(feature)

            self.selected_features = selected_features
            logging.info(f"Selected features (including required): {self.selected_features}")

            # Split dataset into training and testing sets using selected features
            X_selected = self.X[self.selected_features]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_selected, self.y, test_size=0.25
            )

            # Fit logistic regression model on training data
            logging.info("Fitting logistic regression model with the selected features...")
            self.model = LogisticRegression(max_iter=10000)
            self.model.fit(self.X_train, self.y_train)
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during feature selection and model fitting: {e}")

    def find_best_threshold(self):
        """
        Sets the threshold based on the proportion of positive outcomes in the test set.
        """
        try:
            logging.info("Calculating the dynamic threshold based on the proportion of positive outcomes...")
            # Calculate the proportion of positive outcomes in the test set
            positive_proportion = self.y_test.mean()
            negative_proportion = 1 - positive_proportion

            # Log the ratio of 1s to 0s in the outcome variable
            logging.info(f"Ratio of 1s to 0s in the outcome variable: {positive_proportion:.2f} : {negative_proportion:.2f}")

            # Set the best threshold based on the proportion of positive outcomes
            self.best_threshold = positive_proportion
            
            logging.info(f"Dynamic threshold set to: {self.best_threshold}")
        except Exception as e:
            logging.error(f"An error occurred during threshold evaluation: {e}")

    def evaluate_model(self):
        """Evaluates the model using various metrics and logs the results."""
        try:
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)

            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            confusion = confusion_matrix(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)

            logging.info(f"ROC AUC Score: {roc_auc}")
            logging.info(f"Confusion Matrix: \n{confusion}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            return y_pred_proba
            # # Plot ROC Curve
            # fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            # plt.figure()
            # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC)')
            # plt.legend(loc="lower right")
            # plt.show()
        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}")

    #def get_predictions(self):
    """Return the predictions from the model"""
