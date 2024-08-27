from src.train_model import LogisticModel
import logging
import numpy as np
import csv
import os
from datetime import datetime
import statsmodels.api as sm

def train_and_evaluate_models(merged_subsets, selected_outcome, seed):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    predictions = []

    outcomeModel = logisticOutcome()

    for i, subset in enumerate(merged_subsets):
        # Log the demographic makeup of each subset
        demographic_counts = subset['RaceEth'].value_counts().to_dict()
        demographic_str = ", ".join([f"{v} {k}" for k, v in demographic_counts.items()])
        logging.info(f"Subset {i + 1} demographic makeup: {demographic_str}")
        logging.info(f"Processing subset {i + 1}...")

        prediction = outcomeModel.trainAndEvaluate(subset, selected_outcome, i)
        predictions.append(prediction)
    
    predictions = [outcomeModel.train_model.X_test['who']] + predictions
    save_predictions_to_csv(predictions, selected_outcome, seed)
    logging.info(f"Model predictions saved to csv successfully.")
    


def save_predictions_to_csv(predictions, selected_outcome, seed):
    # Define the profiling log file path
    log_dir = "Predictions"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(log_dir, f"predictions_{selected_outcome}_{seed}_{timestamp}.csv")
    # Transpose the list of lists to convert rows to columns
    transposed_predictions = list(zip(*predictions))
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(transposed_predictions)



class outcomeModel():
    def trainAndEvaluate(self, subset, selected_outcome, i):
        logging.info("-----------------------------")
        logging.info(f"TRAIN MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("-----------------------------")
        #try:
        # Initialize and train the logistic model
        self.train(subset, selected_outcome)
        logging.info(f"Model trained and saved successfully for subset {i + 1}.")
        #except Exception as e:
        #    logging.error(f"Error during model training for subset {i + 1}: {e}")
        #    return

        logging.info("---------------------------")
        logging.info(f"TRAIN MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("---------------------------")

        logging.info("--------------------------------")
        logging.info(f"EVALUATE MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("--------------------------------")
        
        try:
            # Evaluate the model
            prediction = self.predict()
            logging.info(f"Model evaluated successfully for subset {i + 1}.")
        except Exception as e:
            logging.error(f"Error during model evaluation for subset {i + 1}: {e}")
            return

        logging.info("------------------------------")
        logging.info(f"EVALUATE MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("------------------------------")
        return prediction

    def train(self, subset, selected_outcome):
        pass

    def predict(self):
        pass



class logisticOutcome(outcomeModel):
    def __init__(self):
        self.train_model = None
    def train(self, subset, selected_outcome):
        self.train_model = LogisticModel(subset, selected_outcome)
        self.train_model.feature_selection_and_model_fitting()
        self.train_model.find_best_threshold()

    def predict(self):
        return self.train_model.evaluate_model()
    


class negativeBinomialOutcome(outcomeModel):
    def __init__(self):
        self.train_model = None
    def train(self, subset, selected_outcome):
        endog = [selected_outcome]
        exog = sm.add_constant(subset)
        self.train_model =model = sm.NegativeBinomial(endog, exog, loglike_method='nb2')
        self.train_model.fit()
        logging.info("AARON DEBUG: ", self.train_model.summary())

    def predict(self):
        return self.train_model.evaluate_model()
        