from src.train_model import LogisticModel
import logging
import numpy as np
import csv
import os
from datetime import datetime

def train_and_evaluate_models(merged_subsets, selected_outcome, seed):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    predictions = []
    for i, subset in enumerate(merged_subsets):
        # Log the demographic makeup of each subset
        demographic_counts = subset['RaceEth'].value_counts().to_dict()
        demographic_str = ", ".join([f"{v} {k}" for k, v in demographic_counts.items()])
        logging.info(f"Subset {i + 1} demographic makeup: {demographic_str}")
        logging.info(f"Processing subset {i + 1}...")
        logging.info("-----------------------------")
        logging.info(f"TRAIN MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("-----------------------------")

        try:
            # Initialize and train the logistic model
            train_model = LogisticModel(subset, selected_outcome)
            train_model.feature_selection_and_model_fitting()
            train_model.find_best_threshold()
            logging.info(f"Model trained and saved successfully for subset {i + 1}.")
        except Exception as e:
            logging.error(f"Error during model training for subset {i + 1}: {e}")
            continue

        logging.info("---------------------------")
        logging.info(f"TRAIN MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("---------------------------")

        logging.info("--------------------------------")
        logging.info(f"EVALUATE MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("--------------------------------")
        
        try:
            # Evaluate the model
            predictions.append(train_model.evaluate_model())
            logging.info(f"Model evaluated successfully for subset {i + 1}.")
        except Exception as e:
            logging.error(f"Error during model evaluation for subset {i + 1}: {e}")
            continue

        logging.info("------------------------------")
        logging.info(f"EVALUATE MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("------------------------------")
    
    #try:
        # Evaluate the model
    predictions = [train_model.X_test['who']] + predictions
    save_predictions_to_csv(predictions, selected_outcome, seed)
    logging.info(f"Model predictions saved to csv successfully.")
    #except Exception as e:
        #logging.error(f"Error during model predictions saving")
    


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