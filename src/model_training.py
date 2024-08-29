from src.train_model import LogisticModel
import logging
import numpy as np
import csv
import os
from datetime import datetime


def train_and_evaluate_models(merged_subsets, seed, selected_outcome, directory):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    predictions = {}

    selectedModel = LogisticModel
    numOfSubsets = len(merged_subsets)

    for i, subset in enumerate(merged_subsets):
        # Log the demographic makeup of each subset
        demographic_counts = subset['RaceEth'].value_counts().to_dict()
        demographic_str = ", ".join([f"{v} {k}" for k, v in demographic_counts.items()])
        logging.info(f"Subset {i + 1} demographic makeup: {demographic_str}")
        logging.info(f"Processing subset {i + 1}...")

        logging.info("-----------------------------")
        logging.info(f"TRAIN MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("-----------------------------")

        outcomeModel = selectedModel(subset, selected_outcome)
        outcomeModel.train()
        
        logging.info(f"Model trained and saved successfully for subset {i + 1}.")

        logging.info("---------------------------")
        logging.info(f"TRAIN MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("---------------------------")

        logging.info("--------------------------------")
        logging.info(f"EVALUATE MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("--------------------------------")
        
        try:
            #Write result to dictionary of arrays
            prediction = outcomeModel.evaluate()
            for id, result in prediction:
                if id not in predictions:
                    predictions[id] = [None] * numOfSubsets
                predictions[id][i] = result

            logging.info(f"Model evaluated successfully for subset {i + 1}.")
        except Exception as e:
            logging.error(f"Error during model evaluation for subset {i + 1}: {e}")
            return

        logging.info("------------------------------")
        logging.info(f"EVALUATE MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("------------------------------")
        
    save_predictions_to_csv(predictions, seed, selected_outcome, directory)
    logging.info(f"Model predictions saved to csv successfully.")
    


def save_predictions_to_csv(predictions, seed, selected_outcome, directory):
    # Define the profiling log file path
    directory = os.path.join(directory, "predictions")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"{selected_outcome}_{seed}_{timestamp}.csv")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

         # Write header
        header = ['who'] + [f'Subset_{i+1}' for i in range(10)]
        writer.writerow(header)
        
        # Write data rows
        for id, trials_data in predictions.items():
            writer.writerow([id] + trials_data)



        
