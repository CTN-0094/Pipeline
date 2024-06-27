from src.train_model import LogisticModel
import logging

def train_and_evaluate_models(merged_subsets, selected_outcome):
    for i, subset in enumerate(merged_subsets):
        demographic_counts = subset['RaceEth'].value_counts().to_dict()
        demographic_str = ", ".join([f"{v} {k}" for k, v in demographic_counts.items()])
        logging.info(f"Subset {i + 1} demographic makeup: {demographic_str}")

        logging.info(f"Processing subset {i + 1}...")
        logging.info("-----------------------------")
        logging.info(f"TRAIN MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("-----------------------------")

        try:
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
            train_model.evaluate_model()
            logging.info(f"Model evaluated successfully for subset {i + 1}.")
        except Exception as e:
            logging.error(f"Error during model evaluation for subset {i + 1}: {e}")
            continue

        logging.info("------------------------------")
        logging.info(f"EVALUATE MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("------------------------------")
