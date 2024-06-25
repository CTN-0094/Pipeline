# Import necessary libraries
import logging  # Used for logging information and errors
import os  # For file and directory operations
from datetime import datetime  # For handling date and time operations
from joblib import load  # For loading joblib-compressed files
import pandas as pd  # For data manipulation using DataFrames

# Import custom modules from the src directory
from src.create_demodf_knn import create_demographic_dfs  # Subsetting data based on demographic criteria
from src.preprocess import DataPreprocessor  # Preprocessing utility for data preparation
from src.utils import (
    setup_logging,  # Utility function to set up logging
    get_outcome_choice  # Utility function to select the desired outcome variable
)
from src.merge_demodf import merge_demographic_data  # Function to merge demographic subsets
from src.train_model import LogisticModel  # Assuming the class is saved under src/logistic_model.py
from src.preprocess_pipeline import preprocess_data  # Importing the new preprocessing function

pd.set_option('display.max_columns', None)

# Main function where the pipeline execution occurs
def main():
    # Initialize logging configuration
    setup_logging()
    
    # Obtain the chosen outcome variable from user input or pre-defined sources
    selected_outcome = get_outcome_choice()

    # Load the master and outcomes datasets
    try:
        # Load the master dataset containing all features and observations
        master_df = pd.read_csv('/Users/richeyjay/Desktop/Relapse_Pipeline/env/data/master_data.csv')
        
        # Load the dataset containing selected binary outcomes
        outcomes_df = pd.read_csv('/Users/richeyjay/Desktop/Relapse_Pipeline/env/data/all_binary_selected_outcomes.csv')
        
        logging.info("Datasets loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return

    # Extract the selected outcome column for analysis
    try:
        # Isolate the 'who' identifier and the specific outcome column selected
        outcome_column = outcomes_df[['who', selected_outcome]]
        
        logging.info(f"Selected outcome column '{selected_outcome}' successfully.")
    except Exception as e:
        logging.error(f"Error selecting outcome column: {e}")
        return

    # Merge the master dataset with the chosen outcome column to form a complete analysis dataset
    try:
        logging.info("Starting merge of dataset and selected outcome...")

        # Perform an inner join on the 'who' identifier to get records with matching observations
        merged_df = pd.merge(master_df, outcome_column, on='who', how='inner')
        
        logging.info("Dataset merged successfully with selected outcome.")
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        return
    
    # Preprocess the entire merged dataset
    try:
        logging.info("Starting preprocessing of the merged dataset...")
        processed_data = preprocess_data(merged_df, selected_outcome)
        logging.info("Merged dataset preprocessed successfully.")
    except Exception as e:
        logging.error(f"Error during preprocessing of the merged dataset: {e}")
        return

    #---------------------------------------------------------------------------   
    # Create demographic subsets from the preprocessed data
    try:
        logging.info("Creating demographic subsets from preprocessed data...")
        subsets = create_demographic_dfs(processed_data)
        logging.info(f"Created {len(subsets)} subsets successfully.")
    except Exception as e:
        logging.error(f"Error creating demographic subsets: {e}")
        return
    
    # Create merged subsets from the create_demographics_dfs function

    try:
        logging.info("Creating merged subsets from create_demographics_dfs function")
        merged_subsets = merge_demographic_data(subsets, processed_data)
        logging.info(f"Created {len(subsets)} subsets successfully.")
    except Exception as e:
        logging.error(f"Error creating demographic subsets: {e}")
        return

    # Loop over each subset
    for i, subset in enumerate(merged_subsets):
        # Logging the demographic makeup of the subset
        demographic_counts = subset['RaceEth'].value_counts().to_dict()
        demographic_str = ", ".join([f"{v} {k}" for k, v in demographic_counts.items()])
        logging.info(f"Subset {i + 1} demographic makeup: {demographic_str}")

        logging.info(f"Processing subset {i + 1}...")
        
        logging.info("-----------------------------")
        logging.info(f"TRAIN MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("-----------------------------")

        # Initialize and train the logistic model
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
    
    # Log the completion of the pipeline
    logging.info("------------------")
    logging.info("PIPELINE COMPLETED")
    logging.info("------------------")

# Execute the main function if this script is the entry point
if __name__ == "__main__":
    main()

# Figure out the most optimal way to loop pipeline
# The loop should be for N= 0, Sample Size = 1000, NHW = 500, Minority = 500
# Next loop is 550 NHW and 450 Minority, but this is where the subset function comes in
# The subset function matches the 50 NHW participants that get added on with the similar
# Demographic features as the minority

# Set threshold at 0.5  calculate at training data probability,
