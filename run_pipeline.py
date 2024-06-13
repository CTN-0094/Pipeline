# Import necessary libraries
import logging  # Used for logging information and errors
import os  # For file and directory operations
from datetime import datetime  # For handling date and time operations
from joblib import load  # For loading joblib-compressed files

# Import custom modules from the src directory
from src.create_subset import create_demographic_subsets  # Subsetting data based on demographic criteria
from src.preprocess import DataPreprocessor  # Preprocessing utility for data preparation
from src.utils import (
    setup_logging,  # Utility function to set up logging
    get_outcome_choice,  # Utility function to select the desired outcome variable
    get_demographic_inputs,  # Utility to collect demographic criteria
    create_demographic_subset  # Function to generate demographic-based subsets
)
import pandas as pd  # For data manipulation using DataFrames
from src.train_model import LogisticModel  # Assuming the class is saved under src/logistic_model.py
from src.preprocess_pipeline import preprocess_data  # Importing the new preprocessing function


pd.set_option('display.max_columns', None)

# Set the option to display all columns (None means no limit to the number of columns displayed)
# Main function where the pipeline execution occurs
def main():
    # Initialize logging configuration
    setup_logging()
    
    # Obtain demographic inputs and the chosen outcome variable from user input or pre-defined sources
    demo_inputs = get_demographic_inputs()
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
    
    # Debug print the merged dataset
    #print(merged_df)

    # Create a demographic subset using the provided demographic inputs
    subset = create_demographic_subset(merged_df, demo_inputs)
    
    # Debug print the subset
    #print(subset)
    #Start for loop here for wrapper, for j in the number of datasets
    #Subset will turn into a list of dataframes  

    # Preprocess the subset
    """
    PREPROCESSING STAGE START
    """
    processed_data = preprocess_data(subset, selected_outcome)
    """
    PREPROCESSING STAGE END
    """

    logging.info("-----------------------------")
    logging.info("TRAIN MODEL STAGE STARTING...")
    logging.info("-----------------------------")

    # Initialize and train the logistic model
    try:
        train_model = LogisticModel(processed_data, selected_outcome)
        train_model.feature_selection_and_model_fitting()
        train_model.find_best_threshold()
        logging.info("Model trained and saved successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")

    
    logging.info("---------------------------")
    logging.info("TRAIN MODEL STAGE COMPLETED")
    logging.info("---------------------------")

    
    logging.info("--------------------------------")
    logging.info("EVALUATE MODEL STAGE STARTING...")
    logging.info("--------------------------------")
    
    try:    
        train_model.evaluate_model()
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
    
    logging.info("------------------------------")
    logging.info("EVALUATE MODEL STAGE COMPLETED")
    logging.info("------------------------------")
    

    # Log the completion of the pipeline

    logging.info("------------------")
    logging.info("PIPELINE COMPLETED")
    logging.info("------------------")

# Execute the main function if this script is the entry point
if __name__ == "__main__":
    main()

#Figure out the most optimal way to loop pipeline
#The loop should be for N= 0, Sample Size = 1000, NHW = 500, Minority = 500
#Next loop is 550 NHW and 450 Minority, but this is where the subset function comes in
#The subset function matches the 50 NHW participants that get added on with the similar
#Demographic features as the minority

#Set thresshold at 0.5  calcuate at training data probability, 