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

    # Initialize a preprocessor object for data transformation tasks
    preprocessor = DataPreprocessor(subset)
    
    # Logging to mark the start of the pre-processing phase
    logging.info("---------------------------------")
    logging.info("PRE-PROCESSING STAGE STARTING...")
    logging.info("---------------------------------")

    # Specify columns to drop initially
    columns_to_drop_1 = [
        'pain_when', 'is_smoker', 'per_day', 'max', 'amount', 'depression', 'anxiety',
        'schizophrenia', "max", "amount", "cocaine_inject_days", "speedball_inject_days",
        "opioid_inject_days", "speed_inject_days", "UDS_Alcohol_Count", 'UDS_Mdma/Hallucinogen_Count'
    ]
    
    # Try to drop the specified columns
    try:
        preprocessor.drop_columns_and_return(columns_to_drop_1)
        logging.info(f"Dropped initial columns: {columns_to_drop_1}")
    except Exception as e:
        logging.error(f"Error dropping initial columns: {e}")

    # Convert Yes/No values to binary (1/0)
    try:
        preprocessor.convert_yes_no_to_binary()
        logging.info("Converted Yes/No to binary.")
    except Exception as e:
        logging.error(f"Error converting Yes/No to binary: {e}")

    # Specify columns for TLFB (Timeline Follow-Back) counts to process
    specified_tlfb_columns = [
        'TLFB_Alcohol_Count', 'TLFB_Amphetamine_Count', 'TLFB_Cocaine_Count',
        'TLFB_Heroin_Count', 'TLFB_Benzodiazepine_Count', 'TLFB_Opioid_Count',
        'TLFB_THC_Count', 'TLFB_Methadone_Count', 'TLFB_Buprenorphine_Count'
    ]
    
    # Process TLFB columns to transform their data
    try:
        preprocessor.process_tlfb_columns(specified_tlfb_columns)
        logging.info(f"Processed TLFB columns with specified columns: {specified_tlfb_columns}")
    except Exception as e:
        logging.error(f"Error processing TLFB columns: {e}")

    # Calculate behavioral columns
    try:
        preprocessor.calculate_behavioral_columns()
        logging.info("Calculated behavioral columns.")
    except Exception as e:
        logging.error(f"Error calculating behavioral columns: {e}")

    # Move the selected outcome column to the end of the DataFrame
    try:
        preprocessor.move_column_to_end(selected_outcome)
        logging.info(f"Moved '{selected_outcome}' column to the end.")
    except Exception as e:
        logging.error(f"Error moving '{selected_outcome}' column to the end: {e}")


# Specify columns to drop initially
    columns_to_drop_2 = ['msm_npt', 'msm_frq', 'txx_prt']
    
    # Try to drop the specified columns
    try:
        preprocessor.drop_columns_and_return(columns_to_drop_2)
        logging.info(f"Dropped initial columns: {columns_to_drop_2}")
    except Exception as e:
        logging.error(f"Error dropping initial columns: {e}")

    # transform_data_with_nan_handling
    try:
        preprocessor.transform_data_with_nan_handling()
        logging.info(f"transform_data_with_nan_handling success")
    except Exception as e:
        logging.error(f"Error dropping initial columns: {e}")

    # Rename columns according to a predefined mapping
    try:
        preprocessor.rename_columns()
        logging.info("Renamed columns according to the mapping.")
    except Exception as e:
        logging.error(f"Error renaming columns: {e}")

    # Transform NaN (missing values) to zeros for binary columns
    try:
        preprocessor.transform_nan_to_zero_for_binary_columns()
        logging.info("Transformed NaN to 0 for binary columns.")
    except Exception as e:
        logging.error(f"Error transforming NaN to 0 for binary columns: {e}")

    # Transform and rename 'heroin_inject_days' to 'rbsivheroin'
    try:
        preprocessor.transform_and_rename_column('heroin_inject_days', 'rbsivheroin')
        logging.info("Transformed and renamed 'heroin_inject_days' column to 'rbsivheroin'.")
    except Exception as e:
        logging.error(f"Error transforming and renaming 'heroin_inject_days' column: {e}")

    # Fill missing values (NaN) with zeros for the 'ftnd' column
    try:
        preprocessor.fill_nan_with_zero('ftnd')
        logging.info("Filled NaN with 0 for 'ftnd' column.")
    except Exception as e:
        logging.error(f"Error filling NaN with 0 for 'ftnd': {e}")

    # Specify additional columns to drop
    columns_to_drop_2 = [
        'rbs_iv_days', 'race', 'RBS_cocaine_Days', 'RBS_heroin_Days',
        'RBS_opioid_Days', 'RBS_speed_Days', 'RBS_speedball_Days'
    ]
    
    # Try to drop additional columns
    try:
        preprocessor.drop_columns_and_return(columns_to_drop_2)
        logging.info(f"Dropped additional columns: {columns_to_drop_2}")
    except Exception as e:
        logging.error(f"Error dropping additional columns: {e}")

    # Convert drug count columns to binary
    try:
        preprocessor.convert_uds_to_binary()
        logging.info("Converted uds drug counts to binary.")
    except Exception as e:
        logging.error(f"Error converting uds drug counts to binary: {e}")


    # After all processing, save the final processed dataframe
    processed_data = preprocessor.dataframe
    print("Final Processed DataFrame:")
    print(processed_data.head())

    # Log the end of the pre-processing phase
    logging.info("------------------------------")
    logging.info("PRE-PROCESSING STAGE COMPLETED")
    logging.info("------------------------------")

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