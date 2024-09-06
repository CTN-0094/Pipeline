import sys
import os
import random
import numpy as np
import pandas as pd
import cProfile
import pstats
from datetime import datetime
import logging
import io
import src.profiling as pf

# Add the 'src' directory to the system path to allow imports from that directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary functions from the 'src' directory
from utils import log_pipeline_completion, get_outcome_choice
from data_loading import load_datasets
from data_preprocessing import preprocess_merged_data
from demographic_handling import create_and_merge_demographic_subsets
from model_training import train_and_evaluate_models
from logScraper import scrape_log_to_csv  # Import the log scraper function

LOG_DIR = "logs"  # Directory to store log files

def setup_logging(seed):
    """Set up logging for the pipeline, creating a log file specific to each seed."""
    # Ensure the log directory exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Generate a log filename with a timestamp and seed value
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(LOG_DIR, f"pipeline_{timestamp}_{seed}.log")

    # Configure the logging for the pipeline
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers to avoid duplicate logs

    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log message format
        handlers=[
            logging.FileHandler(log_filename),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )
    return log_filename

def run_pipeline(seed, selected_outcome):
    """Run the pipeline using a specific seed and selected outcome."""
    # Set up logging and get the path to the log file
    log_filepath = setup_logging(seed)

    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Log the seed value for tracking
    logging.info(f"Global Seed set to: {seed}")

    # Log the selected outcome for reference
    logging.info(f"Outcome Name: {selected_outcome}")

    # Paths to the data files
    master_path = 'data/master_data.csv'
    outcomes_path = 'data/all_binary_selected_outcomes.csv'
    
    # Load the datasets
    master_df, outcomes_df = load_datasets(master_path, outcomes_path)
    
    # Extract the outcome column from the outcomes dataframe and merge it with the master dataframe
    outcome_column = outcomes_df[['who', selected_outcome]]
    merged_df = pd.merge(master_df, outcome_column, on='who', how='inner')
    
    # Preprocess the merged data based on the selected outcome
    processed_data = preprocess_merged_data(merged_df, selected_outcome)
    
    # Create and merge demographic subsets using the seed
    merged_subsets = create_and_merge_demographic_subsets(processed_data, seed)
    
    # Train and evaluate the models using the merged subsets
    train_and_evaluate_models(merged_subsets, selected_outcome, seed)
    
    # Log the completion of the pipeline
    log_pipeline_completion()

    # Return the path to the log file for further processing
    return log_filepath

def main():
    # Define a list of seeds to iterate over
    seed_list = [2]  # Example seeds; replace with your desired values

    # Get the selected outcome from the user or another source
    selected_outcome = get_outcome_choice()

    # Store the paths of the generated log files
    log_filepaths = []

    # Loop through each seed and run the pipeline
    for seed in seed_list:
        log_filepath = run_pipeline(seed, selected_outcome)
        log_filepaths.append(log_filepath)  # Store the log file path for later use

    # Scrape the logs after all pipelines have been run
    scrape_log_to_csv(log_filepaths)

if __name__ == "__main__":
    main()
    pf.profileAllOutcomes(run_pipeline)
    
