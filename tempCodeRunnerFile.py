import sys
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import log_pipeline_completion, get_outcome_choice
from data_loading import load_datasets
from data_preprocessing import preprocess_merged_data
from demographic_handling import create_and_merge_demographic_subsets
from model_training import train_and_evaluate_models
from logScraper import scrape_log_to_csv  # Import the log scraper function

LOG_DIR = "logs"  # Directory to store log files

def setup_logging(seed):
    """Set up logging for the pipeline."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)  # Ensure the log directory exists

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(LOG_DIR, f"pipeline_{timestamp}_{seed}.log")

    # Configure the logging for each seed run
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def run_pipeline(seed, selected_outcome):
    """Run the pipeline using a specific seed."""
    log_filepath = setup_logging(seed)

    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Log the seed value
    logging.info(f"Global Seed set to: {seed}")

    # Log the selected outcome
    logging.info(f"Outcome Name: {selected_outcome}")

    master_path = 'data/master_data.csv'
    outcomes_path = 'data/all_binary_selected_outcomes.csv'
    
    master_df, outcomes_df = load_datasets(master_path, outcomes_path)
    
    outcome_column = outcomes_df[['who', selected_outcome]]
    merged_df = pd.merge(master_df, outcome_column, on='who', how='inner')
    
    processed_data = preprocess_merged_data(merged_df, selected_outcome)
    
    merged_subsets = create_and_merge_demographic_subsets(processed_data, seed)
    
    train_and_evaluate_models(merged_subsets, selected_outcome, seed)
    
    log_pipeline_completion()

    return log_filepath

def main():
    # Define a list of seeds to iterate over
    seed_list = [4, 23, 4]  # Example seeds; replace with your desired values

    # Get the selected outcome
    selected_outcome = get_outcome_choice()

    # Store the paths of the generated log files
    log_filepaths = []

    # Loop through each seed and run the pipeline
    for seed in seed_list:
        log_filepath = run_pipeline(seed, selected_outcome)
        log_filepaths.append(log_filepath)

    # Scrape the logs after all pipelines are run
    scrape_log_to_csv(log_filepaths)

if __name__ == "__main__":
    main()
