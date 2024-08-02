import sys
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import setup_logging, log_pipeline_completion, get_outcome_choice
from data_loading import load_datasets
from data_preprocessing import preprocess_merged_data
from demographic_handling import create_and_merge_demographic_subsets
from model_training import train_and_evaluate_models
from logScraper import scrape_log_to_csv  # Import the log scraper function

import logging

def run_pipeline_with_seed(seed, selected_outcome, pipeline_number):
    """Run the pipeline using a specific seed."""
    log_filepath = setup_logging()

    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Log the seed and pipeline number
    logging.info(f"Pipeline Number: {pipeline_number}")
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

    # Scrape the log file and write to CSV
    scrape_log_to_csv(pipeline_number)

def main():
    # Define a list of seeds to iterate over
    seed_list = [42, 1, 42]  # Example seeds; replace with your desired values

    # Get the selected outcome
    selected_outcome = get_outcome_choice()

    # Initialize pipeline number
    pipeline_number = 1

    # Loop through each seed and run the pipeline
    for seed in seed_list:
        run_pipeline_with_seed(seed, selected_outcome, pipeline_number)
        pipeline_number += 1  # Increment the pipeline number after each run

if __name__ == "__main__":
    main()