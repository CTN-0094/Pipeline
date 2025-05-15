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
from src.silent_logging import add_silent_handler
from src.logging_setup import setup_logging  # Import the logging setup from logging_setup.py
import argparse

# Add the 'src' directory to the system path to allow imports from that directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary functions from the 'src' directory
from utils import log_pipeline_completion, get_outcome_choice
from data_loading import load_datasets
from data_preprocessing import preprocess_merged_data
from create_demodf_knn import create_demographic_dfs
from model_training import train_and_evaluate_models
from logScraper import scrape_log_to_csv  # Import the log scraper function

#LOG_DIR = "logs"  # Directory to store log files


def run_pipeline(seed, selected_outcome, directory, split_col="RaceEth", sample_size=500):
    """Run the pipeline using a specific seed and selected outcome."""
    # Set up logging and get the path to the log file
    log_filepath = setup_logging(seed, selected_outcome, directory, quiet=False)

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
    # binary disparity indicator and subject ID; the minority group should be coded as 1, 0 otherwise
    bdi_path = 'data/inpatient_care.csv'
    
    # Load the datasets
    master_df, outcomes_df, bdi_df = load_datasets(master_path, outcomes_path, bdi_path)
    
    # Extract the outcome column from the outcomes dataframe and merge it with the master dataframe
    outcome_column = outcomes_df[['who', selected_outcome]]
    merged_df = pd.merge(master_df, outcome_column, on='who', how='inner')
    # Add the binary disparity indicator
    merged_df = pd.merge(merged_df, bdi_df, on='who', how='left')
    
    # Preprocess the merged data based on the selected outcome
    processed_data = preprocess_merged_data(merged_df, selected_outcome)
    
    # Create and merge demographic subsets
    # split_col is a string with the name of the column in bdi_df
    merged_subsets = create_demographic_dfs(processed_data, columnToSplit=split_col, sampleSize=sample_size)

    # import pdb; pdb.set_trace()
    
    # Train and evaluate the models using the merged subsets
    train_and_evaluate_models(merged_subsets, seed, selected_outcome, directory)
    
    # Log the completion of the pipeline
    log_pipeline_completion()

    # Return the path to the log file for further processing
    return log_filepath

def argument_handler():

    # Create the parser
    parser = argparse.ArgumentParser(description='Pipeline for statistical modeling and machine learning on the CTN-0094 database')

    # Add arguments loop (min and max seed, prompt otherwise) target directory, profile
    parser.add_argument('-l', '--loop', type=int, nargs='+', help='minimum and maximum seed', default = None)
    parser.add_argument('-o', '--outcome', '--outcomes', type=str, nargs='+', help='all outcomes to run', default = None)
    parser.add_argument('-d', '--dir', '--directory', type=str, help='directory to save logs, predictions, and evaluations', default="")
    parser.add_argument('-p', '--prof', '--profile', type=str, help='type of profiling to run (\'simple\' or \'complex\')', default="None")
    parser.add_argument('-s', '--split-col', type=str, help='column to split on (e.g., "RaceEth" or "is_inpatient" from the bdi_df data)', default='RaceEth')
    parser.add_argument('-N', '--sample-size', type=int, help='Number of minority samples per trial', default=500)


    # Parse the arguments
    args = parser.parse_args()

    return args.loop, args.outcome, args.dir, args.prof, args.split_col, args.sample_size

def main():

    available_outcomes = [
        'ctn0094_relapse_event', 'Ab_krupitskyA_2011', 'Ab_ling_1998',
        'Rs_johnson_1992', 'Rs_krupitsky_2004', 'Rd_kostenB_1993'
    ]

    seedRange, outcomes, directory, profile, split_col, sample_size = argument_handler()

    if seedRange is not None:
        # Define a list of seeds to iterate over
        seed_list = list(range(min(seedRange), max(seedRange)))
    else:
        seed_list = [0]
        # Get the selected outcome from the user or another source
        outcomes = [get_outcome_choice(available_outcomes)]

    if outcomes == None:
        outcomes = available_outcomes

    # Store the paths of the generated log files
    log_filepaths = []

    # Loop through each seed and run the pipeline
    for seed in seed_list:
        for outcome in outcomes:
            if profile == 'simple' or profile == None:
                pf.simple_profile_pipeline(run_pipeline, seed, outcome, directory, split_col)
            elif profile == 'complex':
                pf.profile_pipeline(run_pipeline, seed, outcome, directory, split_col)
            else:
                log_filepath = run_pipeline(seed, outcome, directory, split_col)
                log_filepaths.append(log_filepath)  # Store the log file path for later use

    # Scrape the logs after all pipelines have been run
    scrape_log_to_csv(log_filepaths, directory)

if __name__ == "__main__":
    main()
    #pf.profileAllOutcomes(run_pipeline)
    
