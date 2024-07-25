import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import setup_logging, log_pipeline_completion, get_outcome_choice
from data_loading import load_datasets
from data_preprocessing import preprocess_merged_data
from demographic_handling import create_and_merge_demographic_subsets
from model_training import train_and_evaluate_models
import pandas as pd

# Import the log scraper
from logScraper import main as scrape_log

def main():
    setup_logging()
    selected_outcome = get_outcome_choice()
    
    master_path = 'data/master_data.csv'
    outcomes_path = 'data/all_binary_selected_outcomes.csv'
    
    master_df, outcomes_df = load_datasets(master_path, outcomes_path)
    
    outcome_column = outcomes_df[['who', selected_outcome]]
    merged_df = pd.merge(master_df, outcome_column, on='who', how='inner')
    
    processed_data = preprocess_merged_data(merged_df, selected_outcome)
    
    merged_subsets = create_and_merge_demographic_subsets(processed_data)
    
    train_and_evaluate_models(merged_subsets, selected_outcome)
    
    log_pipeline_completion()

    # Run the log scraper after the pipeline completes
    scrape_log()

if __name__ == "__main__":
    main()