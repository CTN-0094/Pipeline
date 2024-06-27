import pandas as pd
import logging

def load_datasets(master_path, outcomes_path):
    try:
        master_df = pd.read_csv(master_path)
        outcomes_df = pd.read_csv(outcomes_path)
        logging.info("Datasets loaded successfully.")
        return master_df, outcomes_df
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        raise
