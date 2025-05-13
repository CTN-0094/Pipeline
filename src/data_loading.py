import pandas as pd
import logging

def load_datasets(*paths):
    """Load any number of CSV files and return a list of DataFrames."""
    dataframes = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Error loading datasets: {e}")
            raise
    logging.info("Datasets loaded successfully.")
    return dataframes

