from src.preprocess_pipeline import preprocess_data
import logging

def preprocess_merged_data(merged_df, selected_outcome):
    try:
        logging.info("Starting preprocessing of the merged dataset...")
        processed_data = preprocess_data(merged_df, selected_outcome)
        logging.info("Merged dataset preprocessed successfully.")
        return processed_data
    except Exception as e:
        logging.error(f"Error during preprocessing of the merged dataset: {e}")
        raise
