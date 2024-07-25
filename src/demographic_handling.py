from create_demodf_knn import create_demographic_dfs
from merge_demodf import merge_demographic_data
import logging
import numpy as np

def create_and_merge_demographic_subsets(processed_data):
   
    try:
        logging.info("Creating demographic subsets from preprocessed data...")
        subsets = create_demographic_dfs(processed_data)
        logging.info(f"Created {len(subsets)} subsets successfully.")
        
        logging.info("Creating merged subsets from create_demographics_dfs function")
        merged_subsets = merge_demographic_data(subsets, processed_data)
        logging.info(f"Created {len(subsets)} subsets successfully.")
        
        return merged_subsets
    except Exception as e:
        logging.error(f"Error creating demographic subsets: {e}")
        raise
