import logging  # Provides functionalities for logging information and errors
import os  # For handling file and directory operations
from datetime import datetime  # To get the current date and time
import pandas as pd  # DataFrame operations with pandas
from joblib import load  # For loading and saving compressed files via joblib
# from create_subset import create_demographic_subsets  # Custom module for demographic data subsetting

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'logs/pipeline_{timestamp}.log'
    logger = logging.getLogger()
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.info("-------------------------------------------")
    logging.info("Logging setup complete - starting pipeline.")
    logging.info("-------------------------------------------")

def log_pipeline_completion():
    logging.info("------------------")
    logging.info("PIPELINE COMPLETED")
    logging.info("------------------")

def get_outcome_choice():
    available_outcomes = [
        'ctn0094_relapse_event', 'Ab_krupitskyA_2011', 'Ab_ling_1998',
        'Rs_johnson_1992', 'Rs_krupitsky_2004', 'Rd_kostenB_1993'
    ]
    while True:
        print("Available outcomes:")
        for i, outcome in enumerate(available_outcomes, 1):
            print(f"{i}. {outcome}")
        try:
            choice = int(input("Select an outcome by entering its number: "))
            if 1 <= choice <= len(available_outcomes):
                selected_outcome = available_outcomes[choice - 1]
                
                return selected_outcome
            else:
                print(f"Please enter a number between 1 and {len(available_outcomes)}.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def get_demographic_inputs():
    while True:
        print("Make sure you are using the correct spelling and column name of your demographic feature.")
        demographic_feature = input("Enter demographic feature (e.g., race): ")
        if demographic_feature.strip():
            break
        else:
            print("Demographic feature cannot be empty. Please enter a valid demographic feature.")
    while True:
        races_input = input("Enter races (comma-separated, e.g., NHW, NHB, Hisp): ")
        races = [race.strip() for race in races_input.split(',') if race.strip()]
        if races:
            break
        else:
            print("Races cannot be empty. Please enter valid races separated by commas.")
    while True:
        sample_sizes_input = input("Enter sample sizes corresponding to races (comma-separated, e.g., 500, 300, 200): ")
        try:
            sample_sizes = [int(size.strip()) for size in sample_sizes_input.split(',') if size.strip()]
            if len(sample_sizes) == len(races):
                break
            else:
                print("The number of sample sizes must match the number of races. Please enter equal numbers of sample sizes and races.")
        except ValueError:
            print("Invalid number provided. Please enter valid integers separated by commas.")
    logging.info(f"Selected demographic feature: '{demographic_feature}'")
    logging.info(f"Races: {', '.join(races)}")
    logging.info(f"Sample sizes: {', '.join(map(str, sample_sizes))}")
    return {
        'demographic_feature': demographic_feature,
        'races': races,
        'sample_sizes': sample_sizes
    }

# def create_demographic_subset(merged_df, demo_inputs):
#     try:
#         subset = create_demographic_subsets(
#             data=merged_df,
#             demographic_feature=demo_inputs['demographic_feature'],
#             categories=demo_inputs['races'],
#             sample_sizes=demo_inputs['sample_sizes'],
#             exclude_hispanic=True
#         )
#         logging.info(f"Created demographic subset with feature '{demo_inputs['demographic_feature']}'.")
#     except Exception as e:
#         logging.error(f"Error creating demographic subset: {e}")
#         return None
#     return subset