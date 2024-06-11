# Import necessary modules
import logging  # Provides functionalities for logging information and errors
import os  # For handling file and directory operations
from datetime import datetime  # To get the current date and time
import pandas as pd  # DataFrame operations with pandas
from joblib import load  # For loading and saving compressed files via joblib
from src.create_subset import create_demographic_subsets  # Custom module for demographic data subsetting

# Function to set up logging with a custom format and save it to a file
def setup_logging():
    # Create a directory named 'logs' if it doesn't already exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file name
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Define the name of the log file with the current timestamp
    log_filename = f'logs/pipeline_{timestamp}.log'
    
    # Get a root logger to manage logging
    logger = logging.getLogger()
    
    # Set up a file handler that writes logs to the specified log file
    handler = logging.FileHandler(log_filename)
    
    # Specify the format for log messages: time, log level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    
    # Assign the formatter to the file handler
    handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(handler)
    
    # Set the logging level to INFO to capture general information and above
    logger.setLevel(logging.INFO)
    
    # Log a message indicating the logging setup is complete
    logging.info("-------------------------------------------")
    logging.info("Logging setup complete - starting pipeline.")
    logging.info("-------------------------------------------")

# Function to prompt the user to select an outcome from a predefined list
def get_outcome_choice():
    """
    Prompts the user to select an outcome from a predefined list of options.
    Repeatedly prompts until a valid choice is made and returns the selected outcome.
    """
    # List of possible outcomes to choose from
    available_outcomes = [
        'ctn0094_relapse_event', 'Ab_krupitskyA_2011', 'Ab_ling_1998',
        'Rs_johnson_1992', 'Rs_krupitsky_2004', 'Rd_kostenB_1993'
    ]

    # Start an infinite loop to prompt the user until a valid input is received
    while True:
        print("Available outcomes:")
        # Enumerate and print each outcome option with a corresponding number
        for i, outcome in enumerate(available_outcomes, 1):
            print(f"{i}. {outcome}")

        try:
            # Prompt the user to enter the number of their chosen outcome
            choice = int(input("Select an outcome by entering its number: "))

            # Check if the input number is within the valid range
            if 1 <= choice <= len(available_outcomes):
                # Return the selected outcome (adjusted for zero-based index)
                return available_outcomes[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(available_outcomes)}.")
        except ValueError:
            # Error message for non-numeric input
            print("Invalid input. Please enter a numeric value.")

# Function to collect demographic criteria from user input
def get_demographic_inputs():
    """
    Prompts the user to input a demographic feature, specific races, and sample sizes.
    Returns a dictionary with these values.
    """
    # Prompt the user to input the demographic feature
    while True:
        print("Make sure you are using the correct spelling and column name of your demographic feature.")

        # Input the demographic feature, e.g., 'race'
        demographic_feature = input("Enter demographic feature (e.g., race): ")

        # Ensure that the input is not empty or just whitespace
        if demographic_feature.strip():
            break
        else:
            print("Demographic feature cannot be empty. Please enter a valid demographic feature.")

    # Prompt for races as a comma-separated list
    while True:
        races_input = input("Enter races (comma-separated, e.g., NHW, NHB, Hisp): ")
        races = [race.strip() for race in races_input.split(',') if race.strip()]  # Split and clean
        if races:
            break
        else:
            print("Races cannot be empty. Please enter valid races separated by commas.")

    # Prompt for sample sizes corresponding to races
    while True:
        sample_sizes_input = input("Enter sample sizes corresponding to races (comma-separated, e.g., 500, 300, 200): ")
        try:
            # Convert the input to integers, stripping extra whitespace
            sample_sizes = [int(size.strip()) for size in sample_sizes_input.split(',') if size.strip()]
            # Ensure that the number of sample sizes matches the number of races
            if len(sample_sizes) == len(races):
                break
            else:
                print("The number of sample sizes must match the number of races. Please enter equal numbers of sample sizes and races.")
        except ValueError:
            # Error message for invalid integer input
            print("Invalid number provided. Please enter valid integers separated by commas.")
     
     # Log the user input details
    logging.info(f"Selected demographic feature: '{demographic_feature}'")
    logging.info(f"Races: {', '.join(races)}")
    logging.info(f"Sample sizes: {', '.join(map(str, sample_sizes))}")
    
    # Return the demographic feature, races, and sample sizes as a dictionary
    return {
        'demographic_feature': demographic_feature,
        'races': races,
        'sample_sizes': sample_sizes
    }


# Function to create a demographic subset based on provided inputs
def create_demographic_subset(merged_df, demo_inputs):
    """
    Creates a demographic subset based on given inputs.
    Parameters:
    - merged_df: The merged DataFrame containing the full dataset and chosen outcome.
    - demo_inputs: A dictionary containing the demographic feature, races, and sample sizes.
    """
    try:
        # Use the custom function to create subsets based on demographic criteria
        subset = create_demographic_subsets(
            data=merged_df,
            demographic_feature=demo_inputs['demographic_feature'],
            categories=demo_inputs['races'],
            sample_sizes=demo_inputs['sample_sizes'],
            exclude_hispanic=True  # Optionally exclude Hispanic categories if required
        )
        # Log successful creation of the demographic subset
        logging.info(f"Created demographic subset with feature '{demo_inputs['demographic_feature']}'.")
    except Exception as e:
        # Log any errors encountered while creating the subset
        logging.error(f"Error creating demographic subset: {e}")
        return None

    return subset
