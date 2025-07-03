import sys
import os
import random
import numpy as np
import pandas as pd
import cProfile
import pstats
from datetime import datetime
from src.constants import EndpointType
import logging
import io
import src.profiling as pf
from src.silent_logging import add_silent_handler
from src.logging_setup import setup_logging  # Import the logging setup from logging_setup.py
import argparse
import re
import csv

# Add the 'src' directory to the system path to allow imports from that directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary functions from the 'src' directory
from utils import log_pipeline_completion, get_outcome_choice
from data_loading import load_datasets
from data_preprocessing import preprocess_merged_data
from create_demodf_knn import create_subsets, holdOutTestData, propensityScoreMatch
from model_training import train_and_evaluate_models
from logScraper import scrape_log_to_csv  # Import the log scraper function
from enum import Enum



AVAILABLE_OUTCOMES = [
    {
        'name': 'ctn0094_relapse_event',
        'columnsToUse': ['ctn0094_relapse_event'],
        'endpointType': EndpointType.LOGICAL
    },
    {
        'name': 'Ab_krupitskyA_2011',
        'columnsToUse': ['Ab_krupitskyA_2011'],
        'endpointType': EndpointType.LOGICAL
    },
    {
        'name': 'Ab_ling_1998',
        'columnsToUse': ['Ab_ling_1998'],
        'endpointType': EndpointType.LOGICAL
    },
    {
        'name': 'Rs_johnson_1992',
        'columnsToUse': ['Rs_johnson_1992'],
        'endpointType': EndpointType.LOGICAL
    },
    {
        'name': 'Rs_krupitsky_2004',
        'columnsToUse': ['Rs_krupitsky_2004'],
        'endpointType': EndpointType.LOGICAL
    },
    {
        'name': 'Rd_kostenB_1993',
        'columnsToUse': ['Rd_kostenB_1993'],
        'endpointType': EndpointType.LOGICAL
    },
    {
        'name': 'Ab_schottenfeldB_2008',
        'columnsToUse': ['Ab_schottenfeldB_2008'],
        'endpointType': EndpointType.INTEGER
    },
    {
        'name': 'Ab_mokri_2016',
        'columnsToUse': ['AbT_mokri_2016', 'AbE_mokri_2016'],
        'endpointType': EndpointType.SURVIVAL
    }
]



def main():
    seedRange, outcomes, directory, profile = argument_handler()

    #Process arguments
    seed_list = list(range(min(seedRange), max(seedRange))) if seedRange is not None else [0]
    
    if outcomes is None:
        if(seedRange is None):
            outcomes = [get_outcome_choice(AVAILABLE_OUTCOMES)]
        else:
            outcomes = AVAILABLE_OUTCOMES

    # Loop through each seed and run the pipeline
    for outcome in outcomes:
        #Initialize Pipeline
        processed_data = initialize_pipeline(outcome)

        for seed in seed_list:
            if profile == 'simple' or profile == None:
                pf.simple_profile_pipeline(run_pipeline, seed, outcome, directory)
            elif profile == 'complex':
                pf.profile_pipeline(run_pipeline, seed, outcome, directory)
            else:
                run_pipeline(processed_data, seed, outcome, directory)



def argument_handler():

    # Create the parser
    parser = argparse.ArgumentParser(description='Pipeline for statistical modeling and machine learning on the CTN-0094 database')

    # Add arguments loop (min and max seed, prompt otherwise) target directory, profile
    parser.add_argument('-l', '--loop', type=int, nargs='+', help='minimum and maximum seed', default = None)
    parser.add_argument('-o', '--outcome', '--outcomes', type=str, nargs='+', help='all outcomes to run', default = None)
    parser.add_argument('-d', '--dir', '--directory', type=str, help='directory to save logs, predictions, and evaluations', default="")
    parser.add_argument('-p', '--prof', '--profile', type=str, help='type of profiling to run (\'simple\' or \'complex\')', default="None")

    # Parse the arguments
    args = parser.parse_args()

    return args.loop, args.outcome, args.dir, args.prof



def initialize_pipeline(selected_outcome):

    # Paths to the data files
    #TODO: MAKE THIS DATA NOT HARD CODED!!!!!!!!!!!!!!!!!!!!!!
    master_path = 'data/master_data.csv'
    outcomes_path = ['data/outcomesCTN0094.csv', 'data/all_binary_selected_outcomes.csv']
    columnToSplitOn = "RaceEth"
    columnToDrop= "is_hispanic"
    
    # Load the data with the outcome
    demographic_df, outcomes_df = load_datasets(master_path, outcomes_path)
    outcome_column = outcomes_df[['who'] + selected_outcome['columnsToUse']]
    
    merged_df = pd.merge(demographic_df, outcome_column, on='who', how='inner')


    #FOR BETA REGRESSION SCALING
    '''
    epsilon = 1e-6  # small value to avoid exact 0 or 1
    min_val = merged_df[selected_outcome['columnsToUse']].min()
    max_val = merged_df[selected_outcome['columnsToUse']].max()
    merged_df[selected_outcome['columnsToUse']] = (merged_df[selected_outcome['columnsToUse']] - min_val) / (max_val - min_val)
    merged_df[selected_outcome['columnsToUse']] = merged_df[selected_outcome['columnsToUse']] * (1 - 2 * epsilon) + epsilon
    '''

    # Preprocess the merged data based on the selected outcome
    processed_data = preprocess_merged_data(merged_df, selected_outcome['columnsToUse'])

    processed_data = processed_data.drop(columnToDrop, axis=1)

    return processed_data
    


def run_pipeline(processed_data, seed, selected_outcome, directory):

    idColumn = "who"

    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Global Seed set to: {seed}")

    setup_logging(seed, selected_outcome['name'], directory, quiet=False)

    #Make demographic subsets
    processed_data, processed_data_heldout = holdOutTestData(processed_data, idColumn) #Move to saving to file, but also dont ovewrite current file if run again.

    matched_dataframes = propensityScoreMatch(processed_data, idColumn)

    # Create and merge demographic subsets
    merged_subsets = create_subsets(matched_dataframes)

    # Train and evaluate the models using the merged subsets
    results = train_and_evaluate_models(merged_subsets, idColumn, selected_outcome, processed_data_heldout)
    
    save_predictions_to_csv(results.loc[:, ("subset", "predictions")], seed, selected_outcome, directory, 'subset_predictions')
    save_predictions_to_csv(results.loc[:, ("heldout", "predictions")], seed, selected_outcome, directory, 'heldout_predictions')
    save_evaluations_to_csv(results.loc[:, ("subset", "evaluations")], seed, selected_outcome, directory, 'subset_evaluations')
    save_evaluations_to_csv(results.loc[:, ("heldout", "evaluations")], seed, selected_outcome, directory, 'heldout_evaluations')

    # Log the completion of the pipeline
    log_pipeline_completion()



def save_evaluations_to_csv(results, seed, selected_outcome, directory, name):
    """
    Save the evaluation results to a CSV file, including training demographics.

    Parameters:
    - results: Evaluation results containing metrics and demographics.
    - seed: The random seed used for the run.
    - selected_outcome: The outcome being analyzed.
    - directory: The directory where the CSV file should be saved.
    - name: The name of the subfolder to save the file in.
    """
    
    # Ensure the directory exists
    directory = os.path.join(directory, name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate the filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"{selected_outcome['name']}_{seed}_{timestamp}.csv")

    # Open the file for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        headers = [
            'global_seed',
            'Outcome Type',
            'Outcome Name',
            'Pre-processing script name',
            'Model script name',
            'Demog Comparison',
            'Prop(Demog)',
            'Training Demographics'  # New column
        ]

        if selected_outcome['endpointType'] == EndpointType.LOGICAL:
            # Write the header with the new "Training Demographics" column
            writer.writerow(headers + [
                'TP',
                'TN',
                'FP',
                'FN',
                'Accuracy',
                'Precision',
                'Recall',
                'F1',
                'ROC AUC Score'
            ])
        elif selected_outcome['endpointType'] == EndpointType.INTEGER:
            writer.writerow(headers + [
                'MSE',
                'RMSE',
                'MAE',
                'pearson_r',
                'mcfadden_r2'
            ])
        elif selected_outcome['endpointType'] == EndpointType.SURVIVAL:
            writer.writerow(headers + [
                'C_Index'
            ])

        # Write data rows
        for id, trials_data in enumerate(results):

            # Fetch training demographics
            training_demographics = trials_data['training_demographics']

            sections = [
                seed,
                selected_outcome['endpointType'],
                selected_outcome['name'],
                "pipeline 1-2025",
                "TBD",
                "Race: non hispanic white vs minority",
                trials_data['demographics'],  # Prop(Demog)
                training_demographics  # New column
            ]

            if selected_outcome['endpointType'] == EndpointType.LOGICAL:
                tp = trials_data['confusion_matrix'][0][0]
                fn = trials_data['confusion_matrix'][1][0]
                fp = trials_data['confusion_matrix'][0][1]
                tn = trials_data['confusion_matrix'][1][1]
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                f1 = 2 * (trials_data['precision'] * trials_data['recall']) / (trials_data['precision'] + trials_data['recall'])

                writer.writerow(sections + [
                    tp,
                    fn,
                    fp,
                    tn,
                    accuracy,
                    trials_data['precision'],
                    trials_data['recall'],
                    f1,
                    trials_data['roc']
                ])
            elif selected_outcome['endpointType'] == EndpointType.INTEGER:
                writer.writerow(sections + [
                    trials_data['mse'],
                    trials_data['rmse'],
                    trials_data['mae'],
                    trials_data['pearson_r'],
                    trials_data['mcfadden_r2']
                ])
            elif selected_outcome['endpointType'] == EndpointType.SURVIVAL:
                print("AARON DEBUG INSIDE")
                writer.writerow(sections + [
                    trials_data['concordance_index']
                ])



def save_predictions_to_csv(data, seed, selected_outcome, directory, name):

    predictions = {}
    for subsetNum, predictData in enumerate(data):
        for id, predScore in list(predictData):
            if id not in predictions:
                predictions[id] = [None] * data.shape[0]
            predictions[id][subsetNum] = predScore

    # Define the profiling log file path
    directory = os.path.join(directory, name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"{selected_outcome['name']}_{seed}_{timestamp}.csv")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

         # Write header
        header = ['who'] + [f'Subset_{i+1}' for i in range(10)]
        writer.writerow(header)
        
        # Write data rows
        for id, trials_data in predictions.items():
            writer.writerow([id] + trials_data)




if __name__ == "__main__":
    main()
    