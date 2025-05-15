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
from src.logging_setup import setup_logging
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
from logScraper import scrape_log_to_csv

# List of supported outcome column names
AVAILABLE_OUTCOMES = [
    'ctn0094_relapse_event', 'Ab_krupitskyA_2011', 'Ab_ling_1998',
    'Rs_johnson_1992', 'Rs_krupitsky_2004', 'Rd_kostenB_1993'
]


def main():
    # Parse CLI arguments
    seedRange, outcomes, directory, profile, colSplit, sampleSize = argument_handler()

    # Create list of seeds
    seed_list = list(range(min(seedRange), max(seedRange))) if seedRange else [0]

    # Select outcomes
    if outcomes is None:
        outcomes = [get_outcome_choice(AVAILABLE_OUTCOMES)] if seedRange is None else AVAILABLE_OUTCOMES

    # Loop through outcomes and seeds
    for outcome in outcomes:
        processed_data = initialize_pipeline(outcome, columnToSplit=colSplit)

        for seed in seed_list:
            if profile in ['simple', None]:
                pf.simple_profile_pipeline(run_pipeline, seed, outcome, directory, processed_data, colSplit, sampleSize)
            elif profile == 'complex':
                pf.profile_pipeline(run_pipeline, seed, outcome, directory, processed_data, colSplit, sampleSize)
            else:
                run_pipeline(processed_data, seed, outcome, directory, colSplit, sampleSize)


def argument_handler():
    parser = argparse.ArgumentParser(description='Pipeline for statistical modeling and machine learning on the CTN-0094 database')

    # Seed loop (min and max)
    parser.add_argument('-l', '--loop', type=int, nargs='+', help='Minimum and maximum seed', default=None)

    # Specific outcomes to run
    parser.add_argument('-o', '--outcome', '--outcomes', type=str, nargs='+', help='All outcomes to run', default=None)

    # Directory to save logs and output
    parser.add_argument('-d', '--dir', '--directory', type=str, help='Directory to save logs, predictions, and evaluations', default="")

    # Profiling type
    parser.add_argument('-p', '--prof', '--profile', type=str, help='Type of profiling to run ("simple" or "complex")', default="None")

    # NEW: Column to define minority group (e.g., 'RaceEth', 'is_inpatient')
    parser.add_argument('--col', '--columnToSplit', type=str, default="RaceEth", help='Column used to define minority group')

    # NEW: Sample size for matching
    parser.add_argument('--samp', '--sampleSize', type=int, default=250, help='Number of minority participants to match')

    args = parser.parse_args()
    return args.loop, args.outcome, args.dir, args.prof, args.col, args.samp


def initialize_pipeline(selected_outcome, columnToSplit="RaceEth"):
    # Paths to data files
    master_path = 'data/master_data.csv'
    outcomes_path = 'data/all_binary_selected_outcomes.csv'
    inpatient_path = 'data/inpatient_care.csv'

    # Load demographic and outcome datasets
    demographic_df, outcomes_df = load_datasets(master_path, outcomes_path)

    # Merge 'is_inpatient' if it's the chosen column
    if columnToSplit == "is_inpatient":
        inpatient_df = pd.read_csv(inpatient_path)
        demographic_df = pd.merge(demographic_df, inpatient_df[['who', 'is_inpatient']], on='who', how='left')
        print("✅ is_inpatient value counts after merge:")
        print(demographic_df["is_inpatient"].value_counts(dropna=False))

        if demographic_df['is_inpatient'].isnull().any():
            raise ValueError("Missing values in 'is_inpatient' after merge.")

    # Merge selected outcome column into the main dataset
    outcome_column = outcomes_df[['who', selected_outcome]]
    merged_df = pd.merge(demographic_df, outcome_column, on='who', how='inner')

    # Preprocess and return the data
    processed_data = preprocess_merged_data(merged_df, selected_outcome)
    return processed_data


def run_pipeline(processed_data, seed, selected_outcome, directory, columnToSplit="RaceEth", sampleSize=100):
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Global Seed set to: {seed}")

    setup_logging(seed, selected_outcome, directory, quiet=False)

    processed_data, processed_data_heldout = holdOutTestData(processed_data, columnToSplit=columnToSplit)

    matched_dataframes = propensityScoreMatch(processed_data, columnToSplit=columnToSplit, sampleSize=sampleSize)

    print("✅ Matched groups:")
    print(f"Treated group (1): {matched_dataframes[0].shape[0]}")
    print(f"Control group 1 (0): {matched_dataframes[1].shape[0]}")
    print(f"Control group 2 (0): {matched_dataframes[2].shape[0]}")

    merged_subsets = create_subsets(matched_dataframes, sampleSize=sampleSize)

    results = train_and_evaluate_models(merged_subsets, selected_outcome, processed_data_heldout)

    save_predictions_to_csv(results.loc[:, ("subset", "predictions")], seed, selected_outcome, directory, 'subset_predictions')
    save_predictions_to_csv(results.loc[:, ("heldout", "predictions")], seed, selected_outcome, directory, 'heldout_predictions')
    save_evaluations_to_csv(results.loc[:, ("subset", "evaluations")], seed, selected_outcome, directory, 'subset_evaluations', columnToSplit)
    save_evaluations_to_csv(results.loc[:, ("heldout", "evaluations")], seed, selected_outcome, directory, 'heldout_evaluations', columnToSplit)

    log_pipeline_completion()


def save_evaluations_to_csv(results, seed, selected_outcome, directory, name, columnToSplit="RaceEth"):
    directory = os.path.join(directory, name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"{selected_outcome}_{seed}_{timestamp}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'global_seed', 'Outcome Type', 'Outcome Name',
            'Pre-processing script name', 'Model script name',
            'Demog Comparison', 'Prop(Demog)', 'Training Demographics',
            'TP', 'TN', 'FP', 'FN', 'Accuracy',
            'Precision', 'Recall', 'F1', 'ROC AUC Score'
        ])

        for id, trials_data in enumerate(results):
            tp = trials_data['confusion_matrix'][0][0]
            fn = trials_data['confusion_matrix'][1][0]
            fp = trials_data['confusion_matrix'][0][1]
            tn = trials_data['confusion_matrix'][1][1]
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Safe F1 calculation
            precision = trials_data['precision']
            recall = trials_data['recall']
            f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

            training_demographics = trials_data['training_demographics']

            writer.writerow([
                seed, "Binary", selected_outcome, "pipeline 1-2025", "TBD",
                f"{columnToSplit}: 1 vs 0",  # dynamic label
                trials_data['demographics'], training_demographics,
                tp, fn, fp, tn, accuracy,
                precision, recall, f1, trials_data['roc']
            ])


def save_predictions_to_csv(data, seed, selected_outcome, directory, name):
    predictions = {}
    for subsetNum, predictData in enumerate(data):
        for id, predScore in list(predictData):
            if id not in predictions:
                predictions[id] = [None] * data.shape[0]
            predictions[id][subsetNum] = predScore

    directory = os.path.join(directory, name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"{selected_outcome}_{seed}_{timestamp}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['who'] + [f'Subset_{i+1}' for i in range(10)]
        writer.writerow(header)

        for id, trials_data in predictions.items():
            writer.writerow([id] + trials_data)


# Entry point
if __name__ == "__main__":
    main()
