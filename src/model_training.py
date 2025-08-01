from src.train_model import LogisticModel, NegativeBinomialModel, CoxProportionalHazard, BetaRegression
from src.constants import EndpointType
import logging
import numpy as np
import csv
import os
import pandas as pd
from datetime import datetime



def train_and_evaluate_models(merged_subsets, id_column, selected_outcome, processed_data_heldout):
    columns = pd.MultiIndex.from_product(
        [["heldout", "subset"], ["predictions", "evaluations"]],
        names=["Data Type", "Metric"]
    )
    results = pd.DataFrame(columns=columns)

    selectedModel = None
    if selected_outcome['endpointType'] == EndpointType.LOGICAL:
        selectedModel = LogisticModel
    if selected_outcome['endpointType'] == EndpointType.SURVIVAL:
        selectedModel = CoxProportionalHazard
    if selected_outcome['endpointType'] == EndpointType.INTEGER:
        selectedModel = NegativeBinomialModel#BetaRegression
    
    numOfSubsets = len(merged_subsets)

    for i, subset in enumerate(merged_subsets):
        # Log the demographic makeup of each subset
        #demographic_counts = subset['RaceEth'].value_counts().to_dict()
        #demographicStrings.append[", ".join([f"{v} {k}" for k, v in demographic_counts.items()])]
        #logging.info(f"Subset {i + 1} demographic makeup: {demographicStrings[-1]}")
        logging.info(f"Processing subset {i + 1}...")
        print(f"\nProcessing subset {i + 1} " + "_" * 100)

        logging.info("-----------------------------")
        logging.info(f"TRAIN MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("-----------------------------")

        #subset = subset.drop('RaceEth', axis=1)
        outcomeModel = selectedModel(subset, id_column, selected_outcome['columnsToUse'])
        outcomeModel.selectFeatures()
        #outcomeModel.selected_features=['age', 'RaceEth', 'unstableliving', 'is_female', 'UDS_Amphetamine_Count']
        outcomeModel.train()
        
        logging.info(f"Model trained and saved successfully for subset {i + 1}.")

        logging.info("---------------------------")
        logging.info(f"TRAIN MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("---------------------------")

        logging.info("--------------------------------")
        logging.info(f"EVALUATE MODEL STAGE STARTING FOR SUBSET {i + 1}...")
        logging.info("--------------------------------")
        
        #try:
        #Write result to dictionary of arrays

        results.loc[len(results)] = outcomeModel.evaluate(processed_data_heldout)

        logging.info(f"Model evaluated successfully for subset {i + 1}.")
        # except Exception as e:
        #     logging.error(f"Error during model evaluation for subset {i + 1}: {e}")
        #     return

        logging.info("------------------------------")
        logging.info(f"EVALUATE MODEL STAGE COMPLETED FOR SUBSET {i + 1}")
        logging.info("------------------------------")
        print("_" * 120 + "\n")
            
        '''for id, result in subsetPredsAndResults["predictions"]:
            if id not in subsetPredictions:
                subsetPredictions[id] = [None] * numOfSubsets
            subsetPredictions[id][i] = result
        for id, result in heldOutPredsAndResults["predictions"]:
            if id not in heldoutPredictions:
                heldoutPredictions[id] = [None] * numOfSubsets
            heldoutPredictions[id][i] = result
    save_predictions_to_csv(predictions, seed, selected_outcome, directory, "predictions")
    save_Test_predictions_to_csv(predictions, seed, selected_outcome, directory, testValues)'''
    logging.info(f"Model predictions saved to csv successfully.")

    return results
    