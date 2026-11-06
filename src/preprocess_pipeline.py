# preprocess_pipeline.py

import logging

import pandas as pd

from src.preprocess import DataPreprocessor  # Preprocessing utility for data preparation

def preprocess_data(subset: pd.DataFrame, selected_outcome: list[str]) -> pd.DataFrame:
    """
    Preprocess the given subset of data.

    Any step that fails raises, so a run never continues on a partially
    preprocessed DataFrame (for example one missing engineered features).

    Parameters:
    - subset: DataFrame containing the demographic subset.
    - selected_outcome: The outcome column(s) to move to the end of the DataFrame.

    Returns:
    - processed_data: The preprocessed DataFrame.
    """

    preprocessor = DataPreprocessor(subset)

    # Logging to mark the start of the pre-processing phase
    logging.info("=== Pre-processing stage starting ===")

    # Specify columns to drop initially
    columns_to_drop_1 = [
        'pain_when', 'is_smoker', 'per_day', 'max', 'amount', 'depression', 'anxiety',
        'schizophrenia', "max", "amount", "cocaine_inject_days", "speedball_inject_days",
        "opioid_inject_days", "speed_inject_days", "UDS_Alcohol_Count", 'UDS_Mdma/Hallucinogen_Count'
    ]
    preprocessor.drop_columns_and_return(columns_to_drop_1)

    # Convert Yes/No values to binary (1/0)
    preprocessor.convert_yes_no_to_binary()
    logging.info("Converted Yes/No to binary.")

    # Specify columns for TLFB (Timeline Follow-Back) counts to process
    specified_tlfb_columns = [
        'TLFB_Alcohol_Count', 'TLFB_Amphetamine_Count', 'TLFB_Cocaine_Count',
        'TLFB_Heroin_Count', 'TLFB_Benzodiazepine_Count', 'TLFB_Opioid_Count',
        'TLFB_THC_Count', 'TLFB_Methadone_Count', 'TLFB_Buprenorphine_Count'
    ]

    # Process TLFB columns to transform their data
    preprocessor.process_tlfb_columns(specified_tlfb_columns)
    logging.info(f"Processed TLFB columns with specified columns: {specified_tlfb_columns}")

    # Calculate behavioral columns
    preprocessor.calculate_behavioral_columns()
    logging.info("Calculated behavioral columns.")

    # Move the selected outcome column to the end of the DataFrame
    preprocessor.move_column_to_end(selected_outcome)
    logging.info(f"Moved '{selected_outcome}' column to the end.")

    # Drop the raw columns the behavioral columns were calculated from
    columns_to_drop_2 = ['msm_npt', 'msm_frq', 'txx_prt']
    preprocessor.drop_columns_and_return(columns_to_drop_2)

    # transform_data_with_nan_handling
    preprocessor.transform_data_with_nan_handling()
    logging.info(f"transform_data_with_nan_handling success")

    # Rename columns according to a predefined mapping
    preprocessor.rename_columns()
    logging.info("Renamed columns according to the mapping.")

    # Transform NaN (missing values) to zeros for binary columns
    preprocessor.transform_nan_to_zero_for_binary_columns()
    logging.info("Transformed NaN to 0 for binary columns.")

    # Transform and rename 'heroin_inject_days' to 'rbsivheroin'
    preprocessor.transform_and_rename_column('heroin_inject_days', 'rbsivheroin')
    logging.info("Transformed and renamed 'heroin_inject_days' column to 'rbsivheroin'.")

    # Fill missing values (NaN) with zeros for the 'ftnd' column
    preprocessor.fill_nan_with_zero('ftnd')
    logging.info("Filled NaN with 0 for 'ftnd' column.")

    # Specify additional columns to drop
    columns_to_drop_3 = [
        'rbs_iv_days', 'race', 'RBS_cocaine_Days', 'RBS_heroin_Days',
        'RBS_opioid_Days', 'RBS_speed_Days', 'RBS_speedball_Days'
    ]
    preprocessor.drop_columns_and_return(columns_to_drop_3)

    # Convert drug count columns to binary
    preprocessor.convert_uds_to_binary()
    logging.info("Converted UDS drug counts to binary.")

    # Add RaceEth indicator columns for the models (RaceEth itself is kept for PSM)
    preprocessor.one_hot_encode_raceeth()
    logging.info("One-hot encoded RaceEth into indicator columns.")

    # After all processing, save the final processed dataframe
    processed_data = preprocessor.dataframe

    # Log the end of the pre-processing phase
    logging.info("=== Pre-processing stage complete ===")

    return processed_data
