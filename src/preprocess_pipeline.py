# preprocess_pipeline.py

import logging
from src.preprocess import DataPreprocessor  # Preprocessing utility for data preparation

def preprocess_data(subset, selected_outcome):
    """
    Preprocess the given subset of data.
    
    Parameters:
    - subset: DataFrame containing the demographic subset.
    - selected_outcome: The chosen outcome variable.
    
    Returns:
    - processed_data: The preprocessed DataFrame.
    """
    
    preprocessor = DataPreprocessor(subset)
    
    # Logging to mark the start of the pre-processing phase
    logging.info("---------------------------------")
    logging.info("PRE-PROCESSING STAGE STARTING...")
    logging.info("---------------------------------")

    # Specify columns to drop initially
    columns_to_drop_1 = [
        'pain_when', 'is_smoker', 'per_day', 'max', 'amount', 'depression', 'anxiety',
        'schizophrenia', "max", "amount", "cocaine_inject_days", "speedball_inject_days",
        "opioid_inject_days", "speed_inject_days", "UDS_Alcohol_Count", 'UDS_Mdma/Hallucinogen_Count'
    ]
    
    # Try to drop the specified columns
    try:
        preprocessor.drop_columns_and_return(columns_to_drop_1)
        logging.info(f"Dropped initial columns: {columns_to_drop_1}")
    except Exception as e:
        logging.error(f"Error dropping initial columns: {e}")

    # Convert Yes/No values to binary (1/0)
    try:
        preprocessor.convert_yes_no_to_binary()
        logging.info("Converted Yes/No to binary.")
    except Exception as e:
        logging.error(f"Error converting Yes/No to binary: {e}")

    # Specify columns for TLFB (Timeline Follow-Back) counts to process
    specified_tlfb_columns = [
        'TLFB_Alcohol_Count', 'TLFB_Amphetamine_Count', 'TLFB_Cocaine_Count',
        'TLFB_Heroin_Count', 'TLFB_Benzodiazepine_Count', 'TLFB_Opioid_Count',
        'TLFB_THC_Count', 'TLFB_Methadone_Count', 'TLFB_Buprenorphine_Count'
    ]
    
    # Process TLFB columns to transform their data
    try:
        preprocessor.process_tlfb_columns(specified_tlfb_columns)
        logging.info(f"Processed TLFB columns with specified columns: {specified_tlfb_columns}")
    except Exception as e:
        logging.error(f"Error processing TLFB columns: {e}")

    # Calculate behavioral columns
    try:
        preprocessor.calculate_behavioral_columns()
        logging.info("Calculated behavioral columns.")
    except Exception as e:
        logging.error(f"Error calculating behavioral columns: {e}")

    # Move the selected outcome column to the end of the DataFrame
    try:
        preprocessor.move_column_to_end(selected_outcome)
        logging.info(f"Moved '{selected_outcome}' column to the end.")
    except Exception as e:
        logging.error(f"Error moving '{selected_outcome}' column to the end: {e}")


    # Specify columns to drop initially
    columns_to_drop_2 = ['msm_npt', 'msm_frq', 'txx_prt']
    
    # Try to drop the specified columns
    try:
        preprocessor.drop_columns_and_return(columns_to_drop_2)
        logging.info(f"Dropped initial columns: {columns_to_drop_2}")
    except Exception as e:
        logging.error(f"Error dropping initial columns: {e}")

    # transform_data_with_nan_handling
    try:
        preprocessor.transform_data_with_nan_handling()
        logging.info(f"transform_data_with_nan_handling success")
    except Exception as e:
        logging.error(f"Error handling NaN values: {e}")

    # Rename columns according to a predefined mapping
    try:
        preprocessor.rename_columns()
        logging.info("Renamed columns according to the mapping.")
    except Exception as e:
        logging.error(f"Error renaming columns: {e}")

    # Transform NaN (missing values) to zeros for binary columns
    try:
        preprocessor.transform_nan_to_zero_for_binary_columns()
        logging.info("Transformed NaN to 0 for binary columns.")
    except Exception as e:
        logging.error(f"Error transforming NaN to 0 for binary columns: {e}")

    # Transform and rename 'heroin_inject_days' to 'rbsivheroin'
    try:
        preprocessor.transform_and_rename_column('heroin_inject_days', 'rbsivheroin')
        logging.info("Transformed and renamed 'heroin_inject_days' column to 'rbsivheroin'.")
    except Exception as e:
        logging.error(f"Error transforming and renaming 'heroin_inject_days' column: {e}")

    # Fill missing values (NaN) with zeros for the 'ftnd' column
    try:
        preprocessor.fill_nan_with_zero('ftnd')
        logging.info("Filled NaN with 0 for 'ftnd' column.")
    except Exception as e:
        logging.error(f"Error filling NaN with 0 for 'ftnd': {e}")

    # Specify additional columns to drop
    columns_to_drop_2 = [
        'rbs_iv_days', 'race', 'RBS_cocaine_Days', 'RBS_heroin_Days',
        'RBS_opioid_Days', 'RBS_speed_Days', 'RBS_speedball_Days'
    ]
    
    # Try to drop additional columns
    try:
        preprocessor.drop_columns_and_return(columns_to_drop_2)
        logging.info(f"Dropped additional columns: {columns_to_drop_2}")
    except Exception as e:
        logging.error(f"Error dropping additional columns: {e}")

    # Convert drug count columns to binary
    try:
        preprocessor.convert_uds_to_binary()
        logging.info("Converted UDS drug counts to binary.")
    except Exception as e:
        logging.error(f"Error converting UDS drug counts to binary: {e}")


    # After all processing, save the final processed dataframe
    processed_data = preprocessor.dataframe
    #logging.info("Final Processed DataFrame:")
    #logging.info(processed_data.head())

    # Log the end of the pre-processing phase
    logging.info("------------------------------")
    logging.info("PRE-PROCESSING STAGE COMPLETED")
    logging.info("------------------------------")
    
    return processed_data
