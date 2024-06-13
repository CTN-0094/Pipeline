import pandas as pd

def merge_demographic_data(demographic_dfs, full_data):
    """
    Merge the list of demographic DataFrames with the full dataset.
    
    Parameters:
    - demographic_dfs: List of DataFrames with 'who', 'RaceEth', 'age', and 'is_female' columns.
    - full_data: The full dataset containing all other columns.
    
    Returns:
    - A list of merged DataFrames.
    """
    merged_dfs = []  # Initialize an empty list to store the merged DataFrames

    # Drop duplicate columns from full_data
    columns_to_drop = ['RaceEth', 'age', 'is_female']

    # Loop through each demographic DataFrame in the list
    for demo_df in demographic_dfs:
        # Merge the demographic DataFrame with the full dataset on the 'who' column
        merged_df = pd.merge(demo_df, full_data.drop(columns=columns_to_drop, errors='ignore'), on='who', how='left')
        # Append the merged DataFrame to the list
        merged_dfs.append(merged_df)
    
    return merged_dfs  # Return the list of merged DataFrames
