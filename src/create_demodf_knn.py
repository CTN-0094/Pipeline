import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import logging

def create_demographic_dfs(data, initial_nhw=500, initial_minority=500, step=50):
    """
    Create a list of DataFrames with the specified demographic distributions.
    
    Parameters:
    - data: DataFrame containing 'who', 'RaceEth', 'age', 'is_female', and 'outcome' columns.
    - initial_nhw: Initial number of Non-Hispanic White participants.
    - initial_minority: Initial number of minority participants.
    - step: The decrement step for minority participants in each subsequent DataFrame.
    
    Returns:
    - A list of DataFrames with the required demographic distributions.
    """
    df_list = []  # Initialize an empty list to store the DataFrames
    minority_groups = data[data['RaceEth'] != 1]  # Select rows where 'RaceEth' is not 'NHW'
    nhw_group = data[data['RaceEth'] == 1]  # Select rows where 'RaceEth' is 'NHW'
    
    current_nhw = initial_nhw
    current_minority = initial_minority

    while current_minority >= 0:
        # Check if there are enough participants available for sampling
        nhw_sample_size = min(current_nhw, len(nhw_group))
        minority_sample_size = min(current_minority, len(minority_groups))

        # Select NHW and minority participants for the current subset
        nhw_sample = nhw_group.sample(n=nhw_sample_size, random_state=42)
        minority_sample = minority_groups.sample(n=minority_sample_size, random_state=42)

        # Concatenate to form the current subset
        subset = pd.concat([nhw_sample, minority_sample])

        # Ensure the subset has the correct total number of samples
        if len(subset) < initial_nhw + initial_minority:
            remaining_count = (initial_nhw + initial_minority) - len(subset)
            additional_nhw = nhw_group.sample(n=remaining_count, random_state=42, replace=True)
            subset = pd.concat([subset, additional_nhw])

        # Append the combined DataFrame to the list, keeping only 'who', 'RaceEth', and 'is_female' columns
        df_list.append(subset[['who', 'RaceEth', 'age', 'is_female']])

        # Update counts for the next subset
        current_nhw += step
        current_minority -= step

        # Check if there are still minority participants to replace
        if current_minority > 0:
            # Find similar NHW participants to replace the next batch of minority participants
            if len(nhw_group) >= step:
                knn = NearestNeighbors(n_neighbors=10)  # Use 10 nearest neighbors
                knn.fit(nhw_group[['age', 'is_female']])  # Fit the model using selected features
                minority_to_replace = minority_groups.head(step)  # Select the top minority participants to replace
                minority_to_replace_features = minority_to_replace[['age', 'is_female']]  # Get their features
                distances, indices = knn.kneighbors(minority_to_replace_features)  # Find the nearest NHW participants

                # Add the matched NHW participants to the next subset
                matched_nhw_indices = nhw_group.iloc[indices.flatten()].index
                nhw_group = nhw_group.drop(index=matched_nhw_indices)  # Drop the matched NHW participants from the pool
            else:
                logging.warning("Not enough NHW participants to replace the minority participants for the next step.")

    # Add final subset with all NHW participants if current_nhw is less than 1000
    if current_nhw <= 1000:
        final_nhw_sample = nhw_group.sample(n=1000, random_state=42)
        df_list.append(final_nhw_sample[['who', 'RaceEth', 'age', 'is_female']])

    return df_list  # Return the list of DataFrames


