import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def create_demographic_dfs(data, initial_nhw=500, initial_minority=500, step=50):
    """
    Create a list of DataFrames with different demographic distributions.
    
    Parameters:
    - data: DataFrame containing the dataset.
    - initial_nhw: Initial number of Non-Hispanic White participants.
    - initial_minority: Initial number of minority participants.
    - step: The step size for increasing NHW and decreasing minority participants in each subsequent DataFrame.
    
    Returns:
    - df_list: A list of DataFrames with different demographic distributions.
    """
    
    df_list = []  # Initialize an empty list to store the DataFrames
    minority_groups = data[data['RaceEth'] != 1]  # Select rows where 'RaceEth' is not NHW
    nhw_group = data[data['RaceEth'] == 1]  # Select rows where 'RaceEth' is NHW
    
    initial_nhw_sample = nhw_group.sample(n=initial_nhw)  # Sample initial NHW participants
    initial_minority_sample = minority_groups.sample(n=initial_minority)  # Sample initial minority participants
    
    current_nhw_sample = initial_nhw_sample.copy()  # Copy the initial NHW sample to use as the starting point
    remaining_minority_sample = initial_minority_sample.copy()  # Copy the initial minority sample to use as the starting point
    
    current_nhw = initial_nhw  # Initialize current NHW count
    current_minority = initial_minority  # Initialize current minority count

    while current_minority > 0:  # Loop until there are no more minority participants to decrement
        minority_sample = remaining_minority_sample.head(current_minority)  # Select the first current_minority participants from the remaining sample
        
        if current_nhw > initial_nhw:  # If additional NHW participants are needed
            knn = NearestNeighbors(n_neighbors=1)  # Initialize the K-Nearest Neighbors model with 1 neighbor
            knn.fit(nhw_group[['age', 'is_female']])  # Fit the KNN model on the NHW participants' age and gender
            minority_features = remaining_minority_sample.head(step)[['age', 'is_female']]  # Get the age and gender features of the next batch of minority participants to replace
            distances, indices = knn.kneighbors(minority_features)  # Find the nearest NHW participants for the minority participants to replace
            closest_nhw_indices = indices.flatten()  # Get the indices of the closest NHW participants
            additional_nhw_sample = nhw_group.iloc[closest_nhw_indices]  # Select the closest NHW participants
            current_nhw_sample = pd.concat([current_nhw_sample, additional_nhw_sample])  # Concatenate the additional NHW participants to the current NHW sample
            nhw_group = nhw_group.drop(nhw_group.index[closest_nhw_indices])  # Drop the selected NHW participants from the NHW group

        subset = pd.concat([current_nhw_sample, minority_sample])  # Concatenate the NHW and minority samples to form the subset
        df_list.append(subset[['who', 'RaceEth', 'age', 'is_female']])  # Append the subset to the list with selected columns
        
        current_nhw += step  # Increment the NHW count by the step size
        current_minority -= step  # Decrement the minority count by the step size

        if current_minority <= 0:  # If no more minority participants are left
            current_minority = 0  # Ensure current_minority is set to 0 for the final subset
            break  # Exit the loop

        remaining_minority_sample = remaining_minority_sample.head(current_minority)  # Update the remaining minority sample

    # Ensure the final subset has exactly 1000 NHW participants
    while current_nhw_sample.shape[0] < 1000:
        knn = NearestNeighbors(n_neighbors=1)  # Initialize the K-Nearest Neighbors model with 1 neighbor
        knn.fit(nhw_group[['age', 'is_female']])  # Fit the KNN model on the NHW participants' age and gender
        distances, indices = knn.kneighbors(nhw_group[['age', 'is_female']].head(1))  # Find the nearest NHW participants
        closest_nhw_indices = indices.flatten()  # Get the indices of the closest NHW participants
        additional_nhw_sample = nhw_group.iloc[closest_nhw_indices]  # Select the closest NHW participants
        current_nhw_sample = pd.concat([current_nhw_sample, additional_nhw_sample])  # Add these NHW participants to the current sample
        current_nhw_sample = current_nhw_sample.drop_duplicates()  # Drop duplicate NHW participants
        nhw_group = nhw_group.drop(nhw_group.index[closest_nhw_indices])  # Drop the selected NHW participants

    final_subset = current_nhw_sample.head(1000)  # Ensure the final subset has exactly 1000 NHW participants
    df_list.append(final_subset[['who', 'RaceEth', 'age', 'is_female']])  # Append the final subset to the list

    return df_list  # Return the list of DataFrames

