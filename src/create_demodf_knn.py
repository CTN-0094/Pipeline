import pandas as pd
import numpy as np

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

    # Track index of NHW samples to ensure we keep adding the next available ones
    nhw_index = initial_nhw_sample.index.tolist()  # Start with the indices of the initial NHW sample

    while current_minority > 0:  # Loop until there are no more minority participants to decrement
        minority_sample = remaining_minority_sample.head(current_minority)  # Select the first current_minority participants from the remaining sample

        if current_nhw < 1000:  # Add additional NHW participants until we reach 1000
            additional_nhw_needed = min(step, 1000 - current_nhw)
            # Select the next available NHW participants, skip duplicates
            additional_nhw_sample = nhw_group.drop(nhw_index).sample(n=additional_nhw_needed, random_state=42)
            current_nhw_sample = pd.concat([current_nhw_sample, additional_nhw_sample])
            nhw_index.extend(additional_nhw_sample.index.tolist())  # Track the newly selected NHW indices
            
        subset = pd.concat([current_nhw_sample.head(current_nhw), minority_sample])  # Use only the correct number of NHW participants
        df_list.append(subset[['who', 'RaceEth', 'age', 'is_female']])  # Append the subset to the list with selected columns
        
        current_nhw += step  # Increment the NHW count by the step size
        current_minority -= step  # Decrement the minority count by the step size

        if current_minority <= 0:  # If no more minority participants are left
            current_minority = 0  # Ensure current_minority is set to 0 for the final subset
            break  # Exit the loop

        remaining_minority_sample = remaining_minority_sample.head(current_minority)  # Update the remaining minority sample

    # Handle the final subset if it's not reaching 1000 participants
    while current_nhw_sample.shape[0] < 1000:
        additional_nhw_needed = 1000 - current_nhw_sample.shape[0]
        # Select the next available NHW participants, skip duplicates
        additional_nhw_sample = nhw_group.drop(nhw_index).sample(n=additional_nhw_needed, random_state=42)
        current_nhw_sample = pd.concat([current_nhw_sample, additional_nhw_sample]).drop_duplicates()
        nhw_index.extend(additional_nhw_sample.index.tolist())  # Track the newly selected NHW indices

    final_subset = current_nhw_sample.head(1000)  # Ensure the final subset has exactly 1000 NHW participants
    df_list.append(final_subset[['who', 'RaceEth', 'age', 'is_female']])  # Append the final subset to the list

    return df_list  # Return the list of DataFrames
