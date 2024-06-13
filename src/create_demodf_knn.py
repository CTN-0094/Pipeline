import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from sklearn.neighbors import NearestNeighbors  # Import NearestNeighbors for K-Nearest Neighbors algorithm

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
    
    current_nhw = initial_nhw  # Initialize current NHW count
    current_minority = initial_minority  # Initialize current minority count

    subset_index = 0  # Initialize subset index for tracking

    while current_minority >= 0:  # Loop until there are no more minority participants to decrement
        nhw_sample_size = min(current_nhw, len(nhw_group))  # Determine the sample size for NHW group
        minority_sample_size = min(current_minority, len(minority_groups))  # Determine the sample size for minority group
        
        nhw_sample = nhw_group.sample(n=nhw_sample_size, random_state=42)  # Sample NHW participants
        minority_sample = minority_groups.sample(n=minority_sample_size, random_state=42)  # Sample minority participants
        
        subset = pd.concat([nhw_sample, minority_sample])  # Concatenate NHW and minority samples to form the subset
        print(f"Subset {subset_index + 1}: Initial subset size: {len(subset)} (NHW: {len(nhw_sample)}, Minority: {len(minority_sample)})")

        if len(subset) < initial_nhw + initial_minority:  # Check if additional NHW participants are needed
            remaining_count = (initial_nhw + initial_minority) - len(subset)  # Calculate the number of additional NHW participants needed
            if remaining_count > 0 and len(nhw_group) > 0:  # Check if there are NHW participants available
                additional_nhw = nhw_group.sample(n=min(remaining_count, len(nhw_group)), random_state=42, replace=True)  # Sample additional NHW participants
                subset = pd.concat([subset, additional_nhw])  # Concatenate the additional NHW participants to the subset
                print(f"Subset {subset_index + 1}: Additional NHW added: {len(additional_nhw)}, New subset size: {len(subset)}")

        df_list.append(subset[['who', 'RaceEth', 'age', 'is_female']])  # Append the subset to the list with selected columns
        
        current_nhw += step  # Increment the NHW count by the step size
        current_minority -= step  # Decrement the minority count by the step size

        if current_minority > 0:  # Replace the next batch of minority participants if needed
            if len(nhw_group) >= step:  # Check if there are enough NHW participants to replace the minority participants
                knn = NearestNeighbors(n_neighbors=10)  # Initialize the K-Nearest Neighbors model with 10 neighbors
                knn.fit(nhw_group[['age', 'is_female']])  # Fit the KNN model on the NHW participants' age and gender
                minority_to_replace = minority_groups.head(step)  # Select the next batch of minority participants to replace
                minority_to_replace_features = minority_to_replace[['age', 'is_female']]  # Get the age and gender features of the minority participants to replace
                distances, indices = knn.kneighbors(minority_to_replace_features)  # Find the nearest NHW participants for the minority participants to replace

                # Select only the closest 50 NHW participants (one per minority participant)
                closest_nhw_indices = []
                for i in range(step):
                    closest_nhw_indices.append(indices[i][0])  # Take only the closest NHW participant for each minority participant

                # # Print the matched NHW participants' indices and details
                # print(f"Subset {subset_index + 1}: Minority participants to replace (indices): {minority_to_replace.index.tolist()}")
                # print(f"Subset {subset_index + 1}: Matched NHW participants (indices): {closest_nhw_indices}")
                # print(f"Subset {subset_index + 1}: Matched NHW participants (details):\n{nhw_group.iloc[closest_nhw_indices]}")

                matched_nhw_indices = nhw_group.iloc[closest_nhw_indices].index  # Get the indices of the matched NHW participants
                nhw_group = nhw_group.drop(index=matched_nhw_indices)  # Drop the matched NHW participants from the NHW group
            else:
                print(f"Subset {subset_index + 1}: Not enough NHW participants to replace the minority participants for the next step.")  # Print a warning if there are not enough NHW participants
        
        subset_index += 1  # Increment the subset index
    
    if current_nhw <= 1000:  # Add a final subset if the total NHW participants are less than 1000
        final_nhw_sample = nhw_group.sample(n=min(1000, len(nhw_group)), random_state=42)  # Sample the remaining NHW participants
        df_list.append(final_nhw_sample[['who', 'RaceEth', 'age', 'is_female']])  # Append the final NHW sample to the list with selected columns
        # print(f"Final subset: size {len(final_nhw_sample)}, NHW count: {len(final_nhw_sample)}")
    
    return df_list  # Return the list of DataFrames
