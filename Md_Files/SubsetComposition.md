## Subset Composition Visualization

### Overview
This visualization represents the demographic composition of subsets created through an iterative process. Each subset contains a specified number of Non-Hispanic White (NHW) participants and minority participants. The process replaces a portion of the minority participants with NHW participants in each iteration using the K-Nearest Neighbors (KNN) algorithm.

### Visualization
![Subset Composition](SubsetCompositionMultipleSubs.jpg)

### Key
- **NHW**: Non-Hispanic White
- **NHB**: Non-Hispanic Black
- **Hisp**: Hispanic
- **Other**: Other races or mixed
- **Refused/missing**: Handling missing values explicitly

### Subset Compositions
- **Subset 1**: 
  - 500 NHW
  - 162 NHB
  - 242 Hisp
  - 96 Other

- **Subset 2**:
  - 550 NHW
  - 147 NHB
  - 216 Hisp
  - 87 Other

- **Subset 3**:
  - 600 NHW
  - 131 NHB
  - 194 Hisp
  - 75 Other

- **Subset 4**:
  - 650 NHW
  - 119 NHB
  - 166 Hisp
  - 65 Other

- **Subset 5**:
  - 700 NHW
  - 100 NHB
  - 142 Hisp
  - 58 Other

- **Subset 6**:
  - 750 NHW
  - 83 NHB
  - 119 Hisp
  - 48 Other

- **Subset 7**:
  - 800 NHW
  - 70 NHB
  - 90 Hisp
  - 40 Other

- **Subset 8**:
  - 850 NHW
  - 56 NHB
  - 66 Hisp
  - 28 Other

- **Subset 9**:
  - 900 NHW
  - 36 NHB
  - 47 Hisp
  - 17 Other

- **Subset 10**:
  - 950 NHW
  - 17 NHB
  - 24 Hisp
  - 9 Other

- **Subset 11**:
  - 1000 NHW

### Explanation of the Subsetting Process

#### Initial Setup
1. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    ```

2. **Create Demographic Subsets**:
    ```python
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
    ```

#### Iterative Subsetting
1. **Iteration Loop**:
    ```python
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
    ```

#### Final Adjustment
1. **Ensure Final Subset**:
    ```python
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
    ```

### Visual Explanation
Each subplot in the provided visualization represents one subset generated through the iterative process. Starting from the initial composition of 500 NHW and 500 minority participants, each subsequent subset replaces 50 minority participants with 50 NHW participants. The KNN algorithm is used to select NHW participants that are demographically similar (in terms of age and gender) to the minority participants being replaced.

This iterative replacement continues until the final subset, which contains 1000 NHW participants, demonstrating the gradual demographic shift in each iteration.

### Conclusion
The provided visualization and this README explain the composition and creation process of each demographic subset using the KNN algorithm and iterative replacement method. The subsets are carefully constructed to maintain demographic similarities between replaced participants, ensuring a robust analysis for each iteration.
