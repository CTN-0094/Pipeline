import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

def create_demographic_dfs(data, initial_nhw=500, initial_minority=500, step=50):
    df_list = []
    minority_groups = data[data['RaceEth'] != 1]
    nhw_group = data[data['RaceEth'] == 1]
    
    current_nhw = initial_nhw
    current_minority = initial_minority

    while current_minority >= 0:
        nhw_sample_size = min(current_nhw, len(nhw_group))
        minority_sample_size = min(current_minority, len(minority_groups))
        
        nhw_sample = nhw_group.sample(n=nhw_sample_size, random_state=42)
        minority_sample = minority_groups.sample(n=minority_sample_size, random_state=42)
        
        subset = pd.concat([nhw_sample, minority_sample])
        
        if len(subset) < initial_nhw + initial_minority:
            remaining_count = (initial_nhw + initial_minority) - len(subset)
            if remaining_count > 0 and len(nhw_group) > 0:
                additional_nhw = nhw_group.sample(n=min(remaining_count, len(nhw_group)), random_state=42, replace=True)
                subset = pd.concat([subset, additional_nhw])
        
        df_list.append(subset[['who', 'RaceEth', 'age', 'is_female']])
        
        current_nhw += step
        current_minority -= step

        if current_minority > 0:
            if len(nhw_group) >= step:
                knn = NearestNeighbors(n_neighbors=10)
                knn.fit(nhw_group[['age', 'is_female']])
                minority_to_replace = minority_groups.head(step)
                minority_to_replace_features = minority_to_replace[['age', 'is_female']]
                distances, indices = knn.kneighbors(minority_to_replace_features)
                
                matched_nhw_indices = nhw_group.iloc[indices.flatten()].index
                nhw_group = nhw_group.drop(index=matched_nhw_indices)
            else:
                logging.warning("Not enough NHW participants to replace the minority participants for the next step.")
    
    if current_nhw <= 1000:
        final_nhw_sample = nhw_group.sample(n=min(1000, len(nhw_group)), random_state=42)
        df_list.append(final_nhw_sample[['who', 'RaceEth', 'age', 'is_female']])
    
    return df_list
