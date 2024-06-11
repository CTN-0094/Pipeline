import pandas as pd

def create_demographic_subsets(data, demographic_feature, categories, sample_sizes, **kwargs):
    """
    Creates demographic subsets from the provided DataFrame based on specified demographic features and sample sizes.
    Raises an error if the requested sample size cannot be met for any category.

    Parameters:
    - data: The source DataFrame or data convertible to DataFrame from which subsets are created.
    - demographic_feature: The column in the DataFrame specifying the demographic feature (e.g., 'race', 'gender').
    - categories: A list of categories within the demographic feature for which subsets are to be created.
    - sample_sizes: A list of integers that specifies the number of samples to include from each category.
    - kwargs: Optional keyword arguments, currently used to include/exclude specific subgroups such as 'exclude_hispanic'.
    """
    # Checks if a static variable 'previous_selections' exists, if not, initializes it as an empty dictionary.
    if not hasattr(create_demographic_subsets, "previous_selections"):
        create_demographic_subsets.previous_selections = {}
    
    # Ensures that 'data' is a pandas DataFrame.
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Raises an error if the demographic feature is not a column in the DataFrame.
    if demographic_feature not in data.columns:
        raise ValueError(f"The demographic feature '{demographic_feature}' does not exist in the DataFrame.")
    
    # Checks that the lengths of 'categories' and 'sample_sizes' are equal.
    if len(categories) != len(sample_sizes):
        raise ValueError("The length of 'categories' and 'sample_sizes' lists must be the same.")
    
    # Determines if Hispanic individuals should be excluded from the subset.
    exclude_hispanic = kwargs.get('exclude_hispanic', False) and demographic_feature == 'race'
    # Raises an error if Hispanic exclusion is requested but the 'is_hispanic' column is missing.
    if exclude_hispanic and 'is_hispanic' not in data.columns:
        raise ValueError("'is_hispanic' column is needed for Hispanic exclusion but is not found.")

    subsets = []
    new_selections = {}

    # Iterates over each category and its corresponding sample size.
    for category, sample_size in zip(categories, sample_sizes):
        # Filters data based on the category, applying Hispanic exclusion if needed.
        if category == 'White' and exclude_hispanic:
            filtered_data = data[(data[demographic_feature] == category) & (data['is_hispanic'] == 'No')]
        else:
            filtered_data = data[data[demographic_feature] == category]
        
        # Raises an error if the available data is less than the requested sample size.
        if len(filtered_data) < sample_size:
            raise ValueError(f"Requested sample size for '{category}' exceeds available data count.")

        # Handles the case where there are previous selections to consider.
        if category in create_demographic_subsets.previous_selections:
            prev_selection = create_demographic_subsets.previous_selections[category]
            additional_samples_needed = sample_size - len(prev_selection)
            if additional_samples_needed > 0:
                remaining_data = filtered_data[~filtered_data.index.isin(prev_selection.index)]
                additional_samples = remaining_data.sample(n=min(len(remaining_data), additional_samples_needed), replace=False)
                subset = pd.concat([prev_selection, additional_samples])
            else:
                subset = prev_selection.sample(n=sample_size, replace=False)
        else:
            # Selects a random sample from the filtered data.
            subset = filtered_data.sample(n=sample_size, replace=False)

        subsets.append(subset)
        new_selections[category] = subset

    # Concatenates all subsets into a single DataFrame, ignoring the original indices.
    final_subset = pd.concat(subsets, ignore_index=True)
    # Updates the static variable with the new selections.
    create_demographic_subsets.previous_selections = new_selections

    return final_subset

# Apples to Apples
#Split into two functions
#1 that creates a list of k df with dimension N*2 (N is the sum of the little ns) 
#The df are going to have sample IDs and Race_eth(only two columns)
#first df in list will be half NHW and half minority participants (N=1000)
#Last df in list will have the first 500 rows of the first df, (all NHW samples)
#And the last 500 rows will have the 500 minority participants with 500 other NHW N= 1000
#Each dataset in the middle will have decreasing count of minority participants from 450 down to 50
#For every call we will be replacing 50 samples but the first 500 NHW stay the same, AND 
#The other minority participants will stay the same. 
#who, race-eth, sex, outcome
#2 Second function takes in the list of k dfs of subject id and race eth and uses that
#to merge out the rest of the data that we need for modeling. 
