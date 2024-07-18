# README

## Project Inventory and Implementation Plan

This document outlines the tasks and features to be implemented in our data processing, model training, and optimization pipeline.

### Overview

We aim to improve the performance and efficiency of our pipeline by implementing parallel processing, efficient model training, profiling and optimization, caching, and memory management. Below is an inventory of these tasks along with detailed plans and progress updates.

### Task Inventory

1. **Setup and Initialization**
   - Initialize logging configuration
   - Obtain user inputs for the chosen outcome variable

2. **Data Loading**
   - Load the master dataset
   - Load the outcomes dataset

3. **Data Merging**
   - Merge the master dataset with the outcomes dataset

4. **Preprocessing**
   - Preprocess the merged dataset using caching for efficiency

5. **Create Demographic Subsets**
   - Create demographic subsets using caching

6. **Merge Demographic Data**
   - Merge the demographic subsets

7. **Parallel Processing of Subsets**
   - Process each subset in parallel using `ProcessPoolExecutor`
     - Initialize and train logistic model
     - Feature selection and model fitting
     - Find the best threshold
     - Evaluate the model

8. **Memory Management**
   - Implement memory management to clear memory after processing each subset

9. **Profiling**
   - Profile the pipeline to identify bottlenecks

10. **Completion**
    - Log the completion of the pipeline

### Detailed Implementation Plan

#### Setup and Initialization
- **Goal**: Establish a robust logging configuration and gather necessary user inputs.
- **Status**: Pending
- **Code**:
    ```python
    setup_logging()
    selected_outcome = get_outcome_choice()
    ```

#### Data Loading
- **Goal**: Efficiently load the required datasets.
- **Status**: Pending
- **Code**:
    ```python
    master_df = pd.read_csv('/path/to/master_data.csv')
    outcomes_df = pd.read_csv('/path/to/all_binary_selected_outcomes.csv')
    ```

#### Data Merging
- **Goal**: Merge the master dataset with the outcomes dataset.
- **Status**: Pending
- **Code**:
    ```python
    outcome_column = outcomes_df[['who', selected_outcome]]
    merged_df = pd.merge(master_df, outcome_column, on='who', how='inner')
    ```

#### Preprocessing
- **Goal**: Preprocess the merged dataset using caching for efficiency.
- **Status**: Pending
- **Code**:
    ```python
    processed_data = preprocess_data(merged_df, selected_outcome)
    ```

#### Create Demographic Subsets
- **Goal**: Create demographic subsets using caching.
- **Status**: Pending
- **Code**:
    ```python
    subsets = create_demographic_dfs(processed_data)
    ```

#### Merge Demographic Data
- **Goal**: Merge the demographic subsets.
- **Status**: Pending
- **Code**:
    ```python
    merged_subsets = merge_demographic_data(subsets, processed_data)
    ```

#### Parallel Processing of Subsets
- **Goal**: Process each subset in parallel using `ProcessPoolExecutor`.
- **Status**: Pending
- **Code**:

#### Memory Management
- **Goal**: Implement memory management to clear memory after processing each subset.
- **Status**: Pending
- **Code**:

#### Profiling
- **Goal**: Profile the pipeline to identify bottlenecks.
- **Status**: Pending
- **Code**:

#### Completion
- **Goal**: Log the completion of the pipeline.
- **Status**: Pending

