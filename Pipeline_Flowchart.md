Setup and Initialization
|-- Step 1: Initialize logging configuration
|-- Step 2: Get user inputs for the chosen outcome variable

Data Loading
|-- Step 3: Load master dataset
|-- Step 4: Load outcomes dataset

Data Merging
|-- Step 5: Merge master dataset with outcomes dataset

Preprocessing
|-- Step 6: Preprocess merged data using caching for efficiency

Create Demographic Subsets
|-- Step 7: Create demographic subsets using caching

Merge Demographic Data
|-- Step 8: Merge the demographic subsets

Parallel Processing of Subsets
|-- Step 9: Process each subset in parallel using ProcessPoolExecutor
    |-- Step 9.1: Initialize and train logistic model
    |-- Step 9.2: Feature selection and model fitting
    |-- Step 9.3: Find the best threshold
    |-- Step 9.4: Evaluate model

Memory Management
|-- Step 10: Clear memory after processing each subset

Profiling
|-- Step 11: Profile the pipeline to identify bottlenecks

Completion
|-- Step 12: Log the completion of the pipeline
