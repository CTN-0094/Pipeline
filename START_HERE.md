# Project Directory Structure

/Users/richeyjay/Desktop/Relapse_Pipeline/env/
│
├── run_pipelineV2.py                   # Main script to run the pipeline
│
├── data/
│   ├── master_data.csv                 # Master dataset containing all features and observations
│   └── all_binary_selected_outcomes.csv# Dataset containing selected binary outcomes
│
└── src/
    ├── __init__.py                     # Indicates that this directory is a Python package
    │
    ├── create_demodf_knn.py            # Module for creating demographic subsets using KNN
    ├── data_loading.py                 # Functions to load datasets from files
    ├── data_preprocessing.py           # Preprocessing functions for the dataset
    ├── demographic_handling.py         # Functions for creating and merging demographic subsets
    ├── evaluate_model.py               # Functions for evaluating the trained model
    ├── main.ipynb                      # Jupyter notebook for interactive data exploration or development
    ├── merge_demodf.py                 # Functions to merge demographic subsets
    ├── model_training.py               # Functions for training and evaluating the logistic model
    ├── preprocess_pipeline.py          # Pipeline for preprocessing data
    ├── preprocess.py                   # Preprocessing utility functions
    ├── train_model.py                  # Class for logistic model training and evaluation
    └── utils.py                        # Utility functions, including logging setup and user input handling



## Explanation of Each File

1. **`run_pipelineV2.py`**:
   - The main script to execute the data processing and model training pipeline. It orchestrates the loading, preprocessing, subsetting, and model training/evaluation steps.

2. **`data/`**:
   - **`master_data.csv`**: The main dataset containing all the features and observations.
   - **`all_binary_selected_outcomes.csv`**: A dataset containing the selected binary outcomes.

3. **`src/__init__.py`**:
   - An empty file that indicates the directory is a Python package.

4. **`src/create_demodf_knn.py`**:
   - Contains functions for creating demographic subsets using the K-Nearest Neighbors algorithm.

5. **`src/data_loading.py`**:
   - Functions to load the datasets from CSV files.

6. **`src/data_preprocessing.py`**:
   - Functions to preprocess the merged dataset, preparing it for further analysis and modeling.

7. **`src/demographic_handling.py`**:
   - Functions to create and merge demographic subsets from the processed data.

8. **`src/evaluate_model.py`**:
   - Functions to evaluate the performance of the trained models.

9. **`src/main.ipynb`**:
   - A Jupyter notebook for interactive data exploration, testing, and development.

10. **`src/merge_demodf.py`**:
    - Functions to merge demographic subsets created by `create_demodf_knn.py`.

11. **`src/model_training.py`**:
    - Functions to handle the training and evaluation of logistic regression models.

12. **`src/preprocess_pipeline.py`**:
    - The pipeline for preprocessing the data, integrating various preprocessing steps.

13. **`src/preprocess.py`**:
    - Utility functions for data preprocessing.

14. **`src/train_model.py`**:
    - Defines the `LogisticModel` class used for logistic regression model training and evaluation.

15. **`src/utils.py`**:
    - Utility functions including logging setup (`setup_logging`), user input handling (`get_outcome_choice`), and demographic inputs handling (`get_demographic_inputs`).

