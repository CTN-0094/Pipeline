# README
### DAB Website: https://ctn-0094.github.io/Pipeline/

## Overview

- Purpose: The purpose of this project is to establish a modular, scalable data pipeline for statistical modeling and machine learning on the CTN-0094 database. The pipeline will support various modeling strategies and evaluation metrics to deliver insights into the relationship between demographics and different target outcomes.
- Team: this work is led by Prof. Laura Brandt (clinical arm) and Prof. Gabriel Odom (computational arm); Mr. Ganesh Jainarain is the primary data scientist and statistical programmer.
- Funding: this work is towards the successful completion of "Towards a framework for algorithmic bias and fairness in predicting treatment outcome for opioid use disorder" (NIH AIM-AHEAD 1OT2OD032581-02-267) with contact PI Laura Brandt, City College of New York.

## Quick Start
Enter the command `python3 run_pipelineV2.py --help` for the list of arguments. Predictions, logs, and evaluations folders will be created in the directory specified by `-d`. When running multiple tests at once, specify a a minimum and maximum seed to loop through using the `-l` flag. The default outcomes when using the `-l` flag are all outcomes, if you would like to use only a subset of the outcomes, you can list them after the `-o` flag. 
- Example: `python3 run_pipelineV2.py -d "C:\Users\John\Desktop\Results" -l 5 10` loops through all integer seeds between 5 and 10 and saves the results folders in the specified path on the desktop.


## Step-by-Step Guide

### Step 0: Master Dataset
- **Task**: Keep the "master" dataset that was created by joining tables from the CTN-0094 database.
- **Note**: This dataset remains unaltered throughout the pipeline.

### Step 1: Sampling
- **Task**: Build a dataset of 1000 samples with a specified demographic distribution.
- **Methods**:
  - Random sampling
  - Partial matching
  - Sophisticated matching (to be implemented)

### Step 2: Data Pre-Processing
- **Task**: Perform standard feature engineering and data preparation.
- **Note**: This pre-processing script will likely remain consistent.

### Step 3: Join Dependent Variables
- **Task**: Join a chosen target variable (dependent variable) from the list of 11 in the Latin Square setup with the processed independent variables.
- **Selection**: Choose from 11 target variables identified by Laura and the team.

### Step 4: Machine Learning Model Selection
- **Task**: Select a machine learning model suited for the target variable and use it to predict target values.

### Step 5: Model Evaluation
- **Task**: Evaluate the machine learning model using various metrics:
  - AUC
  - F1 score
  - RMSE
  - Fairness (currently undefined)

- **Output**: Return a tuple containing the demographic makeup, target variable, machine learning model, and metrics.

### Step 6: Iterative Design Points
- **Task**: Repeat Steps 1-5 over various design points to explore the tuple space.

## Target Outcome Buckets
1. **Binary**  
2. **Count (with a fixed max)**
3. **Proportion**

## Model Prioritization

The statistical models will be implemented according to the following priority:

1. **Logistic LASSO (Binary Outcomes)**  
   - Port existing code into the pipeline.
   - Save the `.job` file trained on the full cohort for later.

2. **Random Forests (All Outcomes)**  

3. **Negative Binomial Regression (Count Outcomes)**  

4. **Sigmoidal Regression (Proportion Outcomes)**  

5. **Beta Regression (Proportion Outcomes)**  

## Future Direction

The immediate objective is to build a proof-of-concept pipeline with logistic LASSO, then extend it to incorporate random forests. Other models can be implemented as required in the future.
