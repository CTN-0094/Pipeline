README
======

Overview
--------

- **Purpose**: The purpose of this project is to establish a modular, scalable data pipeline for statistical modeling and machine learning on the CTN-0094 database. The pipeline will support various modeling strategies and evaluation metrics to deliver insights into the relationship between demographics and different target outcomes.
- **Team**: This work is led by Prof. Laura Brandt (clinical arm) and Prof. Gabriel Odom (computational arm); Mr. Ganesh Jainarain is the primary data scientist and statistical programmer.
- **Funding**: This work is towards the successful completion of *"Towards a framework for algorithmic bias and fairness in predicting treatment outcome for opioid use disorder"* (NIH AIM-AHEAD 1OT2OD032581-02-267) with contact PI Laura Brandt, City College of New York.



Quick Start
-----------

To see the list of arguments available, run:

.. code-block:: python

   python3 run_pipelineV2.py --help

Predictions, logs, and evaluations folders will be created in the directory specified by the `-d` flag. When running multiple tests, you can loop through a range of seeds using the `-l` flag. By default, all outcomes will be considered unless specified using the `-o` flag.

**Example Usage**:

.. code-block:: python

   python3 run_pipelineV2.py -d "C:\\Users\\John\\Desktop\\Results" -l 5 10

This command loops through all integer seeds between 5 and 10 and saves the results in the specified directory.

Step-by-Step Guide
------------------

Step 0: Master Dataset
^^^^^^^^^^^^^^^^^^^^^^

- **Task**: Maintain the "master" dataset created by joining tables from the CTN-0094 database.
- **Note**: This dataset remains unchanged throughout the pipeline.

Step 1: Sampling
^^^^^^^^^^^^^^^^

- **Task**: Generate a dataset of 1000 samples with a specified demographic distribution.
- **Methods**:
  - Random sampling
  - Partial matching
  - Sophisticated matching (future implementation)

Step 2: Data Pre-Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Task**: Apply standard feature engineering and data preparation.
- **Note**: The pre-processing script will likely remain stable.

Step 3: Join Dependent Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Task**: Merge a chosen target variable (dependent variable) with the processed independent variables.
- **Selection**: Choose from 11 predefined target variables.

Step 4: Machine Learning Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Task**: Select an appropriate machine learning model and use it to predict target values.

Step 5: Model Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^

- **Task**: Evaluate model performance using the following metrics:
  - AUC
  - F1 score
  - RMSE
  - Fairness (currently undefined)

- **Output**: Return a tuple containing demographic composition, target variable, machine learning model, and metrics.

Step 6: Iterative Design Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Task**: Repeat Steps 1–5 across various design points.

Target Outcome Buckets
----------------------

1. **Binary**  
2. **Count (with a fixed max)**  
3. **Proportion**  

Model Prioritization
--------------------

The models will be implemented in the following order:

1. **Logistic LASSO (Binary Outcomes)**
   - Port existing code into the pipeline.
   - Save the `.job` file trained on the full cohort.

2. **Negative Binomial Regression (Count Outcomes)**

3. **Sigmoidal Regression (Proportion Outcomes)**

4. **Beta Regression (Proportion Outcomes)**

Future Direction
----------------

The immediate goal is to develop a proof-of-concept pipeline using logistic LASSO, followed by an expansion to random forests. Additional models will be integrated as needed.

References
==========

**Luo SX, Feaster DJ, Liu Y et al. _Individual‑Level Risk Prediction of Return to Use During Opioid Use Disorder Treatment_. JAMA Psychiatry. 2024;81(1):45–56. doi:10.1001/jamapsychiatry.2023.3596**

Multicenter decision‑analytic prediction model using CTN trial data.

.. image:: https://img.shields.io/badge/View–JAMA%20Psychiatry-blue
   :target: https://jamanetwork.com/journals/jamapsychiatry/fullarticle/2810311
   :alt: View full article on JAMA Psychiatry
