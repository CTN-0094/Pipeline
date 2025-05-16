Data Overview
=============

This project relies on several key datasets derived from the CTN-0094 database and various transformation stages in the pipeline. Below is a description of each dataset used throughout the pipeline, including purpose and example columns.

Master Dataset
--------------

**Filename**: `master_data.csv`

- **Purpose**: The foundational dataset containing all raw independent variables and features extracted from CTN-0094.
- **Example Columns**:
  - `age`, `race`, `education`, `pain`, `depression`, `ftnd`, `TLFB_Alcohol_Count`

Merged Data
-----------

**Filename**: `merged_data.csv`

- **Purpose**: A version of the master dataset where the `ctn0094_relapse_event` binary outcome is appended.
- **Notes**: Used for binary classification models on relapse.
- **Example Columns**:
  - All columns from `master_data.csv` + `ctn0094_relapse_event`

Outcomes Merged Dataset
-----------------------

**Filename**: `outcomes_merged_dataset.csv`

- **Purpose**: Contains both predictors and a wide variety of outcome variables used across various modeling tasks.
- **Example Outcome Columns**:
  - `Ab_krupitskyA_2011`, `Ab_ling_1998`, `Rs_johnson_1992`, `Rd_kostenB_1993`, `AbT_mokri_2016`

CTN-0094 Outcomes
-----------------

**Filename**: `outcomesCTN0094.csv`

- **Purpose**: A standalone file that contains all known outcome measures from the CTN-0094 dataset.
- **Types of Outcomes**: Abstinence (Ab), Retention (Rs), Dropout Rate (Rd), among others.
- **Example Columns**:
  - `Ab_ctnNinetyFour_2023`, `AbT_shufman_1994`, `Rd_strang_2019`, `RsE_ctnFiftyOne_2018`

All Outcome Selections
----------------------

**Filename**: `all_outcome_selections.csv`

- **Purpose**: Subset of outcomes selected for model testing, spanning multiple outcome categories.
- **Example Columns**:
  - `Ab_ctnNinetyFour_2023`, `Rd_kostenB_1993`, `RsT_lee_2016`

Master Outcome Selections
-------------------------

**Filename**: `master_outcome_selections.csv`

- **Purpose**: Combines all outcome columns with the cleaned features used in modeling.
- **Example Columns**:
  - Includes demographic and drug usage variables + outcome variables like `RsT_ctnFiftyOne_2018`

Binary Outcome Selections
-------------------------

**Filename**: `binary_outcome_selections.csv`

- **Purpose**: Outcomes that are binary (e.g., yes/no, true/false).
- **Example Columns**:
  - `Ab_krupitskyA_2011`, `Rd_kostenB_1993`, `Rs_johnson_1992`

All Binary Selected Outcomes
----------------------------

**Filename**: `all_binary_selected_outcomes.csv`

- **Purpose**: Subset of the merged dataset that includes only binary outcomes.
- **Example Columns**:
  - `ctn0094_relapse_event`, `Ab_ling_1998`, `Rs_krupitsky_2004`

Other Outcome Selections
------------------------

**Filename**: `other_outcome_selections.csv`

- **Purpose**: Collection of outcome variables that are count-based or numeric (e.g., session count, time until dropout).
- **Example Columns**:
  - `AbT_mokri_2016`, `RsT_ctnFiftyOne_2018`

Dataset Usage Notes
-------------------

- All datasets use `who` as a unique identifier for participants.
- Some datasets may have `NaN` values, especially in derived outcomes or less common drug use metrics.
- The pipeline joins or filters these datasets at various stages depending on the outcome type and modeling goal.

