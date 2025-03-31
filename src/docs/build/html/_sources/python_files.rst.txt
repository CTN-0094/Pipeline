Python Modules
==================

-----


train_model.py
---------------

This module defines classes for training models on various types of outcomes (binary and count-based) using logistic regression and negative binomial regression. These classes are integrated into the pipeline for training and evaluation.

OutcomeModel
^^^^^^^^^^^^
.. py:class:: OutcomeModel(data, target_column, seed=None)

   Base class for modeling an outcome from a dataset.

   :param data: The input dataset.
   :type data: pandas.DataFrame
   :param target_column: The name of the column to predict.
   :type target_column: str
   :param seed: Random seed for reproducibility.
   :type seed: int, optional

   .. py:method:: train()

      Placeholder method to train a model.

   .. py:method:: evaluate()

      Placeholder method to evaluate a model.

LogisticModel
^^^^^^^^^^^^^
.. py:class:: LogisticModel(data, target_column, Cs=[1.0], seed=None)

   Logistic regression model with L1 regularization for feature selection.

   :param data: The input dataset.
   :type data: pandas.DataFrame
   :param target_column: The name of the target column.
   :type target_column: str
   :param Cs: List of inverse regularization strengths.
   :type Cs: list
   :param seed: Random seed.
   :type seed: int, optional

   .. py:method:: feature_selection_and_model_fitting()

      Perform feature selection using L1 regularization and fit logistic regression model.

   .. py:method:: find_best_threshold()

      Determine the optimal classification threshold based on the proportion of positive outcomes.

   .. py:method:: train()

      Train the logistic regression model.

   .. py:method:: evaluateOverallTest()

      Evaluate the model on the overall test set.

   .. py:method:: evaluate()

      Evaluate the logistic model using chosen performance metrics.

   .. py:method:: _evaluateOnValidation()

      Internal method to evaluate performance on a validation split.

   .. py:method:: _countDemographic()

      Count demographic group membership for fairness evaluation.

NegativeBinomialModel
^^^^^^^^^^^^^^^^^^^^^
.. py:class:: NegativeBinomialModel(data, target_column, seed=None)

   Negative Binomial model for count-based outcomes using statsmodels.

   .. py:method:: train()

      Train a Negative Binomial regression model using statsmodels.

   .. py:method:: predict()

      Make predictions using the trained Negative Binomial model.


--------

create_demodf_knn.py
---------------------

This module provides tools for creating balanced demographic datasets using propensity score matching and data splitting techniques. It supports both Python-based (PsmPy) and R-based (MatchIt) methods for matching.

.. py:function:: holdOutTestData(df, testCount=100, seed=42)

   Hold out test data by sampling a fixed number of majority and minority cases.

   :param df: Full dataset.
   :type df: pandas.DataFrame
   :param testCount: Total number of test samples.
   :type testCount: int
   :param seed: Random seed.
   :type seed: int
   :return: Combined test set with both majority and minority samples.
   :rtype: pandas.DataFrame

.. py:function:: propensityScoreMatch(df)

   Perform a simple train/test split for race-ethnicity classification.

   :param df: Full dataset.
   :type df: pandas.DataFrame
   :return: Tuple of train and test sets.
   :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]

.. py:function:: create_subsets(df, demographic_col="RaceEth")

   Split dataset into majority and minority subsets based on a demographic column.

   :param df: Full dataset.
   :type df: pandas.DataFrame
   :param demographic_col: Column used for splitting groups.
   :type demographic_col: str
   :return: Tuple of (majority_df, minority_df)
   :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]

.. py:function:: PropensityScoreMatchPsmPy(df)

   Apply Propensity Score Matching using the PsmPy library.

   :param df: Full dataset.
   :type df: pandas.DataFrame
   :return: Matched dataset.
   :rtype: pandas.DataFrame

.. py:function:: PropensityScoreMatchRMatchit(df)

   Apply Propensity Score Matching using the R MatchIt package via rpy2.

   :param df: Full dataset.
   :type df: pandas.DataFrame
   :return: Matched dataset using R's MatchIt.
   :rtype: pandas.DataFrame

--------

preprocess.py
--------------

This module includes a `DataPreprocessor` class and various helper functions for transforming, cleaning, and preparing clinical and behavioral data for analysis.

DataPreprocessor
^^^^^^^^^^^^^^^^

.. py:class:: DataPreprocessor(dataframe)

   A class to preprocess pandas DataFrames by handling column drops and validation checks.

   :param dataframe: The pandas DataFrame to preprocess.
   :type dataframe: pandas.DataFrame

   .. py:method:: drop_columns_and_return(columns_to_drop)

      Drops specified columns from the DataFrame and returns the modified DataFrame. 
      Logs both successful drops and invalid column names.

      :param columns_to_drop: List of column names to drop.
      :type columns_to_drop: list of str
      :return: Modified DataFrame.
      :rtype: pandas.DataFrame


.. py:function:: convert_yes_no_to_binary(df, columns)

   Convert 'Yes'/'No' categorical values to 1/0 binary in specified columns.

   :param df: Input DataFrame.
   :param columns: List of column names to convert.
   :return: Updated DataFrame.

.. py:function:: process_tlfb_columns(df)

   Normalize TLFB (Timeline Follow-Back) columns using binary encoding.

   :param df: Input DataFrame.
   :return: Updated DataFrame.

.. py:function:: calculate_behavioral_columns(df)

   Generate and normalize behavioral columns like opioid use frequency.

   :param df: Input DataFrame.
   :return: Updated DataFrame.

.. py:function:: move_column_to_end(df, column_name)

   Move the specified column to the end of the DataFrame.

   :param df: Input DataFrame.
   :param column_name: Name of the column to move.
   :return: Updated DataFrame.

.. py:function:: rename_columns(df, rename_dict)

   Rename columns in the DataFrame using a provided mapping.

   :param df: Input DataFrame.
   :param rename_dict: Dictionary of old-to-new column names.
   :return: Updated DataFrame.

.. py:function:: transform_nan_to_zero_for_binary_columns(df, columns)

   Replace NaN values with 0 in binary columns.

   :param df: Input DataFrame.
   :param columns: List of column names.
   :return: Updated DataFrame.

.. py:function:: transform_and_rename_column(df, original_col, new_col)

   Rename a column and fill missing values with 0.

   :param df: Input DataFrame.
   :param original_col: Original column name.
   :param new_col: New column name.
   :return: Updated DataFrame.

.. py:function:: fill_nan_with_zero(df, columns)

   Fill NaNs with 0 for specified columns.

   :param df: Input DataFrame.
   :param columns: List of column names.
   :return: Updated DataFrame.

.. py:function:: transform_data_with_nan_handling(df, columns)

   Replace NaNs with 0 and standardize column values to 1.

   :param df: Input DataFrame.
   :param columns: List of column names.
   :return: Updated DataFrame.

.. py:function:: convert_uds_to_binary(df)

   Convert Urine Drug Screen (UDS) result columns from text to binary values.

   :param df: Input DataFrame.
   :return: Updated DataFrame.


-------

preprocess_pipeline.py
----------------------

This module provides a single entry point for preprocessing data within the modeling pipeline.


.. py:function:: preprocess_data(df)

   Preprocesses a dataset by cleaning, transforming, and formatting features for modeling.

   This function performs operations such as:
   - Dropping irrelevant or highly sparse columns
   - Converting categorical values to binary
   - Normalizing behavioral features
   - Handling missing values
   - Renaming columns for consistency
   - Converting drug test results to binary format

   :param df: The raw input DataFrame from the master dataset.
   :type df: pandas.DataFrame
   :return: Preprocessed DataFrame ready for modeling.
   :rtype: pandas.DataFrame
