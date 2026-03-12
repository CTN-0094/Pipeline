Python Modules
==================

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

CoxProportionalHazard
^^^^^^^^^^^^^^^^^^^^^
.. py:class:: CoxProportionalHazard(data, target_column, seed=None)

   Cox Proportional Hazards model for time-to-event (survival) analysis using the `lifelines` package.

   :param data: Input dataset containing features and event/time columns.
   :type data: pandas.DataFrame
   :param target_column: List with two elements: [duration_column, event_column].
   :type target_column: list of str
   :param seed: Random seed for reproducibility.
   :type seed: int, optional

   .. py:method:: train()

      Fit a Cox Proportional Hazards model using lifelines' `CoxPHFitter`.

   .. py:method:: predict()

      Return model predictions (placeholder — not implemented in full).

   .. py:method:: _evaluateOnValidation(X, y, id)

      Evaluate model on the validation set using Concordance Index.

   .. py:method:: selectFeatures()

      Use Lasso-based feature selection for survival outcomes.

BetaRegression
^^^^^^^^^^^^^^
.. py:class:: BetaRegression(data, target_column, seed=None)

   Beta regression model for modeling outcomes constrained between 0 and 1, using `statsmodels`.

   :param data: The input dataset with features and the beta-distributed target.
   :type data: pandas.DataFrame
   :param target_column: The name of the outcome column to predict.
   :type target_column: str
   :param seed: Random seed.
   :type seed: int, optional

   .. py:method:: train()

      Fit a Beta regression model using `statsmodels.othermod.betareg.BetaModel`.

   .. py:method:: predict()

      Return model predictions (placeholder — not implemented in full).

   .. py:method:: _evaluateOnValidation(X, y, id)

      Evaluate model performance using MSE, MAE, RMSE, Pearson R, and McFadden R².

   .. py:method:: selectFeatures()

      Perform Lasso-based feature selection for beta regression.


--------

create_demodf_knn.py
---------------------

This module provides tools for creating balanced demographic datasets using propensity score matching and data splitting techniques. It supports both Python-based (PsmPy) and R-based (MatchIt) methods for matching.

.. py:function:: holdOutTestData(df, id_column, testCount=100, columnToSplit='RaceEth', majorityValue=1, percentMajority=58, seed=42)

   Hold out a fixed, demographically stratified test set from the full dataset.

   :param df: Full dataset.
   :type df: pandas.DataFrame
   :param id_column: Name of the unique identifier column.
   :type id_column: str
   :param testCount: Total number of holdout samples. Default 100.
   :type testCount: int
   :param columnToSplit: Column used to define majority/minority groups. Default ``'RaceEth'``.
   :type columnToSplit: str
   :param majorityValue: Value in ``columnToSplit`` that identifies the majority group. Default ``1``.
   :type majorityValue: int
   :param percentMajority: Percentage of holdout that should be majority group (0–100). Default ``58``.
   :type percentMajority: int
   :param seed: Random seed for reproducibility. Default ``42``.
   :type seed: int
   :return: Tuple of (train_df, holdout_df).
   :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]

.. py:function:: propensityScoreMatch(df, idColumn, columnToSplit='RaceEth', majorityValue=1, columnsToMatch=['age', 'is_female'], sampleSize=500)

   Perform propensity score matching to create demographically balanced paired subsets.

   :param df: Full dataset.
   :type df: pandas.DataFrame
   :param idColumn: Name of the unique identifier column.
   :type idColumn: str
   :param columnToSplit: Column used to define majority/minority groups.
   :type columnToSplit: str
   :param majorityValue: Value identifying the majority group.
   :type majorityValue: int
   :param columnsToMatch: Covariates used for matching.
   :type columnsToMatch: list of str
   :param sampleSize: Number of treated samples to match against.
   :type sampleSize: int
   :return: List of matched DataFrames (treated, control_0, control_1).
   :rtype: list of pandas.DataFrame

.. py:function:: create_subsets(dfs, splits=11, sampleSize=500)

   Create a sequence of training subsets at incrementally varying majority/minority ratios.

   Each subset shifts the proportion of treated (minority) vs matched control (majority) samples
   across ``splits`` steps, enabling evaluation of model performance across demographic compositions.

   :param dfs: List of three DataFrames — [treated, control_0, control_1].
   :type dfs: list of pandas.DataFrame
   :param splits: Number of ratio steps to generate. Default ``11``.
   :type splits: int
   :param sampleSize: Total sample size per subset. Default ``500``.
   :type sampleSize: int
   :return: List of DataFrames, one per demographic ratio split.
   :rtype: list of pandas.DataFrame

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

      Drop specified columns from the DataFrame in place. Silently skips any
      column names that are not present.

      :param columns_to_drop: List of column names to drop.
      :type columns_to_drop: list of str

   .. py:method:: convert_yes_no_to_binary()

      Convert all columns whose values are exclusively ``'Yes'``/``'No'`` (and NaN)
      to binary ``1``/``0``. Other columns are left untouched.

   .. py:method:: process_tlfb_columns(specified_tlfb_columns)

      Aggregate all TLFB columns not in ``specified_tlfb_columns`` into a new
      ``TLFB_Other`` column, then drop those unspecified columns.

      :param specified_tlfb_columns: TLFB columns to retain individually.
      :type specified_tlfb_columns: list of str

   .. py:method:: calculate_behavioral_columns()

      Derive ``Homosexual_Behavior`` (based on ``msm_npt`` and ``Sex``) and
      ``Non_monogamous_Relationships`` (based on ``txx_prt``) and append them
      to the DataFrame.

   .. py:method:: move_column_to_end(column_names)

      Reorder the DataFrame so the specified columns appear last. Ignores any
      column names not present in the DataFrame.

      :param column_names: Column(s) to move to the end.
      :type column_names: list of str

   .. py:method:: rename_columns()

      Apply the hardcoded rename mapping in place:
      ``Sex`` → ``is_female``, ``job`` → ``unemployed``,
      ``is_living_stable`` → ``unstableliving``.

   .. py:method:: transform_nan_to_zero_for_binary_columns()

      For every column that contains NaN values and has unique non-NaN values
      of exactly ``[0, 1]``, fill NaN with ``0``.

   .. py:method:: transform_and_rename_column(original_column_name, new_column_name)

      Convert a column to binary (``1`` where non-null, ``0`` where null) and
      rename it in place, preserving its position.

      :param original_column_name: Name of the column to transform.
      :type original_column_name: str
      :param new_column_name: Replacement column name.
      :type new_column_name: str

   .. py:method:: fill_nan_with_zero(column_name)

      Fill NaN values in the specified column with ``0``. If the column does
      not exist the call is a no-op.

      :param column_name: Name of the column to fill.
      :type column_name: str

   .. py:method:: transform_data_with_nan_handling()

      Apply categorical-to-numeric mappings for ``Sex``, ``education``,
      ``marital``, ``job``, ``is_living_stable``, ``race``, ``XTRT``,
      ``RaceEth``, and ``pain``. Columns absent from the DataFrame are
      skipped without error.

   .. py:method:: convert_uds_to_binary()

      Convert all ``UDS``-prefixed columns to binary: values ``> 0`` become
      ``1``, values ``== 0`` stay ``0``.


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

model_training.py
------------------

This module provides the primary interface for training and evaluating outcome models in the pipeline. Depending on the selected outcome type (logical, integer, or survival), it dynamically loads the appropriate model class (Logistic Regression, Negative Binomial Regression, Cox Proportional Hazards, or Beta Regression). Each model is trained and evaluated on one or more data subsets and held-out validation data.

.. py:function:: train_and_evaluate_models(merged_subsets, selected_outcome, processed_data_heldout)

   Train and evaluate models on each demographic or data subset and return evaluation results.

   This function dynamically selects the correct model type based on the `endpointType` of the selected outcome. It then loops through each data subset, trains the selected model, and evaluates performance on both the subset and a held-out dataset.

   :param merged_subsets: A list of DataFrames representing stratified or demographically-split training datasets.
   :type merged_subsets: list of pandas.DataFrame

   :param selected_outcome: A dictionary containing the outcome column name(s) and the type of model to use.
   :type selected_outcome: dict
     - **columnsToUse**: list of str — target variable columns.
     - **endpointType**: Enum — one of `EndpointType.LOGICAL`, `EndpointType.SURVIVAL`, or `EndpointType.INTEGER`.

   :param processed_data_heldout: The held-out dataset used for validation.
   :type processed_data_heldout: pandas.DataFrame

   :return: A multi-indexed pandas DataFrame with predictions and evaluation metrics for both the held-out and subset data.
   :rtype: pandas.DataFrame

   .. note::
      Logging is extensively used to track training and evaluation progress for each subset. Evaluation metrics vary depending on the model type (e.g., accuracy and ROC for classification, RMSE and McFadden R² for regression).

run_pipelineV2.py
=================

This is the main pipeline orchestrator script for training, evaluating, and profiling statistical and machine learning models across demographic subsets using the CTN-0094 dataset. It supports multiple model types including logistic regression, negative binomial regression, survival analysis (Cox), and beta regression.

The script handles argument parsing, data loading, preprocessing, subset generation, model training, evaluation, and CSV logging of all results.

Functions
---------

.. py:function:: main()

   Entry point for the pipeline. Parses arguments, initializes outcome and seed configurations, and runs profiling or standard pipeline execution for each outcome and seed.

.. py:function:: argument_handler()

   Parse command-line arguments including seed range, outcome name, output directory, and profiling method.

   :return: A tuple of (loop range, outcomes, output directory, profiling flag).
   :rtype: Tuple

.. py:function:: initialize_pipeline(selected_outcome)

   Load and merge the demographic and outcome datasets, apply preprocessing, and prepare the data for modeling.

   :param selected_outcome: A dictionary defining the outcome variable and endpoint type.
   :type selected_outcome: dict
   :return: Preprocessed dataset ready for modeling.
   :rtype: pandas.DataFrame

.. py:function:: run_pipeline(processed_data, seed, selected_outcome, directory)

   Executes the core pipeline logic for one run: splits data, performs matching, creates subsets, trains and evaluates models, and writes predictions and evaluations to CSV.

   :param processed_data: Cleaned and merged input dataset.
   :type processed_data: pandas.DataFrame
   :param seed: Random seed for reproducibility.
   :type seed: int
   :param selected_outcome: Dictionary describing the outcome and model type.
   :type selected_outcome: dict
   :param directory: Output path for saving logs and results.
   :type directory: str

.. py:function:: save_evaluations_to_csv(results, seed, selected_outcome, directory, name)

   Save evaluation metrics for all subsets and held-out predictions into a CSV file. Automatically adjusts headers based on model type.

   :param results: Dictionary of evaluation results from each subset.
   :type results: dict
   :param seed: The random seed used for training.
   :type seed: int
   :param selected_outcome: The outcome configuration dict.
   :type selected_outcome: dict
   :param directory: Directory to save output CSVs.
   :type directory: str
   :param name: Subfolder name for organizing evaluation files.
   :type name: str

.. py:function:: save_predictions_to_csv(data, seed, selected_outcome, directory, name)

   Save prediction scores for each individual across subsets and held-out data.

   :param data: Prediction tuples (id, score) across subsets.
   :type data: list
   :param seed: The random seed used for training.
   :type seed: int
   :param selected_outcome: The outcome configuration dict.
   :type selected_outcome: dict
   :param directory: Output directory for saving results.
   :type directory: str
   :param name: Folder name under which to store predictions.
   :type name: str

Globals
-------

.. py:data:: AVAILABLE_OUTCOMES

   A predefined list of outcomes from the CTN-0094 dataset, each with its name, outcome column(s), and associated `EndpointType`.

   Used for automatic selection of outcomes when not specified via command-line arguments.

CLI Usage Example
-----------------

Run the pipeline for a specific outcome and seed range:

.. code-block:: bash

   python run_pipelineV2.py --loop 42 45 --outcome Ab_ling_1998 --dir logs/run_test --prof simple

Or run all outcomes with profiling off:

.. code-block:: bash

   python run_pipelineV2.py


--------

validate.py
-----------

This module validates a dataset before passing it to a model, checking that required
columns exist, data types are correct, and outcome values match the expected format for
the chosen model type.

.. py:function:: validate_dataset_for_model(df, model_type, outcome_col, time_col=None)

   Validate a dataset against the requirements of the specified model type.

   Accepts either a string or an ``EndpointType`` enum value for ``model_type``.
   Raises ``ValueError`` with a descriptive message if any check fails.

   :param df: The dataset to validate.
   :type df: pandas.DataFrame
   :param model_type: Model type — one of ``'logical'``, ``'integer'``, ``'survival'``,
      or the corresponding ``EndpointType`` enum member.
   :type model_type: str or EndpointType
   :param outcome_col: Name of the outcome column to validate.
   :type outcome_col: str
   :param time_col: Name of the time column. Required for survival models. Default ``None``.
   :type time_col: str, optional
   :raises ValueError: If the outcome column is missing, values are of the wrong type,
      or required survival columns are absent or non-numeric.

   **Validation rules by model type:**

   - ``logical`` — outcome must contain only ``0`` and ``1``.
   - ``integer`` — outcome must have an integer dtype.
   - ``survival`` — requires a numeric ``time_col`` and a binary event column.

--------

constants.py
------------

This module defines the ``EndpointType`` enum used throughout the pipeline to identify
which type of statistical model should be applied to a given outcome.

.. py:class:: EndpointType

   Enum of supported outcome/model types.

   .. py:attribute:: LOGICAL
      :value: 'logical'

      Binary outcome (0/1). Use with ``LogisticModel``.

   .. py:attribute:: INTEGER
      :value: 'integer'

      Count-based outcome. Use with ``NegativeBinomialModel``.

   .. py:attribute:: SURVIVAL
      :value: 'survival'

      Time-to-event outcome. Use with ``CoxProportionalHazard``.

