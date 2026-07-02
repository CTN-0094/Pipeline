"""Cohort construction for the fairness pipeline.

This module builds the demographic cohorts the models train and evaluate on:

1. ``holdOutTestData`` carves off a stratified held-out evaluation set.
2. ``propensityScoreMatch`` matches the minority group to majority controls
   (via R's ``MatchIt``) so the two groups are comparable on chosen covariates.
3. ``create_subsets`` sweeps the majority/minority ratio across a ladder of
   subsets, which is how the pipeline measures performance shifts as cohort
   demographics change.

The "majority" group is defined by ``columnToSplit == majorityValue`` (by
default ``RaceEth == 1``, Non-Hispanic White); every other value is treated as
minority.
"""

import pandas as pd
from psmpy import PsmPy
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, PackageNotInstalledError
from sklearn.model_selection import train_test_split
import pandas as pd
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import r, pandas2ri



PRINT_SUMMARY = False



def holdOutTestData(
    df: pd.DataFrame,
    id_column: str,
    testCount: int = 100,
    columnToSplit: str = 'RaceEth',
    majorityValue: int = 1,
    percentMajority: int = 58,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a stratified held-out evaluation set with a fixed group balance.

    The held-out set holds a constant ``percentMajority`` / minority ratio so
    that evaluation demographics stay fixed regardless of how the training
    cohort is later rebalanced. Sampling uses the module-local ``seed`` (default
    42) rather than the pipeline's run seed, so the held-out rows are identical
    across multi-seed runs.

    Args:
        df: Full preprocessed dataset.
        id_column: Name of the participant id column (e.g. ``"who"``); used to
            exclude held-out rows from the training frame.
        testCount: Total number of rows in the held-out set.
        columnToSplit: Column defining the majority/minority groups.
        majorityValue: Value of ``columnToSplit`` marking the majority group.
        percentMajority: Percent of the held-out set drawn from the majority
            group; the remainder is drawn from the minority group.
        seed: Random state for reproducible sampling.

    Returns:
        A ``(train_df, test_df)`` tuple where ``test_df`` is the held-out set and
        ``train_df`` is every remaining row (matched by ``id_column``). If a
        group has fewer rows than requested, the whole group is taken.
    """
    majority_count = percentMajority * testCount // 100
    minority_count = testCount - majority_count
    majority_count = percentMajority * testCount // 100
    minority_count = testCount - majority_count

    # Separate DataFrames
    majority_heldout_df = df[df[columnToSplit] == majorityValue]
    minority_heldout_df = df[df[columnToSplit] != majorityValue]

    # Sample from each group
    sample_majority_heldout = majority_heldout_df.sample(n=min(majority_count, len(majority_heldout_df)), random_state=seed)
    sample_minority_heldout = minority_heldout_df.sample(n=min(minority_count, len(minority_heldout_df)), random_state=seed)

    # Combine the samples
    test_df = pd.concat([sample_majority_heldout, sample_minority_heldout]).reset_index(drop=True)
    train_df = df[~df[id_column].isin(test_df[id_column])]
    return train_df, test_df



def propensityScoreMatch(
    df: pd.DataFrame,
    idColumn: str,
    columnToSplit: str = 'RaceEth',
    majorityValue: int = 1,
    columnsToMatch: list[str] = ['age', 'is_female'],
    sampleSize: int = 500,
) -> list[pd.DataFrame]:
    """Match minority participants to majority controls via propensity scoring.

    Flags each row as minority (``columnToSplit != majorityValue``), runs 1:2
    optimal matching in R (see :func:`PropensityScoreMatchRMatchit`), then
    reassembles the result into one DataFrame per matched column: the treated
    (minority) group and its two majority control groups. Each returned frame is
    the matched ids joined back to the full feature set.

    Args:
        df: Full preprocessed dataset (one row per participant).
        idColumn: Name of the participant id column used for matching/merging.
        columnToSplit: Column defining the majority/minority groups.
        majorityValue: Value of ``columnToSplit`` marking the majority group.
        columnsToMatch: Covariates to match on (e.g. ``age``, ``is_female``).
        sampleSize: Number of minority (treated) participants to match.

    Returns:
        A list of three DataFrames ``[treated, control_0, control_1]``, each
        with ``sampleSize`` rows and the full feature columns (the internal
        ``is_minority`` flag is dropped before returning).
    """
    df = df.copy()
    df.loc[:, 'is_minority'] = (df[columnToSplit] != majorityValue).astype(int)
    # Run propensity score matching in R and get back the matched id triples
    matched_participants =  PropensityScoreMatchRMatchit(df, idColumn, columnsToMatch, sampleSize)
    # Rebuild each matched column (treated / control_0 / control_1) into a full
    # feature frame by joining the matched ids back onto the source data.
    column_dfs = [matched_participants[[col]].rename(columns={col: idColumn}) for col in matched_participants.columns]
    matched_dfs = [pd.merge(col_df, df.drop(columns=['is_minority']), on=idColumn, how='left') for col_df in column_dfs]
    return matched_dfs


def create_subsets(dfs: list[pd.DataFrame], splits: int = 11, sampleSize: int = 500) -> list[pd.DataFrame]:
    """Build a demographic ladder that sweeps the majority/minority ratio.

    Given the three matched groups from :func:`propensityScoreMatch`
    (``dfs = [treated_minority, control_majority, control_majority]``), this
    produces ``splits`` cohorts of constant total size that progressively swap
    minority rows for majority rows. The first subset is majority-heavy and the
    last is all minority + one majority control group, letting downstream models
    measure how performance changes with cohort composition.

    Args:
        dfs: The ``[minority, majority, majority]`` matched frames.
        splits: Number of subsets (rungs) in the ladder.
        sampleSize: Size of each matched group; also the step denominator.

    Returns:
        A list of ``splits`` concatenated DataFrames, one per ratio rung.
    """
    # Vary how many minority (dfs[0]) vs majority (dfs[1]) rows are taken at each
    # rung, while always including the second majority control group (dfs[2]).
    subsets = [
        pd.concat(
            [dfs[0].iloc[:splitLen], dfs[1].iloc[splitLen:], dfs[2].iloc[:]], 
            axis=0, 
            ignore_index=True
        )
        
        for splitLen in range(0, sampleSize + 1, sampleSize // (splits-1))
    ]

    merged_subsets = [
        demo_df 
        for demo_df in subsets
    ]

    return merged_subsets



def PropensityScoreMatchPsmPy(
    df: pd.DataFrame,
    idColumn: str,
    columnsToMatch: list[str],
    sampleSize: int,
) -> pd.DataFrame:
    """Match minority to majority participants using the Python ``psmpy`` library.

    A pure-Python alternative to :func:`PropensityScoreMatchRMatchit`: it fits a
    logistic propensity model, performs 1:2 nearest-neighbour matching on the
    propensity logit, and returns a random ``sampleSize`` of matched id groups.

    Note:
        Not currently used by the pipeline, which relies on the R ``MatchIt``
        implementation for optimal matching. Retained as a dependency-light
        fallback.

    Args:
        df: Dataset containing an ``is_minority`` treatment column.
        idColumn: Name of the participant id column.
        columnsToMatch: Covariates to match on.
        sampleSize: Number of matched id groups to sample from the result.

    Returns:
        A DataFrame of matched participant ids (treated plus their controls).
    """
    treatmentCol = 'is_minority'
    columnsToExclude = list(df.columns.difference(columnsToMatch + [treatmentCol]).drop(idColumn))
    psm = PsmPy(df, treatment=treatmentCol, indx=idColumn, exclude=columnsToExclude)
    psm.logistic_ps()
    psm.knn_matched_12n(matcher='propensity_logit', how_many=2)
    matched_participants = psm.matched_ids.sample(n=sampleSize)
    return matched_participants



def PropensityScoreMatchRMatchit(
    df: pd.DataFrame,
    idColumn: str,
    columnsToMatch: list[str],
    sampleSize: int,
) -> pd.DataFrame:
    """Run 1:2 optimal propensity matching in R via ``MatchIt`` (through rpy2).

    The DataFrame is passed to R, where ``matchit`` fits a probit-link GLM
    propensity model on ``columnsToMatch`` and optimally matches the first
    ``sampleSize`` minority (treated) rows to two majority controls each. The
    matched pairs are reshaped so every treated participant is aligned with its
    two controls on the same row.

    Installs the ``MatchIt`` R package on first use if it is not already
    available. Set the module-level ``PRINT_SUMMARY`` flag to also print R's
    balance summary.

    Args:
        df: Dataset containing a binary ``is_minority`` treatment column plus
            ``idColumn`` and every column in ``columnsToMatch``.
        idColumn: Name of the participant id column.
        columnsToMatch: Covariates entered into the propensity model.
        sampleSize: Number of treated (minority) rows to match.

    Returns:
        A DataFrame with columns ``[treated_row, control_row_0, control_row_1]``
        holding the participant ids for each matched triple.
    """
    try:
        # Check if MatchIt is installed
        importr('MatchIt')
    except PackageNotInstalledError:
        print("MatchIt is not installed. Installing now...")
        # Install MatchIt
        r('install.packages("MatchIt", repos="http://cran.r-project.org")')
        # Verify installation
        importr('MatchIt')
        print("MatchIt installed successfully.")
    pandas2ri.activate()
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_data = robjects.conversion.py2rpy(df)
    robjects.globalenv['inData'] = r_data

    variables = " + ".join(columnsToMatch)

    robjects.r(f'''
        library(MatchIt)

        # Limit the number of treated data points (e.g., first 50 treated rows)
        treated_subset <- inData[inData$is_minority == 1, ][1:{sampleSize}, ]  # Adjust the subset size as needed
        control_subset <- inData[inData$is_minority == 0, ]

        subset_data <- rbind(treated_subset, control_subset)

        # Perform optimal matching
        m.out1 <- matchit(
            is_minority ~ {variables},
            data = subset_data,
            method = "optimal",
            distance = "glm",
            link = "probit",
            ratio = 2 # Specify 1 treated : 2 controls
        )
        
        # Extract matched data
        matched_data <- match.data(m.out1)

        # Filter matched data by subclass
        matched_data <- matched_data[!is.na(matched_data$subclass), ]

        # Split into treated and control groups
        treated <- matched_data[matched_data$is_minority == 1, ]
        control <- matched_data[matched_data$is_minority == 0, ]

        # Create matched pairs based on subclass
        matched_pairs <- merge(
            treated[, c("subclass", \"{idColumn}\")],
            control[, c("subclass", \"{idColumn}\")],
            by = "subclass",
            suffixes = c("_treated", "_control")
        )

        # Create final DataFrame of treated and control row indices
        pair_df <- data.frame(
            treated_row = matched_pairs${idColumn}_treated,
            control_row = matched_pairs${idColumn}_control
        )
    ''')

    # Retrieve the paired matches DataFrame from R
    pair_df_r = robjects.r('pair_df')
    with localconverter(robjects.default_converter + pandas2ri.converter):
        matched_data = robjects.conversion.rpy2py(pair_df_r)

    if(PRINT_SUMMARY):
        summary = robjects.r('summary(m.out1, un = FALSE)')
        print("SUMMARY: ", summary)

    matched_data['control_index'] = matched_data.groupby('treated_row').cumcount()
    final_matched_df = matched_data.pivot(index='treated_row', columns='control_index', values='control_row').reset_index()

    final_matched_df.columns = ['treated_row', 'control_row_0', 'control_row_1']

    return final_matched_df