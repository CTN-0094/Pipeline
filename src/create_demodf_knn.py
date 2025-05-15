# Import required libraries for data handling and modeling
import pandas as pd  # for handling DataFrames
from psmpy import PsmPy  # for propensity score matching using Python
import rpy2.robjects as robjects  # interface to R
from rpy2.robjects.packages import importr, PackageNotInstalledError  # R package management
from sklearn.model_selection import train_test_split  # unused, but typically for data splitting
from rpy2.robjects.conversion import localconverter  # context manager for data conversion
from rpy2.robjects import r, pandas2ri  # base R interface and pandas conversion tools

# Global flag to control printing of match summary from R
PRINT_SUMMARY = False

# Function to hold out a small test set with fixed majority/minority split
def holdOutTestData(df, columnToSplit='RaceEth', majorityValue=1, testCount=100, seed=42):
    majority_count = int(testCount * 0.58)  # majority sample size (e.g., 58%)
    minority_count = testCount - majority_count  # remainder for minority

    # Filter the majority group using the selected column
    majority_heldout_df = df[df[columnToSplit] == majorityValue]

    # Filter the minority group using the selected column
    minority_heldout_df = df[df[columnToSplit] != majorityValue]

    # Sample a subset from the majority group
    sample_majority_heldout = majority_heldout_df.sample(n=min(majority_count, len(majority_heldout_df)), random_state=seed)

    # Sample a subset from the minority group
    sample_minority_heldout = minority_heldout_df.sample(n=min(minority_count, len(minority_heldout_df)), random_state=seed)

    # Combine both samples to form the test set
    test_df = pd.concat([sample_majority_heldout, sample_minority_heldout]).reset_index(drop=True)

    # Exclude the test set from the original dataframe to form the training set
    train_df = df.drop(test_df.index)

    # Return both training and test sets
    return train_df, test_df

# Function to perform propensity score matching using an R backend
def propensityScoreMatch(df, columnToSplit='RaceEth', columnsToMatch=['age', 'is_female'], sampleSize=100):
    majorityValue = df[columnToSplit].value_counts().idxmax()
    df['is_minority'] = (df[columnToSplit] != majorityValue).astype(int)  # binary treatment indicator

    # Run optimal matching using R's MatchIt
    matched_participants = PropensityScoreMatchRMatchit(df, columnsToMatch, sampleSize)

    # For each column (treated, control_0, control_1), reshape into a 'who' column
    column_dfs = [matched_participants[[col]].rename(columns={col: 'who'}) for col in matched_participants.columns]

    # Merge each ID set with original data to recover matched participant rows
    matched_dfs = [pd.merge(col_df, df.drop(columns=['is_minority']), on='who', how='left') for col_df in column_dfs]

    # Return the three resulting matched groups (treated, control_0, control_1)
    return matched_dfs

# Create multiple overlapping subsets of matched data to support comparative analysis
def create_subsets(dfs, splits=11, sampleSize=100):
    # Generate subsets by blending varying amounts of treated/control samples
    subsets = [
        pd.concat(
            [dfs[0].iloc[:splitLen], dfs[1].iloc[splitLen:], dfs[2].iloc[:]],  # varying amounts of dfs[0] and dfs[1], full dfs[2]
            axis=0,
            ignore_index=True  # reset index after concat
        )
        for splitLen in range(0, sampleSize + 1, sampleSize // (splits - 1))  # loop in equal steps
    ]

    # Return all subset DataFrames
    merged_subsets = [demo_df for demo_df in subsets]
    return merged_subsets

# Alternative Python-based matcher using PsmPy instead of R (currently unused)
def PropensityScoreMatchPsmPy(df, idColumn, columnsToMatch, sampleSize):
    treatmentCol = 'is_minority'  # binary indicator for treatment

    # Identify columns not to use for matching
    columnsToExclude = list(df.columns.difference(columnsToMatch + [treatmentCol]).drop(idColumn))

    # Create PsmPy object for matching
    psm = PsmPy(df, treatment=treatmentCol, indx=idColumn, exclude=columnsToExclude)

    # Estimate propensity scores using logistic regression
    psm.logistic_ps()

    psm.knn_matched_12n(matcher='propensity_logit', how_many=2)

    matched_participants = psm.matched_ids.sample(n=sampleSize)

    return matched_participants


# Main matching function using R’s MatchIt package
def PropensityScoreMatchRMatchit(df, columnsToMatch, sampleSize):
    try:
        importr('MatchIt')  # Try to import R package
    except PackageNotInstalledError:
        print("MatchIt is not installed. Installing now...")
        r('install.packages("MatchIt", repos="http://cran.r-project.org")')
        importr('MatchIt')

    pandas2ri.activate()

  
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_data = robjects.conversion.py2rpy(df)
    robjects.globalenv['inData'] = r_data

    variables = " + ".join(columnsToMatch)

    robjects.r(f'''
        library(MatchIt)

        treated_subset <- inData[inData$is_minority == 1, ][1:{sampleSize}, ]
        control_subset <- inData[inData$is_minority == 0, ]

        subset_data <- rbind(treated_subset, control_subset)

        m.out1 <- matchit(
            is_minority ~ {variables},
            data = subset_data,
            method = "optimal",
            distance = "glm",
            link = "probit",
            ratio = 2
        )

        matched_data <- match.data(m.out1)
        matched_data <- matched_data[!is.na(matched_data$subclass), ]

        treated <- matched_data[matched_data$is_minority == 1, ]
        control <- matched_data[matched_data$is_minority == 0, ]

        matched_pairs <- merge(
            treated[, c("subclass", "who")],
            control[, c("subclass", "who")],
            by = "subclass",
            suffixes = c("_treated", "_control")
        )

        pair_df <- data.frame(
            treated_row = matched_pairs$who_treated,
            control_row = matched_pairs$who_control
        )
    ''')

    # ✅ New: handle case where R returned 0 rows
    if int(robjects.r('nrow(pair_df)')[0]) == 0:
        print("❌ MatchIt returned 0 matched pairs. Try lowering sample size or reviewing feature overlap.")
        return None

    # Convert matched pair DataFrame back to pandas
    pair_df_r = robjects.r('pair_df')
    with localconverter(robjects.default_converter + pandas2ri.converter):
        matched_data = robjects.conversion.rpy2py(pair_df_r)

    # Optional debug summary
    if PRINT_SUMMARY:
        summary = robjects.r('summary(m.out1, un = FALSE)')
        print("MatchIt Summary:", summary)

    matched_data['control_index'] = matched_data.groupby('treated_row').cumcount()
    final_matched_df = matched_data.pivot(index='treated_row', columns='control_index', values='control_row').reset_index()
    final_matched_df.columns = ['treated_row', 'control_row_0', 'control_row_1']

    return final_matched_df
