import pandas as pd
from psmpy import PsmPy
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, PackageNotInstalledError
from sklearn.model_selection import train_test_split
import pandas as pd
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import r, pandas2ri



PRINT_SUMMARY = False



#make comment for what values in matrix mean
#control percentages of heldout data 58/42 majority/minority
def holdOutTestData(df, id_column, testCount = 100, columnToSplit='RaceEth', majorityValue=1, percentMajority = 58, seed=42):
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



def propensityScoreMatch(df, idColumn, columnToSplit='RaceEth', majorityValue=1, columnsToMatch = ['age', 'is_female'], sampleSize=500):
    #Propensity Score Match data
    df = df.copy()
    df.loc[:, 'is_minority'] = (df[columnToSplit] != majorityValue).astype(int)
    # Run propensity score matching
    matched_participants =  PropensityScoreMatchRMatchit(df, idColumn, columnsToMatch, sampleSize)
    column_dfs = [matched_participants[[col]].rename(columns={col: idColumn}) for col in matched_participants.columns]
    matched_dfs = [pd.merge(col_df, df.drop(columns=['is_minority']), on=idColumn, how='left') for col_df in column_dfs]
    #print("TESTING: ", matched_dfs[0])
    #matched_participants = pd.merge(matched_participants, df, on='who', how='left') 
    return matched_dfs


def create_subsets(dfs, splits=11, sampleSize=500):
    
    #load all 3 groups (minority + majority + majority) at different ratios for each split
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



def PropensityScoreMatchPsmPy(df, idColumn, columnsToMatch, sampleSize):
    treatmentCol = 'is_minority'
    columnsToExclude = list(df.columns.difference(columnsToMatch + [treatmentCol]).drop(idColumn))
    psm = PsmPy(df, treatment=treatmentCol, indx=idColumn, exclude=columnsToExclude)
    psm.logistic_ps()
    psm.knn_matched_12n(matcher='propensity_logit', how_many=2)
    matched_participants = psm.matched_ids.sample(n=sampleSize)
    return matched_participants



def PropensityScoreMatchRMatchit(df, idColumn, columnsToMatch, sampleSize):
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

        # Perform nearest neighbor matching
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
