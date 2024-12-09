import pandas as pd
from psmpy import PsmPy

def create_demographic_dfs(df, columnToSplit='RaceEth', majorityValue=1, idColumn = 'who', sampleSize=500, splits=11, columnsToMatch = ['age', 'is_female']):
    
    columnsToExclude = list(df.columns.difference(columnsToMatch).drop(idColumn))
    df['is_minority'] = (df[columnToSplit] != majorityValue).astype(int)

    # Run propensity score matching
    psm = PsmPy(df, treatment='is_minority', indx=idColumn, exclude=columnsToExclude)
    psm.logistic_ps()
    psm.knn_matched_12n(matcher='propensity_logit', how_many=2)
    matched_participants = psm.matched_ids.sample(n=sampleSize)

    #load all 3 groups (minority + majority + majority) at different ratios for each split
    subsets = [
        pd.DataFrame(
            list(matched_participants.iloc[:splitLen, 0]) + 
            list(matched_participants.iloc[splitLen:, 1]) + 
            list(matched_participants.iloc[:, 2]), 
            columns=[idColumn]
        )
        
        for splitLen in range(0, sampleSize + 1, sampleSize // (splits-1))
    ]


    merged_subsets = [
        pd.merge(demo_df, df, on='who', how='left') 
        for demo_df in subsets
    ]

    return merged_subsets
