from enum import Enum


class EndpointType(Enum):
    LOGICAL = 'logical'
    SURVIVAL = 'survival'
    INTEGER = 'integer'


# Canonical numeric encoding for the RaceEth column.
# Matches the mapping applied in DataPreprocessor.transform_data_with_nan_handling().
RACEETH_LABELS: dict[int, str] = {
    0: "Refused/Missing",
    1: "NHW",   # Non-Hispanic White
    2: "NHB",   # Non-Hispanic Black
    3: "Hisp",  # Hispanic
    4: "Other",
}

# NHW is the reference level, so it gets no indicator column.
RACEETH_REFERENCE_CODE: int = 1

# One indicator column per non-reference RaceEth code. Models use these rather
# than the raw code, which would impose a false ordering on nominal groups.
# The raw RaceEth column is kept for splitting, matching and demographics.
RACEETH_INDICATOR_COLUMNS: dict[int, str] = {
    0: "RaceEth_Missing",
    2: "RaceEth_NHB",
    3: "RaceEth_Hisp",
    4: "RaceEth_Other",
}
