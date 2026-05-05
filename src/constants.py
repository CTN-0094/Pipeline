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
