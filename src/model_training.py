from src.train_model import LogisticModel, NegativeBinomialModel, CoxProportionalHazard
from src.constants import EndpointType
import logging
import pandas as pd


def _resolve_model_class(endpoint_type: EndpointType | str) -> type:
    """
    Return the model class that handles the given endpoint type.

    Parameters:
    - endpoint_type: An EndpointType enum or its string value ('logical', 'integer', 'survival').

    Returns:
    - The model class to instantiate for each subset.

    Raises:
    - ValueError if the endpoint type is not supported.
    """
    # Built per call rather than at module level so patched model classes are picked up.
    model_by_endpoint = {
        EndpointType.LOGICAL: LogisticModel,
        EndpointType.SURVIVAL: CoxProportionalHazard,
        EndpointType.INTEGER: NegativeBinomialModel,
    }

    if isinstance(endpoint_type, str):
        try:
            endpoint_type = EndpointType(endpoint_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported endpoint type: {endpoint_type!r}")

    if endpoint_type not in model_by_endpoint:
        raise ValueError(f"Unsupported endpoint type: {endpoint_type!r}")

    return model_by_endpoint[endpoint_type]


def train_and_evaluate_models(
    merged_subsets: list[pd.DataFrame],
    id_column: str,
    selected_outcome: dict,
    processed_data_heldout: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train one model per subset and evaluate each on its subset and the held-out data.

    Parameters:
    - merged_subsets: The demographic subsets to train on, one model per subset.
    - id_column: The name of the participant ID column.
    - selected_outcome: Outcome config with 'endpointType' (EndpointType or its string value)
      and 'columnsToUse' (the outcome column names).
    - processed_data_heldout: The shared held-out DataFrame every model is evaluated on.

    Returns:
    - A DataFrame with one row per subset and MultiIndex columns
      (heldout|subset) × (predictions|evaluations).

    Raises:
    - ValueError if the endpoint type is not supported.
    """
    selectedModel = _resolve_model_class(selected_outcome['endpointType'])

    columns = pd.MultiIndex.from_product(
        [["heldout", "subset"], ["predictions", "evaluations"]],
        names=["Data Type", "Metric"]
    )
    results = pd.DataFrame(columns=columns)

    numOfSubsets = len(merged_subsets)

    for i, subset in enumerate(merged_subsets):
        logging.info(f"=== Subset {i + 1}/{numOfSubsets} | Train stage starting ===")

        outcomeModel = selectedModel(subset, id_column, selected_outcome['columnsToUse'])
        outcomeModel.selectFeatures()
        outcomeModel.train()

        logging.info(f"=== Subset {i + 1}/{numOfSubsets} | Train complete — Evaluate stage starting ===")

        results.loc[len(results)] = outcomeModel.evaluate(processed_data_heldout)

        logging.info(f"=== Subset {i + 1}/{numOfSubsets} | Evaluate stage complete ===")

    logging.info("All subsets processed. Predictions ready.")

    return results
