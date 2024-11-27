import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.feature_selection import VarianceThreshold
import logging
from typing import Optional, Union

class DataCleaner:
    def __init__(self, data):
        """
        Initializes the DataCleaner with a DataFrame.
        
        Parameters:
        - data (DataFrame): The dataset to be cleaned and analyzed.
        """
        if isinstance(data, pd.DataFrame):
            # If data is already a DataFrame, use it directly
            self.data = data.copy()
        elif isinstance(data, str): # If data is a file path
            try:
                # Attempt to read ther file into a DataFrame
                self.data = pd.read_csv(data)
                logging.info(f"Data loadedd successfully from file: {data}")
            except Exception as e:
                raise ValueError(f"Failed to read the file as a Dataframe. Error: {e}")
        else:
            # Raise an error for unsupported types
            raise TypeError("Input must be a pandas DataFrame or a file path to a CSV.")

    # Data Analysis Methods
    def summary_statistics(self) -> pd.DataFrame:
        """
        Returns summary statistics for numerical columns in the dataset.

        Returns:
            pd.DataFrame: Summary statistics (mean, std, min, max, etc.) for numerical columns.
        """
        try:
            logging.info("Generating summary statistics.")
            return self.data.describe()
        except Exception as e:
            logging.error(f"Error in summary_statistics: {e}")
            return pd.DataFrame()
        
    def missing_values_report(self) -> pd.Series:
        """
        Returns a report of missing values as percentages for each column.

        Returns:
            pd.Series: Percentage of missing values per column, sorted in descending order.
        """
        try:
            logging.info("Calculating missing values report.")
            return self.data.isnull().mean().sort_values(ascending=False) * 100
        except Exception as e:
            logging.error(f"Error in missing_values_report: {e}")
            return pd.Series(dtype=float)
    
    def drop_high_missing_columns(self, threshold: float = 0.5) -> "DataCleaner":
        """
        Drops columns with missing values above a specified threshold.

        Args:
            threshold (float): The percentage of missing values above which columns are dropped (0-1).

        Returns:
            DataCleaner: The updated DataCleaner instance.
        """
        try:
            if not (0 <= threshold <= 1):
                raise ValueError("Threshold must be between 0 and 1.")
            cols_to_drop = self.data.columns[self.data.isnull().mean() > threshold]
            self.data.drop(columns=cols_to_drop, inplace=True)
            logging.info(f"Dropped columns with missing values above {threshold}: {list(cols_to_drop)}")
            return self
        except Exception as e:
            logging.error(f"Error in drop_high_missing_columns: {e}")
            return self
        
    def fill_missing_values(self, method: str = "mean", fill_value: Optional[Union[int, float, str]] = None) -> "DataCleaner":    
        try:
            for column in  self.data.columns:
            # Check if the column has missing values
                if self.data[column].isnull().any():
                    if method == "mean":
                        # Fill with the column's mean
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    elif method == "median":
                        # Fill with the column's median
                        self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                    elif method == "constant":
                        # Ensure a value is provided for constant fill
                        if fill_value is None:
                            raise ValueError("Fill value must be provided for 'constant' method.")
                        self.data[column].fillna(fill_value, inplace=True)
                    else:
                        # Raise an error for unsupported methods
                        raise ValueError("Invalid method. Use 'mean', 'median', 'mode', or 'constant'.")
            logging.info(f"Filled missing values using method '{method}'.")
            return self
        except Exception as e:
            logging.error(f"Error in fill_missing_values: {e}")
            return self
        
    
    
    def drop_duplicates(self) -> "DataCleaner":
        """
        Drop duplicate rows from the dataset.

        Returns:
            DataCleaner: The updated DataCleaner instance.
        """

        try:
            # Get the original number of rows
            original_count = len(self.data)
            # Drop duplicate rows
            self.data.drop_duplicates(inplace=True)
            # Log the number of rows removed
            logging.info(f"Dropped {original_count - len(self.data)} duplicate rows.")
            return self
        except Exception as e:
            # Log any errors that occur
            logging.error(f"Error in drop_duplicates: {e}")
            return self
        
    def scale_features(self, method: str = "standard") -> "DataCleaner":
        """
        Scale numerical features using the specified method.

        Args:
            method (str): Scaling method ("standard" for StandardScaler, "min-max" for MinMaxScaler).

        Returns:
            DataCleaner: The updated DataCleaner instance.
        """
        try:
            # Select the appropriate scaler based on the method
            scaler = StandardScaler() if method == "standard" else MinMaxScaler()
            # Identify numerical columns to scale
            numerical_features = self.data.select_dtypes(include=np.number).columns
            # Apply scaling to numerical columns
            self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])
            logging.info(f"Scaled numerical features using '{method}' scaling.")
            return self
        except Exception as e:
            # Log any errors that occur
            logging.error(f"Error in scale_features: {e}")
            return self
    
    def get_data(self) -> pd.DataFrame:
        """
        Return the cleaned dataset.

        Returns:
            pd.Datafram: The cleaned dataset.
        """
        # Return the cleaned DataFrame
        return self.data







