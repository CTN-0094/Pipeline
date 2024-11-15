import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore

class DataCleaner:
    def __init__(self, data):
        """
        Initializes the DataCleaner with a DataFrame.
        
        Parameters:
        - data (DataFrame): The dataset to be cleaned and analyzed.
        """
        self.data = data.copy()  # Make a copy to avoid modifying the original data

    # Analysis Methods
    def summary_statistics(self):
        """Returns summary statistics for numerical columns in the dataset."""
        return self.data.describe()

    def missing_values_report(self):
        """Returns a report of missing values for each column."""
        missing_report = self.data.isnull().mean() * 100  # Percentage of missing values
        return missing_report[missing_report > 0].sort_values(ascending=False)

    def outlier_summary(self, z_thresh=3):
        """
        Returns a count of outliers in each numerical column based on a Z-score threshold.
        
        Parameters:
        - z_thresh (float): The Z-score threshold to identify outliers.
        
        Returns:
        - Series: Number of outliers per column.
        """
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        outlier_counts = ((np.abs(zscore(self.data[numerical_cols])) > z_thresh).sum(axis=0))
        return outlier_counts[outlier_counts > 0]

    def data_type_summary(self):
        """Returns a summary of data types in the dataset."""
        return self.data.dtypes.value_counts()

    def correlation_matrix(self):
        """Returns the correlation matrix of numerical features in the dataset."""
        return self.data.corr()

    # Data Cleaning Methods
    def drop_high_missing_columns(self, threshold=0.5):
        """Drops columns with missing values above a specified threshold."""
        missing_ratio = self.data.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > threshold].index
        self.data.drop(columns=columns_to_drop, inplace=True)
        return self

    def fill_missing_values(self, method="mean", fill_value=None):
        """Fills missing values in the dataset using the specified method."""
        for column in self.data.columns:
            if self.data[column].isnull().any():
                if method == "mean":
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif method == "median":
                    self.data[column].fillna(self.data[column].median(), inplace=True)
                elif method == "mode":
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                elif method == "constant" and fill_value is not None:
                    self.data[column].fillna(fill_value, inplace=True)
        return self

    def remove_outliers(self, z_thresh=3):
        """Removes rows that contain outliers based on a Z-score threshold."""
        z_scores = self.data.apply(zscore)
        self.data = self.data[(z_scores < z_thresh).all(axis=1)]
        return self

    def scale_features(self, method="standard"):
        """Scales numerical features using the specified method."""
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        numerical_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])
        return self

    def encode_categorical_features(self, method="one-hot"):
        """Encodes categorical features using the specified method."""
        categorical_features = self.data.select_dtypes(include=['object']).columns
        if method == "one-hot":
            self.data = pd.get_dummies(self.data, columns=categorical_features)
        elif method == "label":
            for col in categorical_features:
                self.data[col] = self.data[col].astype('category').cat.codes
        return self

    def drop_duplicates(self):
        """Drops duplicate rows in the dataset."""
        self.data.drop_duplicates(inplace=True)
        return self

    def get_data(self):
        """Returns the cleaned data."""
        return self.data
