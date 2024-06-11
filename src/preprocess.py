# src/preprocess.py
import pandas as pd
import numpy as np

class DataPreprocessor:

    def __init__(self, dataframe):
        """
        Initializes the DataPreprocessor with a pandas DataFrame.
        """
        self.dataframe = dataframe


    def drop_columns_and_return(self, columns_to_drop):
        """
        Drops specified columns from the DataFrame, prints out the dropped columns, and handles potential errors.

        Parameters:
        - columns_to_drop: A list of strings, where each string is a column name to be dropped.

        Returns:
        - The modified DataFrame with specified columns dropped.
        """
        try:
            # Ensure columns_to_drop are in the DataFrame
            valid_columns_to_drop = [col for col in columns_to_drop if col in self.dataframe.columns]
            invalid_columns = [col for col in columns_to_drop if col not in self.dataframe.columns]

            # Print out valid and invalid columns to drop
            if valid_columns_to_drop:
                print("Dropping columns:", valid_columns_to_drop)
            if invalid_columns:
                print("Invalid columns not found in DataFrame:", invalid_columns)

            # Drop the specified columns and update the instance's DataFrame
            self.dataframe.drop(columns=valid_columns_to_drop, inplace=True)
            
            # Display a snippet of the new DataFrame
            print("Snippet of the new DataFrame after dropping columns:")
            print(self.dataframe)

        except KeyError as e:
            print(f"Column error: {e}. Please check the columns you are trying to drop.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # Example usage:
        # # Assuming df is your original DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(df)
        # columns_to_drop = ['pain_when', 'is_smoker', 'per_day', 'max', 'amount', 'depression', 'anxiety', 'schizophrenia', 
        #                 "days", "max", "amount", "cocaine_inject_days", "speedball_inject_days", "opioid_inject_days", 
        #                 "speed_inject_days", "UDS_Alcohol_Count", 'UDS_Mdma/Hallucinogen_Count']
        # preprocessor.drop_columns_and_return(columns_to_drop)


    def convert_yes_no_to_binary(self):
        """
        Converts columns in the DataFrame that contain only 'Yes' or 'No' values (and NaNs)
        to binary (1 for 'Yes', 0 for 'No').

        The DataFrame is modified in place, and the method returns None.
        """
        # Iterate over each column in the DataFrame
        for column in self.dataframe.columns:
            # Check if the column contains only 'Yes', 'No', and NaN values
            unique_values = self.dataframe[column].dropna().unique()
            if set(unique_values) <= {'Yes', 'No'}:
                # Convert 'Yes' to 1 and 'No' to 0
                self.dataframe[column] = self.dataframe[column].map({'Yes': 1, 'No': 0})

        # # Example usage:
        # # Assuming df is your original DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(df)

        # # Convert 'Yes'/'No' columns to binary directly within the class's DataFrame
        # preprocessor.convert_yes_no_to_binary()

        # # Optionally, display the modified DataFrame to verify the changes
        # print(preprocessor.dataframe.head())


    def process_tlfb_columns(self, specified_tlfb_columns):
        """
        Processes TLFB columns in the DataFrame by summing unspecified TLFB columns into a 'TLFB_Other' column,
        removing these unspecified columns, without duplicating 'TLFB_Other'. The 'TLFB_Other' column is placed
        appropriately among TLFB columns.

        Parameters:
        - specified_tlfb_columns: A list of TLFB column names to be kept and not summed into 'TLFB_Other'.
        """
        # Identify TLFB columns not listed in the specified list for 'Other' calculations
        tlfb_other_columns = [col for col in self.dataframe.columns if col.startswith('TLFB_') and col not in specified_tlfb_columns]

        # If 'TLFB_Other' already exists, update it directly, otherwise calculate and insert it
        self.dataframe['TLFB_Other'] = self.dataframe[tlfb_other_columns].sum(axis=1)
        
        # Remove the 'other' TLFB columns from the DataFrame
        self.dataframe.drop(columns=tlfb_other_columns, inplace=True)

        if 'TLFB_Other' not in specified_tlfb_columns:
            # Temporarily store TLFB_Other data
            tlfb_other_data = self.dataframe['TLFB_Other']

            # Remove the existing 'TLFB_Other' to avoid confusion in reordering
            self.dataframe.drop(columns=['TLFB_Other'], inplace=True)

            # Calculate the position for 'TLFB_Other' based on specified columns
            position = max(self.dataframe.columns.get_loc(col) for col in specified_tlfb_columns if col in self.dataframe.columns) + 1

            # Reinsert 'TLFB_Other' at the calculated position
            self.dataframe.insert(position, 'TLFB_Other', tlfb_other_data)

        # # Example usage:
        # # Assuming transformed_df is your DataFrame containing TLFB data loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(transformed_df)
        # specified_tlfb_columns = [
        #     'TLFB_Alcohol_Count', 'TLFB_Amphetamine_Count', 'TLFB_Cocaine_Count',
        #     'TLFB_Heroin_Count', 'TLFB_Benzodiazepine_Count', 'TLFB_Opioid_Count',
        #     'TLFB_THC_Count', 'TLFB_Methadone_Count', 'TLFB_Buprenorphine_Count'
        # ]
        # preprocessor.process_tlfb_columns(specified_tlfb_columns)

        # # Optionally, display the modified DataFrame to verify the changes
        # print(preprocessor.dataframe.head())

    
    def calculate_behavioral_columns(self):
        """
        Calculates and adds 'Homosexual_Behavior' and 'Non_monogamous_Relationships' columns to the DataFrame.
        
        'Homosexual_Behavior' is calculated based on 'msm_npt' and 'Sex' columns, where it's set to NaN if 'msm_npt'
        is NaN, otherwise it checks if 'msm_npt' > 0 and the participant is male.
        
        'Non_monogamous_Relationships' is calculated based on 'txx_prt' column, where it's set to NaN if 'txx_prt' is NaN,
        otherwise it checks if 'txx_prt' > 1.
        """
        # Calculate 'Homosexual_Behavior'
        self.dataframe['Homosexual_Behavior'] = np.where(
            self.dataframe['msm_npt'].isna(), 
            np.nan, 
            (self.dataframe['msm_npt'] > 0) & (self.dataframe['Sex'] == 'male')
        )

        # Calculate 'Non_monogamous_Relationships'
        self.dataframe['Non_monogamous_Relationships'] = np.where(
            self.dataframe['txx_prt'].isna(), 
            np.nan, 
            self.dataframe['txx_prt'] > 1
        )

        # # Example usage:
        # # Assuming transformed_df is your DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(transformed_df)
        # preprocessor.calculate_behavioral_columns()

        # # Optionally, display the modified DataFrame to verify the changes
        # print(preprocessor.dataframe.head())


    def move_column_to_end(self, column_name):
        """
        Moves a specified column to the end of the DataFrame, if the column exists.

        Parameters:
        - column_name: The name of the column to move to the end of the DataFrame.
        """
        # Check if the specified column is in the DataFrame
        if column_name in self.dataframe.columns:
            # Get a list of all columns excluding the specified column
            columns_except_target = [col for col in self.dataframe.columns if col != column_name]
            
            # Add the specified column to the end of the list
            reordered_columns = columns_except_target + [column_name]
            
            # Reorder the DataFrame according to the new columns order
            self.dataframe = self.dataframe[reordered_columns]
            
        else:
            # Optionally, you can print a message if the column is not found
            print(f"Column '{column_name}' not found in the DataFrame.")

        # '''Example usage:'''
        # # Assuming transformed_df is your DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(transformed_df)
        # column_to_move = 'ctn0094_relapse_event'

        # # Move the specified column to the end using the class method
        # preprocessor.move_column_to_end(column_to_move)

        # # Optionally, display the modified DataFrame to verify the changes
        # print(preprocessor.dataframe.head())
        
    def rename_columns(self):
        """
        Renames specified columns in the DataFrame to new names.
        
        The DataFrame is modified in place, reflecting the new column names.
        """
        # Define a dictionary mapping old column names to new names
        new_column_names = {
            'Sex': 'is_female',
            'job': 'unemployed',
            'is_living_stable': 'unstableliving'
        }
        
        # Rename the columns using the dictionary
        self.dataframe.rename(columns=new_column_names, inplace=True)

        # # Example usage:
        # # Assuming transformed_data_full is your DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(transformed_data_full)

        # # Rename specified columns using the class method
        # preprocessor.rename_columns()

        # # Optionally, display the modified DataFrame to verify the changes
        # print(preprocessor.dataframe.head())

    
    def transform_nan_to_zero_for_binary_columns(self):
        """
        Transforms NaN values to 0 for columns that have NaN values and have exactly two unique values, [1, 0].
        This modification is performed in place.
        """
        # Iterate through each column in the DataFrame
        for column in self.dataframe.columns:
            # Check if the column has NaN values
            if self.dataframe[column].isna().sum() > 0:
                # Check if the unique values in the column, excluding NaN, are exactly [0, 1]
                unique_values = np.sort(self.dataframe[column].dropna().unique())
                if np.array_equal(unique_values, [0, 1]):
                    # Transform NaN values to 0 for the column
                    self.dataframe[column] = self.dataframe[column].fillna(0)

        # # Example usage:
        # # Assuming transformed_data_full is your DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(transformed_data_full)
        # preprocessor.transform_nan_to_zero_for_binary_columns()

        # # Optionally, you can check the transformation by using:
        # print(preprocessor.dataframe[['Homosexual_Behavior', 'has_cocaine_dx', 'Non_monogamous_Relationships', 'has_cannabis_dx']].isna().sum())
    def transform_and_rename_column(self, original_column_name, new_column_name):
        """
        Transforms a specified column to binary based on its non-null values and renames it
        while preserving its position in the DataFrame.

        Parameters:
        - original_column_name: The name of the column to be transformed and renamed.
        - new_column_name: The new name for the transformed column.
        """
        # Step 1: Transform the specified column to binary based on non-null values
        self.dataframe[original_column_name] = self.dataframe[original_column_name].notna().astype(int)
        
        # Step 2: Rename the column while keeping its position
        columns = list(self.dataframe.columns)
        index = columns.index(original_column_name)
        columns[index] = new_column_name
        self.dataframe.columns = columns

        # # Example usage:
        # # Assuming transformed_data_full is your DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(transformed_data_full)
        # preprocessor.transform_and_rename_column('heroin_inject_days', 'rbsivheroin')

        # # Optionally, display the modified DataFrame to verify the changes
        # print(preprocessor.dataframe.head())
    def fill_nan_with_zero(self, column_name):
        """
        Fills NaN values in the specified column with 0. The operation is performed in place on the class's DataFrame.

        Parameters:
        - column_name: The name of the column where NaN values should be filled with 0.
        """
        # Ensure the column exists in the DataFrame
        if column_name in self.dataframe.columns:
            self.dataframe[column_name] = self.dataframe[column_name].fillna(0)
        else:
            print(f"Column '{column_name}' not found in the DataFrame.")

        # # Example usage:
        # # Assuming transformed_data_full is your DataFrame loaded into a pandas DataFrame
        # preprocessor = DataPreprocessor(transformed_data_full)
        # preprocessor.fill_nan_with_zero('ftnd')

        # # Optionally, display the modified DataFrame to verify the changes
        # print(preprocessor.dataframe.head())

        # # Display the new DataFrame
        # transformed_data_full_2    
    def transform_data_with_nan_handling(self):
        """
        Transforms the data with handling for NaN values, ensuring each transformation
        is only applied if the relevant column exists.
        """
        # Use the instance's dataframe
        transformed_data = self.dataframe.copy()

        # Sex to binary, with NaN handled by setting to 0 (assuming male for simplicity)
        if 'Sex' in transformed_data.columns:
            transformed_data['Sex'] = transformed_data['Sex'].apply(lambda x: 1 if x == 'female' else 0)
        else:
            print("'Sex' column not found. Skipping this transformation.")

        # Education mapping
        edu_mapping = {
            'Less than HS': 1,
            'HS/GED': 2,
            'More than HS': 3
        }
        if 'education' in transformed_data.columns:
            transformed_data['education'] = transformed_data['education'].map(edu_mapping).fillna(0)
        else:
            print("'education' column not found. Skipping this transformation.")

        # Marital status mapping
        marital_mapping = {
            'Never married': 2,
            'Married or Partnered': 3,
            'Separated/Divorced/Widowed': 4
        }
        if 'marital' in transformed_data.columns:
            transformed_data['marital'] = transformed_data['marital'].map(marital_mapping).fillna(1)
        else:
            print("'marital' column not found. Skipping this transformation.")

        # Job to binary mapping
        if 'job' in transformed_data.columns:
            transformed_data['job'] = transformed_data['job'].apply(
                lambda x: 0 if x == 'Full Time' or x == 'Part Time' else 1).fillna(1)
        else:
            print("'job' column not found. Skipping this transformation.")

        # Handling is_living_stable, assuming 1 is stable, so inverse for 'unstableliving'
        if 'is_living_stable' in transformed_data.columns:
            transformed_data['is_living_stable'] = transformed_data['is_living_stable'].apply(
                lambda x: 0 if x == 1 else 1).fillna(1)
        else:
            print("'is_living_stable' column not found. Skipping this transformation.")

        # Map 'race' with numeric encoding, including a category for 'Refused/missing'
        race_mapping = {
            'White': 1,
            'Black': 2,
            'Other': 3,
            'Refused/missing': 0
        }
        if 'race' in transformed_data.columns:
            transformed_data['race'] = transformed_data['race'].map(race_mapping).fillna(-1)
        else:
            print("'race' column not found. Skipping this transformation.")

        # Map 'XTRT' with numeric encoding
        xtrt_mapping = {
            'CTN30BUP': 1,
            'CTN51BUP': 2,
            'CTN51NTX': 3,
            'CTN27BUP': 4,
            'CTN27MET': 5
        }
        if 'XTRT' in transformed_data.columns:
            transformed_data['XTRT'] = transformed_data['XTRT'].map(xtrt_mapping).fillna(-1)
        else:
            print("'XTRT' column not found. Skipping this transformation.")

        # Numeric encoding for RaceEth
        race_Eth_mapping = {
            'NHW': 1,  # Non-Hispanic White
            'NHB': 2,  # Non-Hispanic Black
            'Hisp': 3,  # Hispanic
            'Other': 4,  # Other races or mixed
            'Refused/missing': 0  # Handling missing values explicitly
        }
        if 'RaceEth' in transformed_data.columns:
            transformed_data['RaceEth'] = transformed_data['RaceEth'].map(race_Eth_mapping).fillna(0)
        else:
            print("'RaceEth' column not found. Skipping this transformation.")

        # Pain mapping
        pain_mapping = {
            'No Pain': 0,
            'Severe Pain': 1,
            'Very mild to Moderate Pain': 1,
            'Missing': 0  # Categorize 'Missing' explicitly as 'No Pain'
        }
        if 'pain' in transformed_data.columns:
            transformed_data['pain'] = transformed_data['pain'].map(pain_mapping).fillna(0)
        else:
            print("'pain' column not found. Skipping this transformation.")

        # Finally, update the instance's dataframe
        self.dataframe = transformed_data
        
    def convert_uds_to_binary(self):
        """
        Converts UDS columns to binary format. Columns representing drug counts
        are set to 1 if the count is greater than 0, and to 0 otherwise.
        """
        for col in self.dataframe.columns:
            if col.startswith('UDS'):
                self.dataframe[col] = self.dataframe[col].apply(lambda x: 1 if x > 0 else 0)

# # Example usage:
# # Load your dataset into a DataFrame
# df = pd.read_csv('/path/to/your/dataset.csv')

# # Initialize the DataPreprocessor with your DataFrame
# preprocessor = DataPreprocessor(dataframe=df)

# # Apply preprocessing and feature engineering methods
# preprocessor.drop_columns_and_return(columns_to_drop=['example_column'])
# # Continue with other processing methods as needed...
