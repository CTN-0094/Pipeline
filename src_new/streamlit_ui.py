import streamlit as st
import pandas as pd
from data_cleaning import DataCleaner  # Import the DataCleaner class

st.title("Generalized Pipeline: Data Analysis & Cleaning Options")

# Step 1: Dataset Upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    cleaner = DataCleaner(data)  # Initialize DataCleaner with the uploaded data

    st.write("### Raw Data Preview")
    st.write(data.head())

    # Step 2: Data Analysis Options
    st.write("### Data Analysis")

    # Show summary statistics
    if st.checkbox("Show Summary Statistics"):
        st.write(cleaner.summary_statistics())

    # Show missing values report
    if st.checkbox("Show Missing Values Report"):
        st.write(cleaner.missing_values_report())

    # Show outlier summary
    if st.checkbox("Show Outlier Summary"):
        z_thresh = st.slider("Select Z-score threshold for outlier detection", 1, 5, 3)
        st.write(cleaner.outlier_summary(z_thresh=z_thresh))

    # Show data type summary
    if st.checkbox("Show Data Type Summary"):
        st.write(cleaner.data_type_summary())

    # Show correlation matrix
    if st.checkbox("Show Correlation Matrix"):
        st.write(cleaner.correlation_matrix())

    # Step 3: Data Cleaning Options
    st.write("### Data Cleaning Options")

    # Option to drop columns with high missing values
    drop_missing = st.checkbox("Drop columns with high missing values")
    if drop_missing:
        missing_threshold = st.slider("Select missing value threshold", 0.0, 1.0, 0.5)

    # Option to fill missing values
    fill_missing = st.checkbox("Fill missing values")
    if fill_missing:
        fill_method = st.selectbox("Select method for filling missing values", ["mean", "median", "mode", "constant"])
        if fill_method == "constant":
            fill_value = st.number_input("Constant value to fill missing values", value=0)

    # Option to remove outliers
    remove_outliers_option = st.checkbox("Remove outliers")
    if remove_outliers_option:
        outlier_z_thresh = st.slider("Select Z-score threshold for outlier removal", 1, 5, 3)

    # Option to scale numerical features
    scale_features_option = st.checkbox("Scale numerical features")
    if scale_features_option:
        scale_method = st.selectbox("Select scaling method", ["standard", "min-max"])

    # Option to encode categorical features
    encode_features_option = st.checkbox("Encode categorical features")
    if encode_features_option:
        encode_method = st.selectbox("Select encoding method", ["one-hot", "label"])

    # Option to drop duplicate rows
    drop_duplicates_option = st.checkbox("Drop duplicate rows")

    # Step 4: Apply Selected Cleaning Options
    if st.button("Apply Cleaning Options"):
        # Apply selected cleaning options using DataCleaner methods
        if drop_missing:
            cleaner = cleaner.drop_high_missing_columns(threshold=missing_threshold)
        
        if fill_missing:
            if fill_method == "constant":
                cleaner = cleaner.fill_missing_values(method="constant", fill_value=fill_value)
            else:
                cleaner = cleaner.fill_missing_values(method=fill_method)

        if remove_outliers_option:
            cleaner = cleaner.remove_outliers(z_thresh=outlier_z_thresh)

        if scale_features_option:
            cleaner = cleaner.scale_features(method=scale_method)

        if encode_features_option:
            cleaner = cleaner.encode_categorical_features(method=encode_method)

        if drop_duplicates_option:
            cleaner = cleaner.drop_duplicates()

        # Retrieve the cleaned data
        cleaned_data = cleaner.get_data()

        # Step 5: Display Cleaned Data Preview
        st.write("### Cleaned Data Preview")
        st.write(cleaned_data.head())
        
        # Option to download the cleaned dataset
        csv = cleaned_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
