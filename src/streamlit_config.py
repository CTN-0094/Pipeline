import streamlit as st
import yaml

st.title("Pipeline Configuration")

# Data Paths
st.header("Data Paths")
features_file = st.file_uploader("Upload Features File", type="csv")
outcomes_file = st.file_uploader("Upload Outcomes File", type="csv")

# Outcome Column
st.header("Outcome Configuration")
outcome_column = st.text_input("Enter the Outcome Column Name", "")

# Model Selection
st.header("Model Configuration")
model_type = st.selectbox("Select Model Type", ["logistic_regression", "decision_tree", "random_forest"])

# Model Hyperparameters
st.subheader("Hyperparameters")
hyperparameters = {}
if model_type == "logistic_regression":
    hyperparameters["C"] = st.number_input("Regularization Strength (C)", value=1.0, min_value=0.01, step=0.01)
elif model_type in ["decision_tree", "random_forest"]:
    hyperparameters["max_depth"] = st.number_input("Max Depth", value=5, min_value=1, step=1)

# Preprocessing Options
st.header("Preprocessing Configuration")
scale_features = st.checkbox("Scale Features", value=True)
feature_selection = st.selectbox("Feature Selection Method", ["L1", "None", "Manual"])
manual_features = []
if feature_selection == "Manual":
    manual_features = st.text_input("Enter Manual Features (comma-separated)", "").split(",")

# Metrics
st.header("Metrics Configuration")
selected_metrics = st.multiselect("Select Metrics to Calculate", ["accuracy", "precision", "recall", "roc_auc"])

# Output Settings
st.header("Output Settings")
log_file = st.text_input("Log File Path", "output/log.txt")
results_csv = st.text_input("Results CSV Path", "output/results.csv")

# Save Configuration
if st.button("Save Configuration"):
    config = {
        "data_paths": {
            "features": features_file.name if features_file else "",
            "outcomes": outcomes_file.name if outcomes_file else ""
        },
        "outcome": {
            "column_name": outcome_column
        },
        "model": {
            "type": model_type,
            "hyperparameters": hyperparameters
        },
        "preprocessing": {
            "scale_features": scale_features,
            "feature_selection": feature_selection,
            "manual_features": manual_features
        },
        "metrics": selected_metrics,
        "output": {
            "log_file": log_file,
            "save_results_csv": results_csv
        }
    }

    # Save the configuration to a YAML file
    with open("config.yml", "w") as file:
        yaml.dump(config, file)

    st.success("Configuration saved to config.yml")
