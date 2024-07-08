from train_model import LogisticModel
import pandas as pd 

def evaluate_on_new_data(test_data_path, model_path):
    # Assuming test_data_path is a path to a CSV file with your test data
    test_data = pd.read_csv(test_data_path)
    target_column = 'your_target_column_name_here'  # Adjust this

    # Initialize the LogisticModel class (no need to pass data here if we're only loading the model)
    logistic_model = LogisticModel(data=None, target_column=target_column)

    # Load the trained model
    logistic_model.load_model(model_path)

    # Assuming your test data has the same structure as the training data and includes the target column
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Update the model's X_test and y_test with the new data
    logistic_model.X_test = X_test
    logistic_model.y_test = y_test

    # evaluation methods 
    logistic_model.generate_and_decode_prediction_table()
    logistic_model.visualize_feature_importance()
    logistic_model.visualize_confusion_matrix()
    logistic_model.visualize_roc_curve()
    