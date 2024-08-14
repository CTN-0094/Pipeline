import csv
import re
import os
from datetime import datetime

# Start time for performance tracking
start_time = datetime.now()

# Dictionary of hardcoded variables to include in the CSV output
# These are default values that will be used if not found in the log file
hardCodedVars = {
    "global_seed": "N/A",  # Placeholder for the global seed
    "outcomeType": "Binary",  # Type of outcome being analyzed
    "outcomeName": "",  # Placeholder for the outcome name
    "preProcessScriptName": "pipeline 7-2024",  # Name of the preprocessing script
    "modelScriptName": "TBD",  # Placeholder for the model script name
    "demoComparison": "Race: non hispanic white vs minority",  # Demographic comparison being analyzed
}

# Define the phrases you want to match in the log files using regular expressions
regex_patterns = [
    re.compile(r'demographic makeup:\s+(.*?)$'),         # Match demographic makeup
    re.compile(r'ROC AUC Score:\s+(\d+\.\d+)'),          # Match ROC AUC Score
    re.compile(r'\[\[\s*(\d+)\s+(\d+)\]'),               # Match the first line of the confusion matrix
    re.compile(r'\s*\[\s*(\d+)\s+(\d+)\]'),              # Match the second line of the confusion matrix
    re.compile(r'Precision:\s+(\d+\.\d+)'),              # Match Precision
    re.compile(r'Recall:\s+(\d+\.\d+)')                  # Match Recall
]

def parse_log_line(line, pattern_idx):
    """Parse a single line of the log file based on the current regex pattern."""
    match = regex_patterns[pattern_idx].search(line)
    if match:
        return match.groups()  # Return the matched groups from the regex
    return None

def scrape_log_to_csv(log_filepaths):
    """Scrape data from log files and write the results to a CSV file."""
    if not log_filepaths:
        print("No log files provided.")
        return

    # Define the LogOutput directory path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of the script
    log_output_dir = os.path.join(current_dir, 'logOutput')

    # Ensure the LogOutput directory exists
    os.makedirs(log_output_dir, exist_ok=True)

    # Generate a unique CSV filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSVPATH = os.path.join(log_output_dir, f"log_data_{timestamp}.csv") 

    with open(CSVPATH, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the CSV header
        csv_writer.writerow([
            'global_seed',
            'Outcome Type',
            'Outcome Name',
            'Pre-processing script name',
            'Model script name',
            'Demog Comparison',
            'Prop(Demog)',
            'TP',
            'TN',
            'FP',
            'FN',
            'Accuracy',
            'Precision',
            'Recall',
            'F1',
            'ROC AUC Score'
        ])

        # Process each log file
        for log_filepath in log_filepaths:
            with open(log_filepath, 'r') as log_file:
                # Extract the global seed from the first line of the log file
                line = log_file.readline()
                hardCodedVars["global_seed"] = line.split(":")[-1].strip()

                # Extract the outcome name from the second line of the log file
                line = log_file.readline()
                hardCodedVars["outcomeName"] = line.split(":")[-1].strip()

                # Process each subsequent line in the log file
                for i, line in enumerate(log_file):
                    csv_line = []  # Reset for each block of metrics
                    pattern_idx = 0

                    # Iterate over each line and try to match with the patterns
                    while pattern_idx < len(regex_patterns):
                        parsed_line = parse_log_line(line, pattern_idx)
                        if parsed_line:
                            csv_line.extend(parsed_line)
                            pattern_idx += 1
                            if pattern_idx >= len(regex_patterns):
                                # Ensure correct extraction for each component
                                prop_demog = csv_line[0]
                                tp = int(csv_line[2])
                                fn = int(csv_line[3])
                                fp = int(csv_line[4])
                                tn = int(csv_line[5])
                                precision = float(csv_line[6])
                                recall = float(csv_line[7])
                                roc_auc_score = float(csv_line[1])
                                accuracy = (tp + tn) / (tp + tn + fp + fn)
                                f1 = 2 * (precision * recall) / (precision + recall)

                                # Write the extracted data to the CSV file
                                csv_writer.writerow([
                                    hardCodedVars['global_seed'],
                                    hardCodedVars['outcomeType'],
                                    hardCodedVars['outcomeName'],
                                    hardCodedVars['preProcessScriptName'],
                                    hardCodedVars['modelScriptName'],
                                    hardCodedVars['demoComparison'],
                                    prop_demog,  # Demographic data
                                    tp,
                                    tn,
                                    fp,
                                    fn,
                                    accuracy,
                                    precision,
                                    recall,
                                    f1,
                                    roc_auc_score
                                ])
                                # Reset for the next set of metrics
                                break  # Exit the while loop to continue reading the log file
                        # Read the next line of the log file
                        line = next(log_file, None)
                        if line is None:
                            break  # End of file

# End time for performance tracking
end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(f"Elapsed time: {elapsed_time} seconds")
