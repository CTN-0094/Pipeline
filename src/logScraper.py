import csv
import re
import os
from datetime import datetime

# Start time for performance tracking
start_time = datetime.now()

# Dictionary of hardcoded variables to include in the CSV output
hardCodedVars = {
    "global_seed": "N/A",
    "outcomeType": "Binary",
    "outcomeName": "",
    "preProcessScriptName": "pipeline 7-2024",
    "modelScriptName": "TBD",
    "demoComparison": "Race: non hispanic white vs minority",
}

# Define the phrases you want to match
regex_patterns = [
    re.compile(r'demographic makeup:\s+(.*?)$'),          # demographic makeup
    re.compile(r'ROC AUC Score:\s+(\d+\.\d+)'),          # ROC AUC Score
    re.compile(r'\[\[\s*(\d+)\s+(\d+)\]'),               # Confusion matrix first line
    re.compile(r'\s*\[\s*(\d+)\s+(\d+)\]'),              # Confusion matrix second line
    re.compile(r'Precision:\s+(\d+\.\d+)'),              # Precision
    re.compile(r'Recall:\s+(\d+\.\d+)')                 # Recall
    
]

def parse_log_line(line, pattern_idx):
    match = regex_patterns[pattern_idx].search(line)
    if match:
        return match.groups()
    return None

def scrape_log_to_csv(log_filepaths):
    if not log_filepaths:
        print("No log files provided.")
        return

    # Generate a unique CSV filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSVPATH = f"log_data_{timestamp}.csv"

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

        for log_filepath in log_filepaths:
            with open(log_filepath, 'r') as log_file:
                for line in log_file:
                    # Extract global seed
                    if "Global Seed set to:" in line:
                        hardCodedVars["global_seed"] = line.split(":")[-1].strip()
                    # Extract outcome name
                    if "Outcome Name:" in line:
                        hardCodedVars["outcomeName"] = line.split(":")[-1].strip()

                    csv_line = []  # Reset for each block
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

                                # Write the extracted data to CSV
                                csv_writer.writerow([
                                    hardCodedVars['global_seed'],
                                    hardCodedVars['outcomeType'],
                                    hardCodedVars['outcomeName'],
                                    hardCodedVars['preProcessScriptName'],
                                    hardCodedVars['modelScriptName'],
                                    hardCodedVars['demoComparison'],
                                    prop_demog,  # Correctly placed demographic data
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
                        line = next(log_file, None)  # Get the next line
                        if line is None:
                            break  # End of file

# End time for performance tracking
end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(f"Elapsed time: {elapsed_time} seconds")
