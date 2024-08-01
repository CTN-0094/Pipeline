import csv
import re
import glob
import os
import time

# Start time for performance tracking
start_time = time.time()

# Find the most recent log file based on the naming pattern
log_files = glob.glob("logs/pipeline_*.log")
if log_files:
    latest_log_file = max(log_files, key=os.path.getctime)
else:
    print("No log files found.")
    latest_log_file = None

CSVPATH = "log_data.csv"

# Dictionary of hardcoded variables to include in the CSV output
hardCodedVars = {
    "global_seed": "N/A",  # Add global seed here
    "outcomeType": "Binary",
    "outcomeName": "",
    "preProcessScriptName": "pipeline 7-2024",
    "modelScriptName": "TBD",
    "demoComparison": "Race: non hispanic white vs minority",
    "": ""
}

# Define the phrases you want to match
regex_patterns = [
    re.compile(r'demographic makeup:\s+(.*?)$'),
    re.compile(r'\[\[\s*(\d+)\s+(\d+)'), # Confusion matrix first line
    re.compile(r'\s*\[\s*(\d+)\s+(\d+)\]'), # Confusion matrix second line
    re.compile(r'Precision:\s+(\d+\.\d+)'),
    re.compile(r'Recall:\s+(\d+\.\d+)')
]

def parse_log_line(line, pattern_idx):
    match = regex_patterns[pattern_idx].search(line)
    if match:
        return match.groups()
    return None

def scrape_log_to_csv():
    if not latest_log_file:
        return

    with open(latest_log_file, 'r') as log_file, open(CSVPATH, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the CSV header if the file is empty
        if os.stat(CSVPATH).st_size == 0:
            csv_writer.writerow([
                'global_seed',  # Add global seed to the CSV header
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
                'Precision',
                'Accuracy',
                'Recall',
                'F1'
            ])
        
        pattern_idx = 0
        csv_line = []

        for line in log_file:
            # Extract global seed
            if "Global Seed set to:" in line:
                hardCodedVars["global_seed"] = line.split(":")[-1].strip()

            parsed_line = parse_log_line(line, pattern_idx)
            if parsed_line:
                csv_line.extend(parsed_line)
                pattern_idx += 1
                if pattern_idx >= len(regex_patterns):
                    tp, fn, fp, tn = map(int, csv_line[1:5])
                    precision = float(csv_line[5])
                    recall = float(csv_line[6])
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
                        csv_line[0],
                        tp,
                        tn,
                        fp,
                        fn,
                        precision,
                        accuracy,
                        recall,
                        f1
                    ])

                    pattern_idx = 0
                    csv_line = []

if __name__ == "__main__":
    scrape_log_to_csv()

# End time for performance tracking
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")