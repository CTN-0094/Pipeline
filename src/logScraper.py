import csv
import re
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory containing the log files
LOG_DIR = "/Users/richeyjay/Desktop/Relapse_Pipeline/env/logs"  # Update this path to the directory where your log files are stored
CSVPATH = "LogData.csv"

# Define hardcoded variables
hardCodedVars = {
    "seed": "N/A",
    "outcomeType": "Binary",
    "outcomeName": "",
    "preProcessScriptName": "pipeline 7-2024",
    "modelScriptName": "TBD",
    "demoComparison": "Race: non hispanic white vs minority"
}

# Define regex patterns to match specific log lines
regex_patterns = [
    re.compile(r'demographic makeup:\s*(.*)'),
    re.compile(r'\[\[\s*(\d+)\s+(\d+)'),
    re.compile(r'\[\s*(\d+)\s+(\d+)'),
    re.compile(r'Precision:\s*(\d+\.\d+)'),
    re.compile(r'Recall:\s*(\d+\.\d+)')
]

# Function to parse each log line
def parse_log_line(line, pattern_idx):
    if hardCodedVars['outcomeName'] == "":
        name_match = re.search(r'Moved\s+(\S+)', line)
        if name_match:
            hardCodedVars['outcomeName'] = name_match.group(1)

    match = regex_patterns[pattern_idx].search(line)
    if match:
        result = list(match.groups())
        return result
    return None

# Function to find the latest log file in the directory
def find_latest_log_file(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.startswith('pipeline') and f.endswith('.log')]
    latest_log_file = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
    return os.path.join(log_dir, latest_log_file)

# Function to scrape the log file and write to CSV
def scrape_log_to_csv(log_file_path):
    with open(log_file_path, 'r') as log_file, open(CSVPATH, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the CSV header if the file is new
        if os.path.getsize(CSVPATH) == 0:
            csv_writer.writerow([
                'seed',
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
        csvLine = []
        for line in log_file:
            logging.debug(f"Processing line: {line.strip()}")
            parsed_line = parse_log_line(line, pattern_idx)
            
            if parsed_line:
                logging.debug(f"Matched pattern {pattern_idx}: {parsed_line}")
                csvLine.extend(parsed_line)
                pattern_idx += 1
                if pattern_idx >= len(regex_patterns):
                    precision = float(csvLine[5])
                    recall = float(csvLine[6])
                    tp = int(csvLine[1])
                    tn = int(csvLine[4])
                    fp = int(csvLine[3])
                    fn = int(csvLine[2])
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    f1_score = 2 * (precision * recall) / (precision + recall)

                    csv_writer.writerow([
                        hardCodedVars['seed'],
                        hardCodedVars['outcomeType'],
                        hardCodedVars['outcomeName'],
                        hardCodedVars['preProcessScriptName'],
                        hardCodedVars['modelScriptName'],
                        hardCodedVars['demoComparison'],
                        csvLine[0],  # Prop(Demog)
                        tp,
                        tn,
                        fp,
                        fn,
                        precision,
                        accuracy,
                        recall,
                        f1_score
                    ])
                    pattern_idx = 0
                    csvLine = []

# Main function to scrape the latest log file
def main():
    # Find the latest log file
    latest_log_file = find_latest_log_file(LOG_DIR)
    logging.info(f"Found latest log file: {latest_log_file}")
    # Run the scraping function
    scrape_log_to_csv(latest_log_file)

# Run the main function
if __name__ == "__main__":
    main()