import csv
import re
import time

# Start time
start_time = time.time()

LOGFILE = "Rs_krupitsky_2004.log"
CSVPATH = "LogData.csv"


hardCodedVars = {
    "seed":"N/A",
    "outcomeType":"Binary",
    "outcomeName":"",
    "preProcessScriptName":"pipeline 7-2024",
    "modelScriptName":"TBD",
    "demoComparison":"Race: non hispanic white vs minority",
    "":""
}


# Define the phrases you want to match
regex_patterns = [
    re.compile(r'({})\s+(.*?)$'.format(re.escape('demographic makeup:'))),
    re.compile(r'({})\s+(\d+)\s+(\d+)'.format(re.escape('[['))), #Confusion matrix first line
    re.compile(r'({})\s+(\d+)\s+(\d+)'.format(re.escape(' ['))), #Confusion matrix second line
    re.compile(r'({})\s+(\d+\.\d+)'.format(re.escape('Precision:'))),
    re.compile(r'({})\s+(\d+\.\d+)'.format(re.escape('Recall:')))
]


def parse_log_line(line, pattern_idx):
    if(hardCodedVars['outcomeName'] == ""):
        name = re.compile(r'({})\s+(\S+)'.format(re.escape('Moved'))).search(line)
        if name:
            hardCodedVars['outcomeName'] = name.group(2)
    match = regex_patterns[pattern_idx].search(line)
    if match:
        name = match.group(1)
        next_word = [match.group(2)]
        try:
            next_word = next_word + [match.group(3)]
        except:
            pass
        return next_word
    return None

def scrape_log_to_csv():
    with open(LOGFILE, 'r') as log_file, open(CSVPATH, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the CSV header
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
            
            parsed_line = parse_log_line(line, pattern_idx)
            
            
            if parsed_line:
                csvLine.extend(parsed_line)
                print(csvLine)
                pattern_idx += 1
                if pattern_idx >= len(regex_patterns):
                    csv_writer.writerow([
                        hardCodedVars['seed'],
                        hardCodedVars['outcomeType'],
                        hardCodedVars['outcomeName'],
                        hardCodedVars['preProcessScriptName'],
                        hardCodedVars['modelScriptName'],
                        hardCodedVars['demoComparison'],
                        csvLine[0],
                        csvLine[1],
                        csvLine[4],
                        csvLine[3],
                        csvLine[2],
                        csvLine[5], #prec
                        (int(csvLine[1]) + int(csvLine[4])) / (int(csvLine[1]) + int(csvLine[4]) + int(csvLine[3]) + int(csvLine[2])), #acc
                        csvLine[6], #recall
                        (float(csvLine[5]) * float(csvLine[6])) / (float(csvLine[5]) + float(csvLine[6]))#f1
                    ])
                    pattern_idx = 0
                    csvLine = []

scrape_log_to_csv()

# End time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")