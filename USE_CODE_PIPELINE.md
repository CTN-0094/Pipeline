# Pipeline Execution Guide

This guide provides instructions on how to execute the data pipeline, which includes setting up logging, selecting outcomes, and generating results in CSV format.

## Prerequisites

- Ensure you have Python installed on your system.
- Make sure all required Python packages are installed. You can typically install them using `pip install <package_name>`.

## Code Structure

The code is organized into several modules:

- `run_pipelineV2.py`: Main script to execute the pipeline.
- `utils.py`: Contains utility functions for logging and outcome selection.
- `data_loading.py`: Handles dataset loading.
- `data_preprocessing.py`: Preprocesses merged data.
- `demographic_handling.py`: Manages demographic data subsets.
- `model_training.py`: Trains and evaluates models.
- `logScraper.py`: Extracts data from log files and writes to CSV.

## Usage Instructions

1. **Clone the Repository:**

   Clone the repository to your local machine to get all the files needed for the pipeline.

   ```bash
   git clone <repository_url>
   cd <repository_name>
pip install -r requirements.txt
python run_pipelineV2.py
Enter a seed number (or press Enter to use a dynamic seed): <your_seed>

