# logging_setup.py
import logging
import os
from datetime import datetime
from src.silent_logging import add_silent_handler  # Import SilentHandler from silent_logging.py


def setup_logging(seed, selected_outcome, directory, quiet=False):
    """Set up logging for the pipeline, creating a log file specific to each seed."""
    # Ensure the log directory exists
    directory = os.path.join(directory, "logs")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate a log filename with a timestamp and seed value
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(directory, f"{selected_outcome}_{seed}_{timestamp}.log")

    # Configure the logging for the pipeline
    logger = logging.getLogger()

    # Clear existing handlers (if any)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging level based on quiet mode
    log_level = logging.INFO if not quiet else logging.ERROR

    # Only log to a file, no terminal (console) output
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)  # Add file handler to logger
    logger.setLevel(log_level)  # Set logging level

    # Add SilentHandler to suppress logging errors
    add_silent_handler()

    return log_filename
