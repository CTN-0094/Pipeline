import logging

# Retained for backwards compatibility — logging.raiseExceptions = False
# in logging_setup.py now handles internal formatting errors.
def add_silent_handler():
    logging.raiseExceptions = False
