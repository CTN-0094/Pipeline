import logging

class SilentHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            pass  # Just silently pass if there's an error

def add_silent_handler():
    logging.getLogger().addHandler(SilentHandler())
    
