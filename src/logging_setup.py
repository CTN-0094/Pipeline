# logging_setup.py
import logging
import os
import sys
from datetime import datetime


# ANSI color codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta (bold)
}


class _ConsoleFormatter(logging.Formatter):
    """Compact, color-coded formatter for terminal output.

    Format:  HH:MM:SS  LEVEL     module            message
    Colors are suppressed automatically when stdout is not a TTY
    (e.g. when output is piped or redirected).
    """

    def __init__(self):
        super().__init__(datefmt="%H:%M:%S")
        self._use_color = sys.stdout.isatty()

    def format(self, record):
        time_str  = self.formatTime(record, self.datefmt)
        level     = record.levelname

        if self._use_color:
            color     = _LEVEL_COLORS.get(level, _RESET)
            level_str = f"{color}{_BOLD}{level:<8}{_RESET}"
        else:
            level_str = f"{level:<8}"

        module_str = f"{record.module:<18}"
        message    = record.getMessage()

        line = f"{time_str}  {level_str}  {module_str}  {message}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def setup_logging(seed, selected_outcome, directory, quiet=False):
    """Set up logging for the pipeline, creating a log file specific to each seed.

    Two handlers are attached to the root logger:

    - **File** — full detail, nothing lost:
        ``2026-04-30 14:23:01 | INFO     | model_training:train_and_evaluate:38 | message``

    - **Console** — compact, color-coded by level (colors suppressed when piped):
        ``14:23:01  INFO      model_training     message``

    Args:
        seed: Random seed used for this pipeline run.
        selected_outcome: Name of the outcome being modeled (used in log filename).
        directory: Root directory under which the ``logs/`` folder is created.
        quiet: If True, only ERROR and above are logged.

    Returns:
        Path to the log file created for this run.
    """
    log_dir = os.path.join(directory, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{selected_outcome}_{seed}_{timestamp}.log")

    log_level = logging.INFO if not quiet else logging.ERROR

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(log_level)

    # ── File handler ──────────────────────────────────────────────────────────
    # Full context: timestamp, padded level, module + function + line, message.
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # ── Console handler ───────────────────────────────────────────────────────
    # Compact and color-coded; colors are suppressed when not a TTY.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(_ConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Suppress internal logging formatting errors rather than using a no-op handler
    logging.raiseExceptions = False

    return log_filename
