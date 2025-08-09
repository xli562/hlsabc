import logging
import sys
import inspect
import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

class ColorFormatter(logging.Formatter):
    """
    A custom logging formatter to add colors based on the log level.

    :return: (ColorFormatter) Instance with color-coded output for each log level.
    """

    LEVEL_COLORS = {
        logging.DEBUG: colorama.Fore.WHITE,
        logging.INFO: colorama.Fore.GREEN,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.CRITICAL: colorama.Fore.RED + colorama.Style.BRIGHT
    }

    def format(self, record):
        """
        Format a log record with color based on its level.

        :param record: (logging.LogRecord) The log record to format.
        :return: (str) The formatted log message, including color codes.
        """
        log_color = self.LEVEL_COLORS.get(record.levelno, colorama.Fore.WHITE)
        colored_fmt = log_color + self._fmt + colorama.Style.RESET_ALL
        formatter = logging.Formatter(colored_fmt, self.datefmt)
        return formatter.format(record)

LOG_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root logger initialization
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Prevent duplicate logs if this module is re-imported
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Console handler with our color formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(ColorFormatter(LOG_FORMAT, DATE_FORMAT))
root_logger.addHandler(console_handler)

def get_logger():
    """
    Obtain a logger named after the calling module.

    :return: (logging.Logger) Logger instance for the caller's module.
    """
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    logger_name = caller_module.__name__ if caller_module else '__main__'
    return logging.getLogger(logger_name)

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle uncaught exceptions by logging them at the CRITICAL level.

    :param exc_type: (type) Exception type.
    :param exc_value: (BaseException) Exception instance.
    :param exc_traceback: (traceback) Traceback object for the exception.
    :return: None
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    root_logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

# Register the custom exception hook
sys.excepthook = handle_exception

def set_logging_level(level_str: str):
    """
    Set the logging level of the root logger and all attached handlers.

    :param level_str: (str) Desired logging level. Valid options:
                      'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
                      Case-insensitive.
    :return: None
    """
    valid_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    new_level = valid_levels.get(level_str.upper(), logging.INFO)
    root_logger.setLevel(new_level)
    for handler in root_logger.handlers:
        handler.setLevel(new_level)
    root_logger.debug(f"Logging level set to {level_str.upper()}")

