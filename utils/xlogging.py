# Provides logger for other modules.

import sys
import os

from loguru._handler import Message
from loguru import logger


LOG_FILE_PATH = 'logs/xlogging.log'
# Logging level above which the application will crash
CRASH_LEVEL = 'CRITICAL'
FORCE_CRASH = False


def crash_application(message:Message):
    """ Crashes application. 
    
    :param record: provided by logger.add. Contains all contextual information
            of the logging call (time, function, file, line, level, etc.)
    """
    if message.record['level'].no >= logger.level(CRASH_LEVEL).no:
        sys.stderr.flush()
        sys.stdout.flush()
        if FORCE_CRASH:
            os._exit(1)
        else:
            sys.exit(1)

def get_logger():
    stdout_lavel = 'DEBUG'
    stderr_level = 'ERROR'
    logfile_level = 'DEBUG'
    logger.remove()
    logger.add(sys.stdout, level=stdout_lavel)
    logger.debug(f'Logging level set to {stdout_lavel} for stdout.')
    logger.add(sys.stderr, level=stderr_level)
    logger.debug(f'Logging level set to {stderr_level} for stderr.')
    logger.add(LOG_FILE_PATH, level=logfile_level)
    logger.debug(f'Logging level set to {logfile_level} for {LOG_FILE_PATH}.')
    logger.add(crash_application, level=CRASH_LEVEL)
    logger.debug(f'Application will crash on {CRASH_LEVEL}.')
    return logger
