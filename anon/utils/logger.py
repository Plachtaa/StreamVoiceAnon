import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_dir='logs', filename=None):
    """
    Sets up a logger. If a filename is given, the logger will log to a
    file and the console, otherwise only to console.

    :param name: String: name of the logger
    :param filename: String: path to the log file
    :return: Logger: logger
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s- %(levelname)s - %(message)s')
    if filename:
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(Path(log_dir, filename), mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    #ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def get_current_time():
    return datetime.strftime(datetime.today(), '%d-%m-%y_%H:%M')
