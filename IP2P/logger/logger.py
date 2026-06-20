"""
IP2P/logger/logger.py — MUKSB
Timestamped logger writing to both file and console.
"""
import logging
import os
from datetime import datetime


def setup_logger(log_dir="logs", name="muksb"):
    """
    Create and return (logger, log_file_path).

    Parameters
    ----------
    log_dir : str
        Directory where the log file is written.
    name : str
        Logger name (also used as the log filename prefix).
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_path
