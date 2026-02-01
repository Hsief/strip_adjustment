#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging utilities module
Provides a colored log formatter and logging configuration helpers
"""

import logging
import sys
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # cyan
        'INFO': '\033[32m',      # green
        'WARNING': '\033[33m',   # yellow
        'ERROR': '\033[31m',     # red
        'CRITICAL': '\033[35m',  # magenta
        'RESET': '\033[0m'       # reset
    }
    
    def format(self, record):
        # Get color
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format time
        formatted_time = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # Create colored log message
        log_message = f"{color}[{record.levelname}]{reset} {formatted_time} - {record.getMessage()}"
        
        return log_message


def setup_logging(log_level=logging.DEBUG, log_file=None, enable_console=True):
    """
    Configure logging

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, None means do not save to file
        enable_console: Whether to enable console output
    """
    import os
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger():
    """Get logger instance"""
    return logging.getLogger()
