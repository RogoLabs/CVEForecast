"""Logging configuration with rotation support."""

import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(config):
    """Configure logging with rotation and configurable output."""
    level = getattr(logging, config.get('level', 'INFO').upper(), logging.INFO)
    fmt = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('file', 'debug.log')

    handlers = [logging.StreamHandler(sys.stdout)]

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3)
    handlers.append(file_handler)

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
