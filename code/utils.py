import logging
import sys

def setup_logging(config):
    """Configures logging for the application."""
    logging.basicConfig(
        level=config['level'],
        format=config['format'],
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
