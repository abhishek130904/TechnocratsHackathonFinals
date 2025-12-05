"""
Logging utilities.

This module provides centralized logging configuration
for the application.
"""

import logging
import sys
from pathlib import Path
from app.config import LOG_LEVEL, LOG_FILE

def setup_logging():
    """
    Set up logging configuration for the application.
    
    Configures:
    - Log level (INFO, DEBUG, etc.)
    - Log format (timestamp, level, message)
    - File and console handlers
    """
    # Create logs directory if it doesn't exist
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)

