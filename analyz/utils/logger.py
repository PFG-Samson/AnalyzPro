"""Logging utilities for Analyz application."""

import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_level="INFO", log_file=None):
    """
    Configure the logger for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
    
    return logger


def get_logger(name=None):
    """Get a logger instance."""
    return logger.bind(name=name) if name else logger
