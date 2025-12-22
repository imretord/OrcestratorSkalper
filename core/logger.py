"""
Logging configuration for AI Trading System V3.
Uses loguru with rich console output and file rotation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.theme import Theme

# Rich console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green",
})
console = Console(theme=custom_theme)


def setup_logger(
    log_file: str = "logs/trading.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> Any:
    """
    Configure loguru logger with file rotation and rich console output.

    Args:
        log_file: Path to log file
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log file (e.g., "10 MB", "1 day")
        retention: How long to keep old logs (e.g., "7 days")

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom format for file logging
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Console format (simpler, colored)
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    )

    # Add file handler with rotation
    logger.add(
        log_file,
        format=file_format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,  # Thread-safe
        backtrace=True,
        diagnose=True,
    )

    # Add console handler with colors
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        enqueue=True,
    )

    logger.info(f"Logger initialized: file={log_file}, level={level}")

    return logger


def get_logger(name: str = "trading") -> Any:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (appears in log messages)

    Returns:
        Logger instance bound to the given name
    """
    return logger.bind(name=name)


# Module-level logger instance
log = get_logger("core")
