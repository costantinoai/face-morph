"""Structured logging for face morphing pipeline.

This module provides a centralized logging system that replaces all print() and
vprint() calls throughout the codebase with proper structured logging at
appropriate levels (INFO, DEBUG, WARNING, ERROR).
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Add colors to console output for better readability.

    Uses ANSI escape codes to colorize log levels in terminal output.
    Colors are only applied to the level name, not the entire message.
    """

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colored level name.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with ANSI color codes
        """
        # Save original levelname
        original_levelname = record.levelname

        # Add color to levelname if it's a known level
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.RESET}"
            )

        # Format the record
        formatted = super().format(record)

        # Restore original levelname
        record.levelname = original_levelname

        return formatted


def setup_logger(
    name: str = 'face_morph',
    verbose: bool = True,
    log_file: Optional[Path] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Configure structured logging for the face morphing pipeline.

    Creates a logger with both file and console handlers. The file handler
    always logs at DEBUG level for comprehensive logs, while the console
    handler respects the specified log_level and verbose settings.

    Args:
        name: Logger name (typically 'face_morph' or module-specific)
        verbose: If True, enable console output. If False, only log to file.
        log_file: Optional file path for persistent logs. If None, no file logging.
        log_level: Console logging level: "DEBUG", "INFO", "WARNING", or "ERROR"

    Returns:
        Configured logger instance ready for use

    Example:
        >>> from pathlib import Path
        >>> logger = setup_logger(
        ...     name='face_morph',
        ...     verbose=True,
        ...     log_file=Path('results/session.log'),
        ...     log_level='INFO'
        ... )
        >>> logger.info("Starting morphing pipeline")
        INFO: Starting morphing pipeline
        >>> logger.debug("Device: cuda:0")  # Only in file, not console

    Notes:
        - File handler logs everything at DEBUG level
        - Console handler respects log_level parameter
        - Colors are only shown in console, not in file
        - Calling this multiple times with same name updates existing logger
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Prevent propagation to root logger
    logger.propagate = False

    # File handler (always DEBUG level for comprehensive logs)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)  # File gets everything
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)

    # Console handler (respects verbose and log_level)
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, log_level.upper()))
        ch.setFormatter(ColoredFormatter('%(levelname)s: %(message)s'))
        logger.addHandler(ch)

    return logger


def get_logger(name: str = 'face_morph') -> logging.Logger:
    """
    Get existing logger instance by name.

    Retrieves a logger that was previously configured with setup_logger().
    If the logger hasn't been set up yet, returns an unconfigured logger
    (which will use Python's default logging behavior).

    Args:
        name: Logger name to retrieve (default: 'face_morph')

    Returns:
        Logger instance

    Example:
        >>> # In main script
        >>> from face_morph.utils.logging import setup_logger
        >>> setup_logger(name='face_morph', verbose=True)

        >>> # In any module
        >>> from face_morph.utils.logging import get_logger
        >>> logger = get_logger('face_morph')
        >>> logger.info("Processing mesh...")
        INFO: Processing mesh...

    Notes:
        - Always use the same name across modules for consistency
        - Typically use 'face_morph' as the base logger name
        - For module-specific loggers, use: get_logger('face_morph.core')
    """
    return logging.getLogger(name)


# Convenience function for migration from vprint
def migrate_vprint_to_logger(verbose: bool = True) -> logging.Logger:
    """
    Helper function for migrating from old vprint() pattern to logging.

    This is a drop-in replacement for the old set_verbose() + vprint() pattern.
    Creates a basic console-only logger that mimics vprint behavior.

    Args:
        verbose: If True, logger outputs to console. If False, logger is silent.

    Returns:
        Logger configured to mimic vprint behavior

    Example:
        >>> # OLD CODE:
        >>> from lib.utils import set_verbose, vprint
        >>> set_verbose(True)
        >>> vprint("Processing...")

        >>> # NEW CODE:
        >>> from face_morph.utils.logging import migrate_vprint_to_logger
        >>> logger = migrate_vprint_to_logger(verbose=True)
        >>> logger.debug("Processing...")  # Use debug for verbose messages

    Deprecated:
        This is a migration helper only. New code should use setup_logger()
        and get_logger() directly for better control.
    """
    return setup_logger(
        name='face_morph',
        verbose=verbose,
        log_file=None,
        log_level='DEBUG' if verbose else 'WARNING'
    )
