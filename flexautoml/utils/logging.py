"""
Logging Utilities for FlexAutoML
=================================
Provides structured logging with support for:
  - Console output with colored formatting
  - File logging
  - Training progress tracking
  - Optimization status
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Custom Formatter with Colors
# ---------------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color codes for terminal output.
    Falls back to plain text when colors are not supported.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",      # Reset
    }

    def __init__(self, fmt: str, use_colors: bool = True) -> None:
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.use_colors = use_colors and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        """Check if the terminal supports colors."""
        try:
            return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        except Exception:
            return False

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def get_logger(
    name: str = "flexautoml",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Returns a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically module name).
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str | None
        Optional path to a log file. Creates parent directories if needed.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    console_handler.setFormatter(ColoredFormatter(console_fmt))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_fmt = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
        file_handler.setFormatter(logging.Formatter(file_fmt))
        logger.addHandler(file_handler)

    return logger


def set_log_level(level: int, logger_name: str = "flexautoml") -> None:
    """
    Sets the logging level for a logger and all its handlers.

    Parameters
    ----------
    level : int
        New logging level (e.g., logging.DEBUG).
    logger_name : str
        Name of the logger to configure.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# ---------------------------------------------------------------------------
# Progress Logger
# ---------------------------------------------------------------------------


class ProgressLogger:
    """
    Tracks and logs training/optimization progress.

    Provides structured progress updates without cluttering the console.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use.
    total_steps : int
        Total number of steps/models to process.
    task_name : str
        Name of the task being tracked.

    Examples
    --------
    >>> progress = ProgressLogger(logger, total_steps=5, task_name="Training")
    >>> progress.step("RandomForest")
    >>> progress.step("XGBoost")
    >>> progress.complete()
    """

    def __init__(
        self,
        logger: logging.Logger,
        total_steps: int,
        task_name: str = "Processing",
    ) -> None:
        self.logger = logger
        self.total_steps = total_steps
        self.task_name = task_name
        self.current_step = 0
        self.start_time = datetime.now()

    def step(self, item_name: str, extra_info: str = "") -> None:
        """Log progress for a single step."""
        self.current_step += 1
        pct = (self.current_step / self.total_steps) * 100
        msg = f"[{self.current_step}/{self.total_steps}] {pct:5.1f}% │ {item_name}"
        if extra_info:
            msg += f" │ {extra_info}"
        self.logger.info(msg)

    def complete(self) -> None:
        """Log completion of the entire task."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(
            f"{self.task_name} complete │ {self.total_steps} items │ "
            f"Elapsed: {elapsed.total_seconds():.1f}s"
        )

    def warning(self, item_name: str, message: str) -> None:
        """Log a warning for a specific item."""
        self.logger.warning(f"[{item_name}] {message}")

    def error(self, item_name: str, message: str) -> None:
        """Log an error for a specific item."""
        self.logger.error(f"[{item_name}] {message}")


# ---------------------------------------------------------------------------
# Optimization Logger
# ---------------------------------------------------------------------------


class OptimizationLogger:
    """
    Logs Optuna optimization progress.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use.
    model_name : str
        Name of model being optimized.
    n_trials : int
        Total number of trials.

    Examples
    --------
    >>> opt_log = OptimizationLogger(logger, "XGBClassifier", 30)
    >>> opt_log.trial_complete(1, 0.85, {"max_depth": 5})
    >>> opt_log.best_found(0.92, {"max_depth": 8})
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_name: str,
        n_trials: int,
    ) -> None:
        self.logger = logger
        self.model_name = model_name
        self.n_trials = n_trials

    def trial_complete(
        self,
        trial_number: int,
        score: float,
        params: dict,
    ) -> None:
        """Log completion of a single trial."""
        if trial_number % max(1, self.n_trials // 10) == 0:  # Log every 10%
            self.logger.debug(
                f"[{self.model_name}] Trial {trial_number}/{self.n_trials} │ "
                f"Score: {score:.4f}"
            )

    def best_found(self, score: float, params: dict) -> None:
        """Log when a new best trial is found."""
        top_params = dict(list(params.items())[:3])
        self.logger.info(
            f"[{self.model_name}] Best score: {score:.4f} │ "
            f"Top params: {top_params}"
        )

    def optimization_complete(
        self,
        best_score: float,
        n_completed: int,
        elapsed_seconds: float,
    ) -> None:
        """Log optimization completion summary."""
        self.logger.info(
            f"[{self.model_name}] Optimization complete │ "
            f"Best: {best_score:.4f} │ "
            f"Trials: {n_completed}/{self.n_trials} │ "
            f"Time: {elapsed_seconds:.1f}s"
        )
