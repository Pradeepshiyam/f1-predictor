"""
Structured logger factory for the F1 Prediction Platform.

Usage:
    from f1_predictor.common.logger import get_logger
    log = get_logger(__name__)
    log.info("Prediction generated", race="Canadian GP", year=2026)
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR     = Path("outputs/logs")
_LOG_FILE    = _LOG_DIR / "f1_platform.log"
_MAX_BYTES   = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 3                 # keep last 3 rotated files


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_root_logger() -> None:
    """Set up the root logger once per process."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured — avoid duplicate handlers

    root.setLevel(logging.DEBUG)

    # ── Console handler (INFO+) ───────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # ── Rotating file handler (DEBUG+) ────────────────────────────────────────
    try:
        _ensure_log_dir()
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)
    except OSError:
        # If the log directory is read-only (e.g. in some CI environments),
        # fall back to console-only silently.
        pass

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "fastf1", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Call once per module at module level:

        log = get_logger(__name__)
    """
    return logging.getLogger(name)
