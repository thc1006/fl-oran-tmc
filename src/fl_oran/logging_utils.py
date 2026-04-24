"""Centralized logging configuration using Rich for readable output."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def get_log_dir() -> Path:
    root = Path(os.environ.get("FL_ORAN_ARTIFACTS", "artifacts")) / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def setup_logging(
    level: str | int = "INFO",
    run_name: str | None = None,
    log_to_file: bool = True,
) -> Path | None:
    """Configure root logger once. Returns the log file path if file logging is on.

    Parameters
    ----------
    level : string level name or logging int.
    run_name : used to name the per-run log file.
    log_to_file : write a file under artifacts/logs/.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return None

    log_path: Path | None = None
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, markup=False, show_path=False, log_time_format="[%X]")
    ]

    if log_to_file:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = run_name or "run"
        log_path = get_log_dir() / f"{tag}_{stamp}.log"
        file_h = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_h.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_h)

    logging.basicConfig(
        level=level if isinstance(level, int) else level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
    # Quiet libraries
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s file=%s",
        logging.getLevelName(logging.getLogger().level),
        str(log_path) if log_path else "<stderr only>",
    )
    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Lazy get-logger; ensures setup has run."""
    if not _CONFIGURED:
        setup_logging(level=os.environ.get("FL_ORAN_LOG_LEVEL", "INFO"))
    return logging.getLogger(name)
