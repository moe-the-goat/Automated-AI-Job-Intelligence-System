"""Centralized logging configuration for the pipeline.

Replaces scattered `print()` calls with a real logging module so we get:
  - Log levels (INFO / WARNING / ERROR) that callers can filter independently.
  - A consistent timestamped format across every module for scannable CI logs.
  - A single setup call from entry points (scraper.py, local_companies.py).
  - Env-var override (`LOG_LEVEL`) for noisy debugging without code changes.

Usage:
    # At the top of every pipeline module:
    from pipeline.logging_setup import get_logger
    logger = get_logger(__name__)

    logger.info("Starting scrape...")
    logger.warning("Source returned 0 jobs.")
    logger.error("Gemini exhausted retries: %s", exc)

    # Once, at the start of main() in scraper.py / local_companies.py:
    from pipeline.logging_setup import configure_logging
    configure_logging()
"""
from __future__ import annotations
import logging
import os
import sys


# Standard format chosen for scanability in GitHub Actions console output.
# Example line:
#   12:34:56 INFO  pipeline.core_filter: Reputation filter: flagged 1 row(s).
DEFAULT_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level=None, stream=None):
    """Configure root logger once at process start.

    Idempotent: calling this twice doesn't add duplicate handlers (we wipe and
    reinstall, so a second call with a different level takes effect cleanly).

    Args:
        level: log threshold. Accepts an int (logging.INFO, etc.) or a
               string ("DEBUG", "INFO", ...). Defaults to env var LOG_LEVEL
               or INFO.
        stream: target stream. Defaults to sys.stderr — keeps log output off
                the data path so tools that capture stdout (e.g. our
                run_all.py) don't accidentally swallow log lines.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if isinstance(level, str):
        level = getattr(logging, level, logging.INFO)
    if stream is None:
        stream = sys.stderr

    root = logging.getLogger()
    # Remove any pre-existing handlers (avoids duplicate log lines when
    # configure_logging is called more than once in the same process —
    # e.g. local QA run followed by a smoke test).
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)

    # Silence chatty third-party libraries that flood the stream at DEBUG.
    for noisy in ("urllib3", "requests", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name):
    """Convenience accessor — equivalent to `logging.getLogger(name)`.

    Kept as a thin wrapper so callers import from a single module rather than
    sprinkling `import logging` + setup boilerplate everywhere.
    """
    return logging.getLogger(name)
