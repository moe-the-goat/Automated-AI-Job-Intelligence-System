"""configure_logging + get_logger from pipeline.logging_setup.

Tiny module, easy to lock down. Verifies:
  - configure_logging is idempotent (no duplicate handlers on second call)
  - level overrides take effect
  - get_logger returns a proper logger instance
  - chatty third-party libraries are quieted to WARNING
"""
import io
import logging
from pipeline.logging_setup import configure_logging, get_logger


def _root_handlers():
    return logging.getLogger().handlers


def test_get_logger_returns_logger_instance():
    log = get_logger("test.sample")
    assert isinstance(log, logging.Logger)
    assert log.name == "test.sample"


def test_configure_logging_attaches_a_handler():
    configure_logging(level="INFO")
    assert len(_root_handlers()) >= 1


def test_configure_logging_is_idempotent():
    """Two calls in a row should not stack handlers — would lead to duplicate lines."""
    configure_logging(level="INFO")
    n_first = len(_root_handlers())
    configure_logging(level="INFO")
    n_second = len(_root_handlers())
    assert n_first == n_second


def test_configure_logging_respects_string_level():
    configure_logging(level="WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_respects_int_level():
    configure_logging(level=logging.ERROR)
    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_silences_noisy_libraries():
    """We don't want requests/urllib3 DEBUG flooding the output."""
    configure_logging(level="DEBUG")
    for noisy in ("urllib3", "requests", "httpx", "httpcore"):
        assert logging.getLogger(noisy).level == logging.WARNING


def test_configure_logging_writes_to_provided_stream():
    """Pass our own stream so we can inspect what got written."""
    buf = io.StringIO()
    configure_logging(level="INFO", stream=buf)
    log = get_logger("test.writes")
    log.info("hello from test")
    output = buf.getvalue()
    assert "hello from test" in output
    assert "INFO" in output
    assert "test.writes" in output


def test_configure_logging_default_level_is_info():
    """No explicit level + no LOG_LEVEL env var -> INFO."""
    import os
    original = os.environ.pop("LOG_LEVEL", None)
    try:
        configure_logging()
        assert logging.getLogger().level == logging.INFO
    finally:
        if original is not None:
            os.environ["LOG_LEVEL"] = original
