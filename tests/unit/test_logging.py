import json
import logging
from io import StringIO

import structlog

from src.core.logging import configure_logging, get_logger


def test_configure_logging():
    """Test that logging is set up correctly."""
    # This test is basic and assumes configure_logging doesn't crash.
    configure_logging()
    logger = structlog.get_logger("test_logger")
    assert isinstance(logger, structlog.BoundLogger)


def test_get_logger():
    """Test that get_logger returns a logger with the correct name."""
    logger = get_logger("my_test_module")
    assert logger.name == "my_test_module"


def test_structured_logging_output():
    """Test that the logger produces structured JSON logs in production."""
    log_stream = StringIO()

    # Reconfigure structlog for testing production-style logs
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=log_stream),
        wrapper_class=structlog.BoundLogger,
    )

    logger = get_logger("structured_log_test")

    logger.info("This is a test message", key="value")

    log_output = log_stream.getvalue()
    log_data = json.loads(log_output)

    assert log_data["level"] == "info"
    assert log_data["message"] == "This is a test message"
    assert log_data["key"] == "value"


def test_log_levels():
    """Test that the logger respects different log levels."""
    log_stream = StringIO()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=log_stream),
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )

    logger = get_logger("log_level_test")

    logger.debug("This should not be logged.")
    logger.info("This should also not be logged.")
    logger.warning("This is a warning.")
    logger.error("This is an error.")

    log_output = log_stream.getvalue()

    assert "This should not be logged." not in log_output
    assert "This should also not be logged." not in log_output

    # Each log is a separate JSON object on a new line
    log_lines = [json.loads(line) for line in log_output.strip().split("\n")]

    assert len(log_lines) == 2
    assert log_lines[0]["level"] == "warning"
    assert log_lines[0]["message"] == "This is a warning."
    assert log_lines[1]["level"] == "error"
    assert log_lines[1]["message"] == "This is an error."


# Reset logging configuration after tests
def teardown_function(function):
    """Reset structlog configuration after each test."""
    structlog.reset_defaults()
