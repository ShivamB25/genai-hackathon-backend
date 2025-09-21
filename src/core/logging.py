"""Structured logging configuration for the AI-Powered Trip Planner Backend.

This module provides structured logging using structlog with support for both
development and production environments, including Google Cloud Logging integration.
"""

import logging
import logging.config
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

from .config import settings

# Context variable for request correlation ID
request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str:
    """Get or generate a request ID for correlation tracking.

    Returns:
        str: The current request ID or a newly generated one.
    """
    request_id = request_id_context.get()
    if not request_id:
        request_id = str(uuid.uuid4())
        request_id_context.set(request_id)
    return request_id


def set_request_id(request_id: str) -> None:
    """Set the request ID for correlation tracking.

    Args:
        request_id: The request ID to set.
    """
    request_id_context.set(request_id)


def clear_request_id() -> None:
    """Clear the current request ID."""
    request_id_context.set(None)


def add_request_id(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """Add request ID to log records for correlation tracking.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary to modify.

    Returns:
        EventDict: The modified event dictionary.
    """
    if settings.log_correlation_id:
        event_dict["request_id"] = get_request_id()
    return event_dict


def add_severity_level(
    _logger: Any, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Add Google Cloud Logging compatible severity level.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary to modify.

    Returns:
        EventDict: The modified event dictionary.
    """
    # Map Python log levels to Google Cloud Logging severity levels
    severity_mapping = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
    }

    level = event_dict.get("level", "info")
    level = level.lower() if isinstance(level, str) else str(level).lower()

    event_dict["severity"] = severity_mapping.get(level, "INFO")
    return event_dict


def add_service_context(
    _logger: Any, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Add service context information to log records.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary to modify.

    Returns:
        EventDict: The modified event dictionary.
    """
    event_dict.update(
        {
            "service": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
            }
        }
    )
    return event_dict


def configure_development_logging() -> None:
    """Configure logging for development environment with pretty console output."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        add_request_id,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )


def configure_production_logging() -> None:
    """Configure logging for production environment with structured JSON output."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        add_request_id,
        add_severity_level,
        add_service_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging for JSON output
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
    )


def configure_google_cloud_logging() -> None:
    """Configure Google Cloud Logging integration for production."""
    try:
        # Import Google Cloud Logging client
        from google.cloud import logging as gcp_logging
        from google.cloud.logging.handlers import CloudLoggingHandler

        # Initialize the Google Cloud Logging client
        gcp_client = gcp_logging.Client(project=settings.google_cloud_project)

        # Create a Cloud Logging handler
        handler = CloudLoggingHandler(
            gcp_client,
            name=settings.app_name,
        )

        # Configure the handler with structured logging
        handler.setLevel(getattr(logging, settings.log_level.upper()))

        # Add the handler to the root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        # Configure structlog to use Google Cloud Logging
        processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            add_request_id,
            add_severity_level,
            add_service_context,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, settings.log_level.upper())
            ),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    except ImportError:
        # Fallback to standard structured logging if Google Cloud Logging is not available
        configure_production_logging()


def configure_logging() -> None:
    """Configure logging based on the current environment and settings."""
    if settings.environment == "development":
        configure_development_logging()
    elif settings.enable_cloud_logging and settings.google_cloud_project:
        configure_google_cloud_logging()
    else:
        configure_production_logging()

    # Set log levels for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("firebase_admin").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: The logger name (usually __name__).

    Returns:
        structlog.BoundLogger: A configured structlog logger instance.
    """
    return structlog.get_logger(name)


# Configure logging when the module is imported
configure_logging()

# Export commonly used logger
logger = get_logger(__name__)
