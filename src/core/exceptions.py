"""Global exception handling for AI-Powered Trip Planner Backend.

This module provides custom exception classes and global exception handlers
for FastAPI with structured error responses and security-focused error handling.
"""

import traceback
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# Base Exception Classes
class TripPlannerBaseException(Exception):
    """Base exception class for the trip planner application."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


# Authentication & Authorization Exceptions
class AuthenticationException(TripPlannerBaseException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", **kwargs) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_401_UNAUTHORIZED, **kwargs
        )


class AuthorizationException(TripPlannerBaseException):
    """Raised when user lacks required permissions."""

    def __init__(self, message: str = "Insufficient permissions", **kwargs) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_403_FORBIDDEN, **kwargs
        )


class TokenExpiredException(AuthenticationException):
    """Raised when authentication token has expired."""

    def __init__(
        self, message: str = "Authentication token has expired", **kwargs
    ) -> None:
        super().__init__(message=message, **kwargs)


class InvalidTokenException(AuthenticationException):
    """Raised when authentication token is invalid."""

    def __init__(self, message: str = "Invalid authentication token", **kwargs) -> None:
        super().__init__(message=message, **kwargs)


# User & Profile Exceptions
class UserNotFoundException(TripPlannerBaseException):
    """Raised when a user is not found."""

    def __init__(self, message: str = "User not found", **kwargs) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_404_NOT_FOUND, **kwargs
        )


class UserAlreadyExistsException(TripPlannerBaseException):
    """Raised when attempting to create a user that already exists."""

    def __init__(self, message: str = "User already exists", **kwargs) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_409_CONFLICT, **kwargs
        )


class ProfileIncompleteException(TripPlannerBaseException):
    """Raised when user profile is incomplete for the requested operation."""

    def __init__(
        self, message: str = "User profile setup is required", **kwargs
    ) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, **kwargs
        )


# Validation Exceptions
class ValidationException(TripPlannerBaseException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        errors: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        self.errors = errors or []
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"validation_errors": self.errors},
            **kwargs,
        )


class InvalidInputException(ValidationException):
    """Raised when input data is invalid."""

    def __init__(self, field: str, message: str, **kwargs) -> None:
        errors = [{"field": field, "message": message}]
        super().__init__(
            message=f"Invalid input for field '{field}': {message}",
            errors=errors,
            **kwargs,
        )


# Business Logic Exceptions
class TripPlanningException(TripPlannerBaseException):
    """Base exception for trip planning operations."""

    def __init__(self, message: str = "Trip planning error", **kwargs) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, **kwargs
        )


class InvalidDestinationException(TripPlanningException):
    """Raised when destination is invalid or not supported."""

    def __init__(self, destination: str, **kwargs) -> None:
        super().__init__(
            message=f"Invalid or unsupported destination: {destination}",
            details={"destination": destination},
            **kwargs,
        )


class TripNotFoundException(TripPlannerBaseException):
    """Raised when a trip is not found."""

    def __init__(self, trip_id: str, **kwargs) -> None:
        super().__init__(
            message=f"Trip not found: {trip_id}",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"trip_id": trip_id},
            **kwargs,
        )


# External Service Exceptions
class ExternalServiceException(TripPlannerBaseException):
    """Base exception for external service errors."""

    def __init__(
        self, service: str, message: str = "External service error", **kwargs
    ) -> None:
        self.service = service
        super().__init__(
            message=f"{service}: {message}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service},
            **kwargs,
        )


class FirebaseException(ExternalServiceException):
    """Raised when Firebase operations fail."""

    def __init__(self, message: str = "Firebase operation failed", **kwargs) -> None:
        super().__init__(service="Firebase", message=message, **kwargs)


class VertexAIException(ExternalServiceException):
    """Raised when Vertex AI operations fail."""

    def __init__(self, message: str = "Vertex AI operation failed", **kwargs) -> None:
        super().__init__(service="Vertex AI", message=message, **kwargs)


class GoogleMapsException(ExternalServiceException):
    """Raised when Google Maps API operations fail."""

    def __init__(
        self, message: str = "Google Maps API operation failed", **kwargs
    ) -> None:
        super().__init__(service="Google Maps API", message=message, **kwargs)


# Rate Limiting Exceptions
class RateLimitException(TripPlannerBaseException):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after} if retry_after else {},
            **kwargs,
        )


# Database Exceptions
class DatabaseException(TripPlannerBaseException):
    """Base exception for database operations."""

    def __init__(self, message: str = "Database operation failed", **kwargs) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, **kwargs
        )


class DatabaseConnectionException(DatabaseException):
    """Raised when database connection fails."""

    def __init__(self, message: str = "Database connection failed", **kwargs) -> None:
        super().__init__(message=message, **kwargs)


class DatabaseTimeoutException(DatabaseException):
    """Raised when database operation times out."""

    def __init__(self, message: str = "Database operation timed out", **kwargs) -> None:
        super().__init__(message=message, **kwargs)


# Configuration Exceptions
class ConfigurationException(TripPlannerBaseException):
    """Raised when configuration is invalid."""

    def __init__(self, message: str = "Configuration error", **kwargs) -> None:
        super().__init__(
            message=message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, **kwargs
        )


class MissingConfigurationException(ConfigurationException):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, **kwargs) -> None:
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            details={"config_key": config_key},
            **kwargs,
        )


# Error Response Utilities
def create_error_response(
    message: str,
    error_code: str,
    status_code: int,
    details: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Create a standardized error response.

    Args:
        message: Error message.
        error_code: Error code identifier.
        status_code: HTTP status code.
        details: Additional error details.
        errors: List of validation errors.
        request_id: Request correlation ID.

    Returns:
        Dict[str, Any]: Standardized error response.
    """
    response = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
            "status_code": status_code,
        },
        "timestamp": logger._context.get("timestamp", None),
    }

    if request_id:
        response["request_id"] = request_id

    if details:
        # Filter out sensitive information
        safe_details = {
            k: v
            for k, v in details.items()
            if k not in ["password", "token", "secret", "key", "private"]
        }
        if safe_details:
            response["error"]["details"] = safe_details

    if errors:
        response["error"]["validation_errors"] = errors

    # Add debug information in development
    if settings.debug:
        response["debug"] = {
            "environment": settings.environment,
            "service": settings.app_name,
            "version": settings.app_version,
        }

    return response


def log_exception(
    exc: Exception,
    request: Request | None = None,
    extra_context: dict[str, Any] | None = None,
) -> None:
    """Log exception with context and sanitized information.

    Args:
        exc: The exception to log.
        request: The FastAPI request object.
        extra_context: Additional context to log.
    """
    context = {
        "exception_type": exc.__class__.__name__,
        "exception_message": str(exc),
    }

    if request:
        context.update(
            {
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent", "unknown"),
                "client_ip": request.client.host if request.client else "unknown",
            }
        )

    if extra_context:
        # Filter out sensitive information
        safe_context = {
            k: v
            for k, v in extra_context.items()
            if not any(
                sensitive in k.lower()
                for sensitive in ["password", "token", "secret", "key"]
            )
        }
        context.update(safe_context)

    # Log with appropriate level
    if isinstance(exc, AuthenticationException | AuthorizationException):
        logger.warning("Authentication/Authorization error", **context)
    elif isinstance(exc, ValidationException):
        logger.info("Validation error", **context)
    elif isinstance(exc, UserNotFoundException | TripNotFoundException):
        logger.info("Resource not found", **context)
    elif isinstance(exc, ExternalServiceException):
        logger.error("External service error", **context, exc_info=True)
    elif isinstance(exc, TripPlannerBaseException):
        logger.error("Application error", **context, exc_info=True)
    else:
        logger.error("Unhandled exception", **context, exc_info=True)


# Exception Handlers
async def trip_planner_exception_handler(
    request: Request, exc: TripPlannerBaseException
) -> JSONResponse:
    """Handle custom application exceptions.

    Args:
        request: The FastAPI request object.
        exc: The custom exception.

    Returns:
        JSONResponse: Formatted error response.
    """
    log_exception(exc, request)

    request_id = request.headers.get("X-Request-ID")

    error_response = create_error_response(
        message=exc.message,
        error_code=exc.error_code,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id,
    )

    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    # Add retry-after header for rate limit errors
    if isinstance(exc, RateLimitException) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=exc.status_code, content=error_response, headers=headers
    )


async def validation_exception_handler(
    request: Request, exc: PydanticValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors.

    Args:
        request: The FastAPI request object.
        exc: The Pydantic validation error.

    Returns:
        JSONResponse: Formatted validation error response.
    """
    log_exception(exc, request)

    # Convert Pydantic errors to our format
    validation_errors = []
    for error in exc.errors():
        validation_errors.append(
            {
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "validation_error"),
                "input": (
                    str(error.get("input", ""))[:100] if error.get("input") else None
                ),
            }
        )

    request_id = request.headers.get("X-Request-ID")

    error_response = create_error_response(
        message="Validation failed",
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        errors=validation_errors,
        request_id=request_id,
    )

    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response,
        headers=headers,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions.

    Args:
        request: The FastAPI request object.
        exc: The HTTP exception.

    Returns:
        JSONResponse: Formatted error response.
    """
    log_exception(exc, request)

    request_id = request.headers.get("X-Request-ID")

    error_response = create_error_response(
        message=exc.detail if isinstance(exc.detail, str) else "HTTP error",
        error_code="HTTP_ERROR",
        status_code=exc.status_code,
        request_id=request_id,
    )

    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=exc.status_code, content=error_response, headers=headers
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: The FastAPI request object.
        exc: The unexpected exception.

    Returns:
        JSONResponse: Generic error response.
    """
    log_exception(exc, request, {"traceback": traceback.format_exc()})

    request_id = request.headers.get("X-Request-ID")

    # Don't leak internal error details in production
    if settings.debug:
        message = str(exc)
        details = {"traceback": traceback.format_exc()}
    else:
        message = "An unexpected error occurred"
        details = None

    error_response = create_error_response(
        message=message,
        error_code="INTERNAL_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details=details,
        request_id=request_id,
    )

    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response,
        headers=headers,
    )


# Exception handler mapping
EXCEPTION_HANDLERS = {
    TripPlannerBaseException: trip_planner_exception_handler,
    PydanticValidationError: validation_exception_handler,
    HTTPException: http_exception_handler,
    StarletteHTTPException: http_exception_handler,
    Exception: generic_exception_handler,
}


def register_exception_handlers(app) -> None:
    """Register all exception handlers with the FastAPI app.

    Args:
        app: The FastAPI application instance.
    """
    for exc_class, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_class, handler)

    logger.info(
        "Exception handlers registered", handlers=list(EXCEPTION_HANDLERS.keys())
    )
