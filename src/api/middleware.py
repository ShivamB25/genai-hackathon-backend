"""Enhanced middleware for AI-Powered Trip Planner Backend.

This module provides comprehensive middleware for security, rate limiting,
performance monitoring, and request correlation tracking.
"""

import time
import uuid
from collections import defaultdict, deque
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings
from src.core.logging import clear_request_id, get_logger, set_request_id
from src.core.security import get_security_headers

logger = get_logger(__name__)


# Rate Limiting Storage
class MemoryRateLimitStore:
    """In-memory rate limiting store with sliding window."""

    def __init__(self) -> None:
        self._store: dict[str, deque] = defaultdict(lambda: deque())
        self._last_cleanup = time.time()

    def is_allowed(
        self, key: str, limit: int, window: int
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is allowed under rate limit.

        Args:
            key: Rate limit key.
            limit: Maximum requests allowed.
            window: Time window in seconds.

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_allowed, rate_limit_info)
        """
        current_time = time.time()

        # Cleanup old entries periodically
        if current_time - self._last_cleanup > 60:  # Cleanup every minute
            self._cleanup_expired(current_time)
            self._last_cleanup = current_time

        # Get request timestamps for this key
        requests = self._store[key]

        # Remove expired requests
        cutoff_time = current_time - window
        while requests and requests[0] <= cutoff_time:
            requests.popleft()

        # Check if limit is exceeded
        request_count = len(requests)
        is_allowed = request_count < limit

        if is_allowed:
            requests.append(current_time)

        # Calculate time until reset
        reset_time = (
            int(requests[0] + window) if requests else int(current_time + window)
        )
        remaining = max(0, limit - request_count - (1 if is_allowed else 0))

        rate_limit_info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "retry_after": (
                max(1, int(reset_time - current_time)) if not is_allowed else None
            ),
        }

        return is_allowed, rate_limit_info

    def _cleanup_expired(self, current_time: float) -> None:
        """Remove expired entries to prevent memory leaks."""
        cutoff_time = current_time - 3600  # Keep entries for 1 hour max

        keys_to_remove = []
        for key, requests in self._store.items():
            while requests and requests[0] <= cutoff_time:
                requests.popleft()

            if not requests:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._store[key]


# Global rate limit store
rate_limit_store = MemoryRateLimitStore()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for headers and request validation."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with security checks.

        Args:
            request: The incoming request.
            call_next: Next middleware in chain.

        Returns:
            Response: The response with security headers.
        """
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > 10_000_000:  # 10MB limit
                    logger.warning("Request too large", size=size, url=str(request.url))
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"error": "Request too large"},
                    )
            except ValueError:
                logger.warning(
                    "Invalid content-length header", content_length=content_length
                )

        # Check for suspicious headers
        suspicious_headers = [
            "x-forwarded-host",
            "x-original-host",
            "x-rewrite-url",
            "x-original-url",
            "x-arbitrary-header",
        ]

        for header in suspicious_headers:
            if header in request.headers:
                logger.warning(
                    "Suspicious header detected",
                    header=header,
                    value=request.headers[header][:100],
                    client_ip=request.client.host if request.client else "unknown",
                )

        # Process request
        response = await call_next(request)

        # Add security headers
        security_headers = get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with configurable limits."""

    def __init__(
        self, app, default_limit: int | None = None, default_window: int = 3600
    ) -> None:
        super().__init__(app)
        self.default_limit = default_limit or settings.api_rate_limit
        self.default_window = default_window

        # Route-specific limits
        self.route_limits = {
            "/api/v1/users/complete-profile": (3, 600),  # 3 writes per 10 minutes
            "/api/v1/users": (50, 3600),  # 50 user operations per hour
            "/api/v1/trips/plan": (20, 3600),  # 20 trip generations per hour
            "/api/v1/trips": (100, 3600),  # 100 trip reads/updates per hour
            "/api/v1/places": (100, 3600),  # 100 maps requests per hour
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting.

        Args:
            request: The incoming request.
            call_next: Next middleware in chain.

        Returns:
            Response: The response with rate limit headers.
        """
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)

        # Determine rate limit key
        rate_limit_key = self._get_rate_limit_key(request)

        # Get limits for this route
        limit, window = self._get_limits_for_path(request.url.path)

        # Check rate limit
        is_allowed, rate_info = rate_limit_store.is_allowed(
            rate_limit_key, limit, window
        )

        if not is_allowed:
            logger.warning(
                "Rate limit exceeded",
                key=rate_limit_key,
                path=request.url.path,
                limit=limit,
                window=window,
                client_ip=request.client.host if request.client else "unknown",
            )

            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {limit} per {window} seconds",
                    "retry_after": rate_info["retry_after"],
                },
            )
        else:
            # Process request
            response = await call_next(request)

        # Add rate limit headers
        response.headers.update(
            {
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset"]),
            }
        )

        if rate_info["retry_after"]:
            response.headers["Retry-After"] = str(rate_info["retry_after"])

        return response

    def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key for the request.

        Args:
            request: The incoming request.

        Returns:
            str: The rate limit key.
        """
        # Try to get authenticated user ID
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            # Use a hash of the token as identifier
            import hashlib

            token_hash = hashlib.blake2s(
                auth_header.encode(), digest_size=16
            ).hexdigest()[:8]
            return f"user:{token_hash}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"

        # Consider X-Forwarded-For if behind proxy (be careful about spoofing)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for and not settings.debug:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"

    def _get_limits_for_path(self, path: str) -> tuple[int, int]:
        """Get rate limits for specific path.

        Args:
            path: The request path.

        Returns:
            Tuple[int, int]: (limit, window_seconds)
        """
        normalized_path = path.rstrip("/") or "/"

        # Check exact matches first
        if normalized_path in self.route_limits:
            return self.route_limits[normalized_path]

        # Check prefix matches
        for route_pattern, (limit, window) in self.route_limits.items():
            pattern = route_pattern.rstrip("/")
            if normalized_path.startswith(pattern):
                return limit, window

        # Use default limits
        return self.default_limit, self.default_window


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware for request correlation ID tracking."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with correlation ID.

        Args:
            request: The incoming request.
            call_next: Next middleware in chain.

        Returns:
            Response: The response with correlation ID header.
        """
        # Get or generate request ID
        request_id = (
            request.headers.get("x-request-id")
            or request.headers.get("x-correlation-id")
            or str(uuid.uuid4())
        )

        # Set request ID in logging context
        set_request_id(request_id)

        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = request_id

            return response

        finally:
            # Clear request ID from context
            clear_request_id()


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics."""

    def __init__(self, app, slow_request_threshold: float = 5.0) -> None:
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with performance monitoring.

        Args:
            request: The incoming request.
            call_next: Next middleware in chain.

        Returns:
            Response: The response with performance headers.
        """
        start_time = time.time()

        # Add request start time to request state
        request.state.start_time = start_time

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Add performance headers
            response.headers["X-Response-Time"] = f"{processing_time:.3f}s"
            response.headers["X-Process-Time"] = str(
                int(processing_time * 1000)
            )  # milliseconds

            # Log performance metrics
            self._log_performance_metrics(request, response, processing_time)

            # Log slow requests
            if processing_time > self.slow_request_threshold:
                logger.warning(
                    "Slow request detected",
                    method=request.method,
                    url=str(request.url),
                    processing_time=processing_time,
                    status_code=response.status_code,
                    client_ip=request.client.host if request.client else "unknown",
                )

            return response

        except Exception as exc:
            # Log error with timing
            end_time = time.time()
            processing_time = end_time - start_time

            logger.error(
                "Request failed with exception",
                method=request.method,
                url=str(request.url),
                processing_time=processing_time,
                error=str(exc),
                exc_info=True,
            )

            raise

    def _log_performance_metrics(
        self, request: Request, response: Response, processing_time: float
    ) -> None:
        """Log performance metrics for the request.

        Args:
            request: The request object.
            response: The response object.
            processing_time: Request processing time in seconds.
        """
        # Log with appropriate level based on performance
        if processing_time > self.slow_request_threshold:
            log_level = "warning"
        elif processing_time > 1.0:
            log_level = "info"
        else:
            log_level = "debug"

        log_method = getattr(logger, log_level)
        log_method(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            processing_time=processing_time,
            response_size=len(response.body) if hasattr(response, "body") else None,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
        )


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with validation.

        Args:
            request: The incoming request.
            call_next: Next middleware in chain.

        Returns:
            Response: The response after validation.
        """
        # Validate HTTP method
        if request.method not in [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "HEAD",
            "OPTIONS",
        ]:
            logger.warning(
                "Invalid HTTP method", method=request.method, url=str(request.url)
            )
            return JSONResponse(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                content={"error": "Method not allowed"},
            )

        # Validate URL path length
        if len(request.url.path) > 2048:
            logger.warning("URL path too long", path_length=len(request.url.path))
            return JSONResponse(
                status_code=status.HTTP_414_URI_TOO_LONG,
                content={"error": "URL too long"},
            )

        # Validate query string
        if len(str(request.url.query)) > 4096:
            logger.warning(
                "Query string too long", query_length=len(str(request.url.query))
            )
            return JSONResponse(
                status_code=status.HTTP_414_URI_TOO_LONG,
                content={"error": "Query string too long"},
            )

        # Check for null bytes in path (potential security issue)
        if "\x00" in request.url.path:
            logger.warning("Null byte in URL path", path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid URL path"},
            )

        # Process request
        return await call_next(request)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression."""

    def __init__(self, app, minimum_size: int = 500) -> None:
        super().__init__(app)
        self.minimum_size = minimum_size

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with compression support.

        Args:
            request: The incoming request.
            call_next: Next middleware in chain.

        Returns:
            Response: The potentially compressed response.
        """
        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "")
        supports_gzip = "gzip" in accept_encoding.lower()

        # Process request
        response = await call_next(request)

        # Add compression hint header
        if supports_gzip:
            response.headers["Vary"] = "Accept-Encoding"

        return response


# Middleware factory functions
def create_security_middleware():
    """Create security middleware instance."""
    return SecurityMiddleware


def create_rate_limit_middleware(limit: int | None = None, window: int = 3600):
    """Create rate limiting middleware instance.

    Args:
        limit: Request limit (uses settings default if None).
        window: Time window in seconds.

    Returns:
        RateLimitMiddleware: Configured middleware instance.
    """
    return lambda app: RateLimitMiddleware(app, limit, window)


def create_correlation_id_middleware():
    """Create correlation ID middleware instance."""
    return CorrelationIdMiddleware


def create_performance_monitoring_middleware(threshold: float = 5.0):
    """Create performance monitoring middleware instance.

    Args:
        threshold: Slow request threshold in seconds.

    Returns:
        PerformanceMonitoringMiddleware: Configured middleware instance.
    """
    return lambda app: PerformanceMonitoringMiddleware(app, threshold)


def create_request_validation_middleware():
    """Create request validation middleware instance."""
    return RequestValidationMiddleware


def create_compression_middleware(minimum_size: int = 500):
    """Create compression middleware instance.

    Args:
        minimum_size: Minimum response size for compression.

    Returns:
        CompressionMiddleware: Configured middleware instance.
    """
    return lambda app: CompressionMiddleware(app, minimum_size)


# Middleware registration helper
def register_middleware(app) -> None:
    """Register all middleware with the FastAPI app in correct order.

    Args:
        app: The FastAPI application instance.
    """
    # Middleware is applied in reverse order, so list them from last to first

    # Last: Compression (closest to response)
    if settings.environment == "production":
        app.add_middleware(CompressionMiddleware, minimum_size=500)

    # Performance monitoring
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=5.0)

    # Security headers
    app.add_middleware(SecurityMiddleware)

    # Rate limiting
    app.add_middleware(RateLimitMiddleware, default_limit=settings.api_rate_limit)

    # Request validation
    app.add_middleware(RequestValidationMiddleware)

    # First: Correlation ID (closest to request)
    app.add_middleware(CorrelationIdMiddleware)

    logger.info("Middleware registered", environment=settings.environment)
