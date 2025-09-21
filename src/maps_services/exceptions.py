"""Maps Service Exceptions for Google Maps API Integration.

This module provides custom exceptions for Maps API operations including
geocoding, place search, directions, and API-specific error handling.
"""

from typing import Any, Dict, List, Optional

from src.core.exceptions import ExternalServiceException, TripPlannerBaseException


class MapsServiceError(ExternalServiceException):
    """Base exception for Maps service operations."""

    def __init__(
        self,
        message: str = "Maps service error",
        api_status: Optional[str] = None,
        details: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.api_status = api_status
        super().__init__(
            service="Google Maps API", message=message, details=details, **kwargs
        )


class MapsAPIKeyError(MapsServiceError):
    """Raised when Google Maps API key is invalid or missing."""

    def __init__(
        self, message: str = "Invalid or missing Google Maps API key", **kwargs
    ) -> None:
        super().__init__(message=message, api_status="REQUEST_DENIED", **kwargs)


class MapsQuotaExceededError(MapsServiceError):
    """Raised when Google Maps API quota is exceeded."""

    def __init__(
        self,
        message: str = "Google Maps API quota exceeded",
        quota_type: Optional[str] = None,
        reset_time: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.quota_type = quota_type
        self.reset_time = reset_time
        details = kwargs.get("details", {})
        details.update({"quota_type": quota_type, "reset_time": reset_time})
        super().__init__(
            message=message, api_status="OVER_QUERY_LIMIT", details=details, **kwargs
        )


class MapsRateLimitError(MapsServiceError):
    """Raised when Google Maps API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Google Maps API rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.retry_after = retry_after
        details = kwargs.get("details", {})
        details.update({"retry_after": retry_after})
        super().__init__(
            message=message, api_status="OVER_DAILY_LIMIT", details=details, **kwargs
        )


class MapsServiceUnavailableError(MapsServiceError):
    """Raised when Google Maps API service is temporarily unavailable."""

    def __init__(
        self,
        message: str = "Google Maps API service unavailable",
        retry_after: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.retry_after = retry_after
        details = kwargs.get("details", {})
        details.update({"retry_after": retry_after})
        super().__init__(
            message=message, api_status="UNKNOWN_ERROR", details=details, **kwargs
        )


class InvalidLocationError(MapsServiceError):
    """Raised when location coordinates or address are invalid."""

    def __init__(
        self, location: Optional[str] = None, message: Optional[str] = None, **kwargs
    ) -> None:
        self.location = location
        if not message:
            message = (
                f"Invalid location: {location}"
                if location
                else "Invalid location provided"
            )
        details = kwargs.get("details", {})
        details.update({"location": location})
        super().__init__(
            message=message, api_status="INVALID_REQUEST", details=details, **kwargs
        )


class GeocodingError(MapsServiceError):
    """Raised when geocoding operation fails."""

    def __init__(
        self,
        address: Optional[str] = None,
        message: str = "Geocoding operation failed",
        geocoding_status: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.address = address
        self.geocoding_status = geocoding_status
        details = kwargs.get("details", {})
        details.update({"address": address, "geocoding_status": geocoding_status})
        super().__init__(
            message=message,
            api_status=geocoding_status or "UNKNOWN_ERROR",
            details=details,
            **kwargs,
        )


class ReverseGeocodingError(GeocodingError):
    """Raised when reverse geocoding operation fails."""

    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        message: str = "Reverse geocoding operation failed",
        **kwargs,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        coordinates = f"{latitude},{longitude}" if latitude and longitude else None
        details = kwargs.get("details", {})
        details.update(
            {"latitude": latitude, "longitude": longitude, "coordinates": coordinates}
        )
        super().__init__(
            address=coordinates, message=message, details=details, **kwargs
        )


class PlaceNotFoundError(MapsServiceError):
    """Raised when a place cannot be found."""

    def __init__(
        self,
        place_identifier: Optional[str] = None,
        search_query: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.place_identifier = place_identifier
        self.search_query = search_query

        if not message:
            if place_identifier:
                message = f"Place not found: {place_identifier}"
            elif search_query:
                message = f"No places found for query: {search_query}"
            else:
                message = "Place not found"

        details = kwargs.get("details", {})
        details.update(
            {"place_identifier": place_identifier, "search_query": search_query}
        )
        super().__init__(
            message=message, api_status="ZERO_RESULTS", details=details, **kwargs
        )


class PlaceDetailsError(MapsServiceError):
    """Raised when place details cannot be retrieved."""

    def __init__(
        self, place_id: str, message: str = "Failed to retrieve place details", **kwargs
    ) -> None:
        self.place_id = place_id
        details = kwargs.get("details", {})
        details.update({"place_id": place_id})
        super().__init__(
            message=f"{message}: {place_id}",
            api_status="INVALID_REQUEST",
            details=details,
            **kwargs,
        )


class PlaceSearchError(MapsServiceError):
    """Raised when place search operation fails."""

    def __init__(
        self,
        search_params: Dict[str, Any],
        message: str = "Place search operation failed",
        search_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.search_params = search_params
        self.search_type = search_type
        details = kwargs.get("details", {})
        details.update({"search_params": search_params, "search_type": search_type})
        super().__init__(message=message, details=details, **kwargs)


class DirectionsError(MapsServiceError):
    """Raised when directions calculation fails."""

    def __init__(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        message: str = "Directions calculation failed",
        **kwargs,
    ) -> None:
        self.origin = origin
        self.destination = destination
        details = kwargs.get("details", {})
        details.update({"origin": origin, "destination": destination})
        super().__init__(message=message, details=details, **kwargs)


class RouteNotFoundError(DirectionsError):
    """Raised when no route can be found between locations."""

    def __init__(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        travel_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.travel_mode = travel_mode
        message = f"No route found from {origin} to {destination}"
        if travel_mode:
            message += f" using {travel_mode}"

        details = kwargs.get("details", {})
        details.update({"travel_mode": travel_mode})
        super().__init__(
            origin=origin,
            destination=destination,
            message=message,
            api_status="ZERO_RESULTS",
            details=details,
            **kwargs,
        )


class InvalidWaypointError(DirectionsError):
    """Raised when waypoints are invalid for directions request."""

    def __init__(
        self,
        invalid_waypoints: List[str],
        message: str = "Invalid waypoints provided",
        **kwargs,
    ) -> None:
        self.invalid_waypoints = invalid_waypoints
        details = kwargs.get("details", {})
        details.update({"invalid_waypoints": invalid_waypoints})
        super().__init__(
            message=f"{message}: {', '.join(invalid_waypoints)}",
            api_status="INVALID_REQUEST",
            details=details,
            **kwargs,
        )


class DistanceMatrixError(MapsServiceError):
    """Raised when distance matrix calculation fails."""

    def __init__(
        self,
        origins: Optional[List[str]] = None,
        destinations: Optional[List[str]] = None,
        message: str = "Distance matrix calculation failed",
        **kwargs,
    ) -> None:
        self.origins = origins or []
        self.destinations = destinations or []
        details = kwargs.get("details", {})
        details.update(
            {
                "origins": origins,
                "destinations": destinations,
                "origins_count": len(self.origins),
                "destinations_count": len(self.destinations),
            }
        )
        super().__init__(message=message, details=details, **kwargs)


class MapsTimeoutError(MapsServiceError):
    """Raised when Maps API request times out."""

    def __init__(
        self,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.timeout_duration = timeout_duration
        self.operation = operation

        if not message:
            message = "Maps API request timed out"
            if operation:
                message += f" for {operation}"
            if timeout_duration:
                message += f" after {timeout_duration}s"

        details = kwargs.get("details", {})
        details.update({"timeout_duration": timeout_duration, "operation": operation})
        super().__init__(message=message, details=details, **kwargs)


class MapsConnectionError(MapsServiceError):
    """Raised when connection to Maps API fails."""

    def __init__(
        self,
        message: str = "Failed to connect to Google Maps API",
        connection_error: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.connection_error = connection_error
        details = kwargs.get("details", {})
        details.update({"connection_error": connection_error})
        super().__init__(message=message, details=details, **kwargs)


class InvalidAPIResponseError(MapsServiceError):
    """Raised when Maps API returns invalid or unexpected response format."""

    def __init__(
        self,
        response_data: Optional[Any] = None,
        expected_format: Optional[str] = None,
        message: str = "Invalid API response format",
        **kwargs,
    ) -> None:
        self.response_data = response_data
        self.expected_format = expected_format
        details = kwargs.get("details", {})
        details.update(
            {
                "expected_format": expected_format,
                "response_type": (
                    type(response_data).__name__ if response_data else None
                ),
            }
        )
        super().__init__(message=message, details=details, **kwargs)


class PlacePhotoError(MapsServiceError):
    """Raised when place photo operations fail."""

    def __init__(
        self,
        photo_reference: Optional[str] = None,
        message: str = "Place photo operation failed",
        **kwargs,
    ) -> None:
        self.photo_reference = photo_reference
        details = kwargs.get("details", {})
        details.update({"photo_reference": photo_reference})
        super().__init__(message=message, details=details, **kwargs)


class AutocompleteError(MapsServiceError):
    """Raised when place autocomplete operations fail."""

    def __init__(
        self,
        input_text: Optional[str] = None,
        message: str = "Place autocomplete operation failed",
        **kwargs,
    ) -> None:
        self.input_text = input_text
        details = kwargs.get("details", {})
        details.update({"input_text": input_text})
        super().__init__(message=message, details=details, **kwargs)


class MapsConfigurationError(TripPlannerBaseException):
    """Raised when Maps service configuration is invalid."""

    def __init__(
        self, config_issue: str, message: Optional[str] = None, **kwargs
    ) -> None:
        self.config_issue = config_issue
        if not message:
            message = f"Maps service configuration error: {config_issue}"
        details = kwargs.get("details", {})
        details.update({"config_issue": config_issue})
        super().__init__(message=message, details=details, **kwargs)


def handle_maps_api_error(
    status: str, error_message: Optional[str] = None
) -> MapsServiceError:
    """Convert Maps API status codes to appropriate exceptions.

    Args:
        status: API response status code
        error_message: Optional error message from API

    Returns:
        MapsServiceError: Appropriate exception for the status code
    """
    error_message = error_message or f"Maps API error: {status}"

    status_exceptions = {
        "REQUEST_DENIED": MapsAPIKeyError,
        "OVER_QUERY_LIMIT": MapsQuotaExceededError,
        "OVER_DAILY_LIMIT": MapsRateLimitError,
        "INVALID_REQUEST": InvalidLocationError,
        "ZERO_RESULTS": PlaceNotFoundError,
        "UNKNOWN_ERROR": MapsServiceUnavailableError,
    }

    exception_class = status_exceptions.get(status, MapsServiceError)
    return exception_class(message=error_message, api_status=status)


def is_retryable_error(exception: Exception) -> bool:
    """Check if a Maps API error is retryable.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the error is retryable
    """
    retryable_exceptions = (
        MapsServiceUnavailableError,
        MapsTimeoutError,
        MapsConnectionError,
    )

    retryable_statuses = ("UNKNOWN_ERROR",)

    if isinstance(exception, retryable_exceptions):
        return True

    return (
        isinstance(exception, MapsServiceError)
        and exception.api_status in retryable_statuses
    )


def get_retry_delay(exception: Exception, attempt: int) -> int:
    """Get suggested retry delay for Maps API errors.

    Args:
        exception: The exception that occurred
        attempt: Current retry attempt number

    Returns:
        int: Suggested delay in seconds
    """
    base_delay = min(2**attempt, 60)  # Exponential backoff with max 60s

    if isinstance(exception, MapsRateLimitError) and exception.retry_after:
        return exception.retry_after

    if isinstance(exception, MapsServiceUnavailableError) and exception.retry_after:
        return exception.retry_after

    return base_delay
