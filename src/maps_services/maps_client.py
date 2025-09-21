"""Google Maps API Client for AI-Powered Trip Planner Backend.

This module provides a comprehensive async HTTP client for Google Maps API
with authentication, rate limiting, caching, error handling, and monitoring.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import httpx
from cachetools import TTLCache

from src.core.config import settings
from src.core.logging import get_logger
from src.maps_services.exceptions import (
    InvalidAPIResponseError,
    MapsConfigurationError,
    MapsConnectionError,
    MapsTimeoutError,
    get_retry_delay,
    handle_maps_api_error,
    is_retryable_error,
)
from src.maps_services.schemas import GeoLocation

logger = get_logger(__name__)


class RateLimiter:
    """Rate limiter for Google Maps API requests."""

    def __init__(
        self, requests_per_second: float = 10.0, burst_limit: int = 20
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst_limit: Maximum burst requests allowed
        """
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self.tokens = float(burst_limit)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a rate limit token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_limit, self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return

            # Need to wait for next token
            wait_time = (1.0 - self.tokens) / self.requests_per_second
            logger.debug("Rate limit reached, waiting", wait_time=wait_time)
            await asyncio.sleep(wait_time)
            self.tokens = 0.0


class RequestCache:
    """Simple TTL cache for Maps API responses."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600) -> None:
        """Initialize request cache.

        Args:
            max_size: Maximum number of cached entries
            ttl: Time to live in seconds
        """
        self._cache = TTLCache(maxsize=max_size, ttl=ttl)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        async with self._lock:
            return self._cache.get(key)

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """Cache response."""
        async with self._lock:
            self._cache[key] = value

    def _generate_cache_key(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key from URL and parameters."""
        # Sort parameters for consistent keys
        sorted_params = sorted(params.items())
        cache_data = f"{url}:{json.dumps(sorted_params, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()


class MapsAPIClient:
    """Google Maps API client with authentication, rate limiting, and caching."""

    BASE_URL = "https://maps.googleapis.com/maps/api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_second: float = 10.0,
        cache_ttl: int = 3600,
        cache_size: int = 1000,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """Initialize Maps API client.

        Args:
            api_key: Google Maps API key
            requests_per_second: Rate limit for requests
            cache_ttl: Cache time to live in seconds
            cache_size: Maximum cache size
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key or settings.google_maps_api_key
        if not self.api_key:
            raise MapsConfigurationError("Google Maps API key not configured")

        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize components
        self.rate_limiter = RateLimiter(requests_per_second)
        self.cache = RequestCache(max_size=cache_size, ttl=cache_ttl)

        # HTTP client will be initialized lazily
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

        # Request statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limited": 0,
            "errors": 0,
            "retries": 0,
        }

    async def __aenter__(self) -> "MapsAPIClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout),
                        headers={
                            "User-Agent": f"{settings.app_name}/{settings.app_version}",
                            "Accept": "application/json",
                        },
                        limits=httpx.Limits(
                            max_keepalive_connections=20,
                            max_connections=100,
                        ),
                    )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_url(self, endpoint: str) -> str:
        """Build full API URL."""
        return f"{self.BASE_URL}/{endpoint.lstrip('/')}"

    def _prepare_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request parameters with API key."""
        prepared_params = params.copy()
        prepared_params["key"] = self.api_key

        # Convert GeoLocation objects to string format
        for key, value in prepared_params.items():
            if isinstance(value, GeoLocation):
                prepared_params[key] = value.to_string()
            elif (
                isinstance(value, list) and value and isinstance(value[0], GeoLocation)
            ):
                prepared_params[key] = "|".join(loc.to_string() for loc in value)

        return prepared_params

    def _generate_cache_key(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        # Exclude API key from cache key for security
        cache_params = {k: v for k, v in params.items() if k != "key"}
        cache_data = f"{url}:{json.dumps(cache_params, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        use_cache: bool = True,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Make HTTP request to Google Maps API."""
        url = self._build_url(endpoint)
        prepared_params = self._prepare_params(params)

        # Check cache first
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(url, prepared_params)
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                self.stats["cache_hits"] += 1
                logger.debug("Cache hit for Maps API request", endpoint=endpoint)
                return cached_response
            self.stats["cache_misses"] += 1

        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Prepare request
        client = await self._ensure_client()
        self.stats["total_requests"] += 1

        start_time = datetime.now(timezone.utc)
        response = None
        try:
            logger.debug(
                "Making Maps API request",
                endpoint=endpoint,
                params_count=len(prepared_params),
                use_cache=use_cache,
                retry_count=retry_count,
            )

            response = await client.get(url, params=prepared_params)
            response.raise_for_status()

            response_data = response.json()
            request_duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.debug(
                "Maps API request completed",
                endpoint=endpoint,
                status_code=response.status_code,
                duration=request_duration,
                api_status=response_data.get("status"),
            )

            # Handle API-specific errors
            api_status = response_data.get("status")
            if api_status not in {"OK", "ZERO_RESULTS"}:
                error_message = response_data.get("error_message")
                exception = handle_maps_api_error(api_status, error_message)

                # Check if we should retry
                if retry_count < self.max_retries and is_retryable_error(exception):
                    retry_delay = get_retry_delay(exception, retry_count + 1)
                    self.stats["retries"] += 1

                    logger.warning(
                        "Maps API request failed, retrying",
                        endpoint=endpoint,
                        api_status=api_status,
                        retry_count=retry_count,
                        retry_delay=retry_delay,
                        error_message=error_message,
                    )

                    await asyncio.sleep(retry_delay)
                    return await self._make_request(
                        endpoint, params, use_cache, retry_count + 1
                    )

                self.stats["errors"] += 1
                raise exception

            # Cache successful responses
            if use_cache and cache_key and api_status == "OK":
                await self.cache.set(cache_key, response_data)

            return response_data

        except httpx.TimeoutException as e:
            exception = MapsTimeoutError(
                timeout_duration=self.timeout,
                operation=endpoint,
                message=f"Maps API request timeout: {e}",
            )

            # Retry on timeout
            if retry_count < self.max_retries:
                retry_delay = get_retry_delay(exception, retry_count + 1)
                self.stats["retries"] += 1

                logger.warning(
                    "Maps API request timeout, retrying",
                    endpoint=endpoint,
                    retry_count=retry_count,
                    retry_delay=retry_delay,
                )

                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    endpoint, params, use_cache, retry_count + 1
                )

            self.stats["errors"] += 1
            raise exception from None

        except httpx.HTTPStatusError as e:
            exception = MapsConnectionError(
                message=f"Maps API HTTP error: {e.response.status_code}",
                connection_error=str(e),
            )

            # Retry on server errors
            if retry_count < self.max_retries and 500 <= e.response.status_code < 600:
                retry_delay = get_retry_delay(exception, retry_count + 1)
                self.stats["retries"] += 1

                logger.warning(
                    "Maps API HTTP error, retrying",
                    endpoint=endpoint,
                    status_code=e.response.status_code,
                    retry_count=retry_count,
                    retry_delay=retry_delay,
                )

                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    endpoint, params, use_cache, retry_count + 1
                )

            self.stats["errors"] += 1
            raise exception from None

        except httpx.RequestError as e:
            exception = MapsConnectionError(
                message=f"Maps API connection error: {e}",
                connection_error=str(e),
            )

            # Retry connection errors
            if retry_count < self.max_retries:
                retry_delay = get_retry_delay(exception, retry_count + 1)
                self.stats["retries"] += 1

                logger.warning(
                    "Maps API connection error, retrying",
                    endpoint=endpoint,
                    retry_count=retry_count,
                    retry_delay=retry_delay,
                    error=str(e),
                )

                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    endpoint, params, use_cache, retry_count + 1
                )

            self.stats["errors"] += 1
            raise exception from None

        except json.JSONDecodeError as e:
            self.stats["errors"] += 1
            response_text = response.text if response else None

            raise InvalidAPIResponseError(
                message="Invalid JSON response from Maps API",
                response_data=response_text,
                expected_format="JSON",
            ) from e

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(
                "Unexpected error in Maps API request",
                endpoint=endpoint,
                error=str(e),
                exc_info=True,
            )
            raise MapsConnectionError(
                message=f"Unexpected error: {e}",
                connection_error=str(e),
            ) from e

    async def geocode(
        self,
        address: Optional[str] = None,
        location: Optional[GeoLocation] = None,
        place_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Geocode an address or reverse geocode coordinates."""
        params = {k: v for k, v in kwargs.items() if v is not None}

        if address:
            params["address"] = address
        elif location:
            params["latlng"] = location
        elif place_id:
            params["place_id"] = place_id
        else:
            raise ValueError("Must provide address, location, or place_id")

        return await self._make_request("geocode/json", params)

    async def places_nearby(
        self, location: GeoLocation, radius: int, **kwargs
    ) -> Dict[str, Any]:
        """Search for places near a location."""
        params = {
            "location": location,
            "radius": radius,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        return await self._make_request("place/nearbysearch/json", params)

    async def places_text_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search for places using text query."""
        params = {"query": query, **{k: v for k, v in kwargs.items() if v is not None}}

        return await self._make_request("place/textsearch/json", params)

    async def place_details(self, place_id: str, **kwargs) -> Dict[str, Any]:
        """Get detailed information about a place."""
        params = {
            "place_id": place_id,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        return await self._make_request("place/details/json", params)

    async def place_autocomplete(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Get place autocomplete suggestions."""
        params = {
            "input": input_text,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        return await self._make_request("place/autocomplete/json", params)

    async def directions(
        self,
        origin: Union[str, GeoLocation],
        destination: Union[str, GeoLocation],
        **kwargs,
    ) -> Dict[str, Any]:
        """Get directions between locations."""
        params = {
            "origin": origin,
            "destination": destination,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        return await self._make_request("directions/json", params)

    async def distance_matrix(
        self,
        origins: List[Union[str, GeoLocation]],
        destinations: List[Union[str, GeoLocation]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Get distance matrix between multiple origins and destinations."""
        # Convert lists to pipe-separated strings
        origin_str = "|".join(
            loc.to_string() if isinstance(loc, GeoLocation) else str(loc)
            for loc in origins
        )
        destination_str = "|".join(
            loc.to_string() if isinstance(loc, GeoLocation) else str(loc)
            for loc in destinations
        )

        params = {
            "origins": origin_str,
            "destinations": destination_str,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        return await self._make_request("distancematrix/json", params)

    async def place_photo(
        self,
        photo_reference: str,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> bytes:
        """Get place photo data."""
        params = {"photo_reference": photo_reference}

        if max_width:
            params["maxwidth"] = str(max_width)
        if max_height:
            params["maxheight"] = str(max_height)

        # Photos return binary data, not JSON
        url = self._build_url("place/photo")
        prepared_params = self._prepare_params(params)

        client = await self._ensure_client()
        response = await client.get(url, params=prepared_params)
        response.raise_for_status()

        return response.content

    def get_stats(self) -> Dict[str, Any]:
        """Get client usage statistics."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"]
                / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
                else 0
            ),
            "error_rate": (
                self.stats["errors"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0
                else 0
            ),
            "retry_rate": (
                self.stats["retries"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0
                else 0
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Maps API."""
        try:
            # Simple geocoding test
            response = await self.geocode(address="Google, Mountain View, CA")

            return {
                "status": "healthy",
                "api_status": response.get("status"),
                "response_time": response.get("response_time", 0),
                "stats": self.get_stats(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats(),
            }


# Global client instance
_maps_client: Optional[MapsAPIClient] = None
_client_lock = asyncio.Lock()


async def get_maps_client() -> MapsAPIClient:
    """Get or create global Maps API client instance."""
    global _maps_client

    if _maps_client is None:
        async with _client_lock:
            if _maps_client is None:
                _maps_client = MapsAPIClient(
                    requests_per_second=settings.maps_api_rate_limit
                    / 60,  # Convert per minute to per second
                    cache_ttl=settings.tool_cache_ttl,
                    timeout=settings.vertex_ai_request_timeout,
                    max_retries=settings.vertex_ai_max_retries,
                )

                logger.info(
                    "Maps API client initialized",
                    rate_limit=settings.maps_api_rate_limit,
                    cache_ttl=settings.tool_cache_ttl,
                )

    return _maps_client


async def close_maps_client() -> None:
    """Close global Maps API client."""
    global _maps_client

    if _maps_client:
        await _maps_client.close()
        _maps_client = None
        logger.info("Maps API client closed")
