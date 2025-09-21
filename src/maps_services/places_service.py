"""Places Service for Google Maps API Integration.

This module provides high-level place search and details functionality
with proper error handling, validation, and response parsing.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from src.core.logging import get_logger
from src.maps_services.exceptions import (
    AutocompleteError,
    InvalidLocationError,
    PlaceDetailsError,
    PlaceNotFoundError,
    PlacePhotoError,
    PlaceSearchError,
)
from src.maps_services.maps_client import get_maps_client
from src.maps_services.schemas import (
    GeoLocation,
    PlaceDetails,
    PlaceSearchResponse,
    PlaceSearchResult,
    PlaceType,
    PriceLevel,
)

logger = get_logger(__name__)


class PlacesService:
    """Service for Google Maps Places API operations."""

    def __init__(self) -> None:
        """Initialize Places service."""
        self._client = None

    async def _get_client(self):
        """Get Maps API client instance."""
        if not self._client:
            self._client = await get_maps_client()
        return self._client

    def _parse_place_details(self, place_data: Dict[str, Any]) -> PlaceDetails:
        """Parse raw place data into PlaceDetails model."""
        try:
            # Handle geometry conversion
            geometry_data = place_data.get("geometry", {})
            if "location" in geometry_data:
                location = geometry_data["location"]
                geometry_data["location"] = {
                    "lat": location.get("lat"),
                    "lng": location.get("lng"),
                }

            # Handle bounds if present
            if "bounds" in geometry_data:
                bounds = geometry_data["bounds"]
                if "northeast" in bounds and "southwest" in bounds:
                    geometry_data["bounds"] = {
                        "northeast": {
                            "lat": bounds["northeast"].get("lat"),
                            "lng": bounds["northeast"].get("lng"),
                        },
                        "southwest": {
                            "lat": bounds["southwest"].get("lat"),
                            "lng": bounds["southwest"].get("lng"),
                        },
                    }

            # Handle viewport if present
            if "viewport" in geometry_data:
                viewport = geometry_data["viewport"]
                if "northeast" in viewport and "southwest" in viewport:
                    geometry_data["viewport"] = {
                        "northeast": {
                            "lat": viewport["northeast"].get("lat"),
                            "lng": viewport["northeast"].get("lng"),
                        },
                        "southwest": {
                            "lat": viewport["southwest"].get("lat"),
                            "lng": viewport["southwest"].get("lng"),
                        },
                    }

            # Convert price_level to enum if present
            if "price_level" in place_data and place_data["price_level"] is not None:
                place_data["price_level"] = PriceLevel(place_data["price_level"])

            return PlaceDetails.model_validate(place_data)

        except ValidationError as e:
            logger.exception(
                "Failed to parse place details",
                place_id=place_data.get("place_id"),
                validation_errors=e.errors(),
            )
            raise PlaceDetailsError(
                place_id=place_data.get("place_id", "unknown"),
                message="Failed to parse place details data",
            ) from e

    def _parse_place_search_result(
        self, place_data: Dict[str, Any]
    ) -> PlaceSearchResult:
        """Parse raw place data into PlaceSearchResult model."""
        try:
            # Handle geometry conversion
            geometry_data = place_data.get("geometry", {})
            if "location" in geometry_data:
                location = geometry_data["location"]
                geometry_data["location"] = {
                    "lat": location.get("lat"),
                    "lng": location.get("lng"),
                }

            # Convert price_level to enum if present
            if "price_level" in place_data and place_data["price_level"] is not None:
                place_data["price_level"] = PriceLevel(place_data["price_level"])

            return PlaceSearchResult.model_validate(place_data)

        except ValidationError as e:
            logger.exception(
                "Failed to parse place search result",
                place_id=place_data.get("place_id"),
                validation_errors=e.errors(),
            )
            raise PlaceSearchError(
                search_params={"place_data": place_data.get("place_id", "unknown")},
                message="Failed to parse place search result data",
            ) from e

    async def search_nearby(
        self,
        location: Union[GeoLocation, Dict[str, float]],
        radius: int,
        keyword: Optional[str] = None,
        place_type: Optional[Union[PlaceType, str]] = None,
        min_price: Optional[PriceLevel] = None,
        max_price: Optional[PriceLevel] = None,
        open_now: Optional[bool] = None,
        language: str = "en",
        page_token: Optional[str] = None,
    ) -> PlaceSearchResponse:
        """Search for places near a location.

        Args:
            location: Center location for search
            radius: Search radius in meters (max 50,000)
            keyword: Search keyword
            place_type: Place type filter
            min_price: Minimum price level
            max_price: Maximum price level
            open_now: Only return currently open places
            language: Response language
            page_token: Token for next page of results

        Returns:
            PlaceSearchResponse: Search results with places

        Raises:
            PlaceSearchError: If search fails
            InvalidLocationError: If location is invalid
        """
        try:
            # Validate inputs
            if isinstance(location, dict):
                location = GeoLocation.model_validate(location)

            if radius <= 0 or radius > 50000:
                raise InvalidLocationError(
                    message="Search radius must be between 1 and 50,000 meters"
                )

            # Prepare search parameters
            params = {
                "location": location,
                "radius": radius,
                "language": language,
            }

            if keyword:
                params["keyword"] = keyword
            if place_type:
                params["type"] = (
                    place_type.value
                    if isinstance(place_type, PlaceType)
                    else place_type
                )
            if min_price is not None:
                params["minprice"] = min_price.value
            if max_price is not None:
                params["maxprice"] = max_price.value
            if open_now is not None:
                params["opennow"] = open_now
            if page_token:
                params["pagetoken"] = page_token

            # Make API request with proper parameter types
            client = await self._get_client()

            # Build kwargs for client method
            client_params: Dict[str, Any] = {"language": language}
            if keyword:
                client_params["keyword"] = keyword
            if place_type:
                client_params["type"] = (
                    place_type.value
                    if isinstance(place_type, PlaceType)
                    else place_type
                )
            if min_price is not None:
                client_params["minprice"] = min_price.value
            if max_price is not None:
                client_params["maxprice"] = max_price.value
            if open_now is not None:
                client_params["opennow"] = open_now
            if page_token:
                client_params["pagetoken"] = page_token

            response = await client.places_nearby(
                location=location, radius=radius, **client_params
            )

            # Parse results
            results = []
            for place_data in response.get("results", []):
                try:
                    result = self._parse_place_search_result(place_data)
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        "Skipping invalid place result",
                        place_id=place_data.get("place_id"),
                        error=str(e),
                    )

            return PlaceSearchResponse(
                status=response.get("status", "OK"),
                results=results,
                html_attributions=response.get("html_attributions", []),
                next_page_token=response.get("next_page_token"),
                error_message=response.get("error_message"),
                info_messages=response.get("info_messages"),
            )

        except Exception as e:
            if isinstance(e, PlaceSearchError | InvalidLocationError):
                raise

            logger.error(
                "Nearby places search failed",
                location=str(location),
                radius=radius,
                keyword=keyword,
                error=str(e),
                exc_info=True,
            )
            raise PlaceSearchError(
                search_params={
                    "location": str(location),
                    "radius": radius,
                    "keyword": keyword,
                },
                search_type="nearby",
                message="Nearby places search failed",
            ) from e

    async def search_text(
        self,
        query: str,
        location: Optional[Union[GeoLocation, Dict[str, float]]] = None,
        radius: Optional[int] = None,
        place_type: Optional[Union[PlaceType, str]] = None,
        min_price: Optional[PriceLevel] = None,
        max_price: Optional[PriceLevel] = None,
        open_now: Optional[bool] = None,
        language: str = "en",
        region: Optional[str] = None,
        page_token: Optional[str] = None,
    ) -> PlaceSearchResponse:
        """Search for places using text query.

        Args:
            query: Search query text
            location: Bias location for search
            radius: Search radius in meters
            place_type: Place type filter
            min_price: Minimum price level
            max_price: Maximum price level
            open_now: Only return currently open places
            language: Response language
            region: Region bias (country code)
            page_token: Token for next page of results

        Returns:
            PlaceSearchResponse: Search results with places

        Raises:
            PlaceSearchError: If search fails
        """
        try:
            if not query or not query.strip():
                raise PlaceSearchError(
                    search_params={"query": query},
                    message="Search query cannot be empty",
                )

            # Prepare search parameters
            params: Dict[str, Any] = {
                "query": query.strip(),
                "language": language,
            }

            if location:
                if isinstance(location, dict):
                    location = GeoLocation.model_validate(location)
                params["location"] = location

            if radius:
                params["radius"] = radius
            if place_type:
                params["type"] = (
                    place_type.value
                    if isinstance(place_type, PlaceType)
                    else place_type
                )
            if min_price is not None:
                params["minprice"] = min_price.value
            if max_price is not None:
                params["maxprice"] = max_price.value
            if open_now is not None:
                params["opennow"] = open_now
            if region:
                params["region"] = region
            if page_token:
                params["pagetoken"] = page_token

            # Make API request with proper parameter types
            client = await self._get_client()

            # Build kwargs for client method
            client_params: Dict[str, Any] = {"language": language}
            if location:
                client_params["location"] = location
            if radius:
                client_params["radius"] = radius
            if place_type:
                client_params["type"] = (
                    place_type.value
                    if isinstance(place_type, PlaceType)
                    else place_type
                )
            if min_price is not None:
                client_params["minprice"] = min_price.value
            if max_price is not None:
                client_params["maxprice"] = max_price.value
            if open_now is not None:
                client_params["opennow"] = open_now
            if region:
                client_params["region"] = region
            if page_token:
                client_params["pagetoken"] = page_token

            response = await client.places_text_search(query=query, **client_params)

            # Parse results
            results = []
            for place_data in response.get("results", []):
                try:
                    result = self._parse_place_search_result(place_data)
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        "Skipping invalid place result",
                        place_id=place_data.get("place_id"),
                        error=str(e),
                    )

            return PlaceSearchResponse(
                status=response.get("status", "OK"),
                results=results,
                html_attributions=response.get("html_attributions", []),
                next_page_token=response.get("next_page_token"),
                error_message=response.get("error_message"),
                info_messages=response.get("info_messages"),
            )

        except Exception as e:
            if isinstance(e, PlaceSearchError):
                raise

            logger.error(
                "Text places search failed",
                query=query,
                error=str(e),
                exc_info=True,
            )
            raise PlaceSearchError(
                search_params={"query": query},
                search_type="text",
                message="Text places search failed",
            ) from e

    async def get_place_details(
        self,
        place_id: str,
        fields: Optional[List[str]] = None,
        language: str = "en",
        region: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> PlaceDetails:
        """Get detailed information about a place.

        Args:
            place_id: Place ID from search results
            fields: Specific fields to retrieve
            language: Response language
            region: Region bias (country code)
            session_token: Session token for billing

        Returns:
            PlaceDetails: Detailed place information

        Raises:
            PlaceDetailsError: If details retrieval fails
            PlaceNotFoundError: If place is not found
        """
        try:
            if not place_id or not place_id.strip():
                raise PlaceDetailsError(
                    place_id=place_id or "empty",
                    message="Place ID cannot be empty",
                )

            # Prepare parameters
            params: Dict[str, Any] = {
                "place_id": place_id.strip(),
                "language": language,
            }

            if fields:
                params["fields"] = ",".join(fields)
            if region:
                params["region"] = region
            if session_token:
                params["sessiontoken"] = session_token

            # Make API request
            client = await self._get_client()
            response = await client.place_details(**params)

            # Check if place was found
            if response.get("status") == "ZERO_RESULTS":
                raise PlaceNotFoundError(
                    place_identifier=place_id,
                    message=f"Place not found: {place_id}",
                )

            # Parse place details
            place_data = response.get("result", {})
            if not place_data:
                raise PlaceDetailsError(
                    place_id=place_id,
                    message="Empty place details in API response",
                )

            return self._parse_place_details(place_data)

        except Exception as e:
            if isinstance(e, PlaceDetailsError | PlaceNotFoundError):
                raise

            logger.error(
                "Place details retrieval failed",
                place_id=place_id,
                error=str(e),
                exc_info=True,
            )
            raise PlaceDetailsError(
                place_id=place_id,
                message="Place details retrieval failed",
            ) from e

    async def autocomplete(
        self,
        input_text: str,
        location: Optional[Union[GeoLocation, Dict[str, float]]] = None,
        radius: Optional[int] = None,
        language: str = "en",
        types: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        strict_bounds: bool = False,
        session_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get place autocomplete suggestions.

        Args:
            input_text: Input text to autocomplete
            location: Bias location for suggestions
            radius: Bias radius in meters
            language: Response language
            types: Place type restrictions
            components: Country component restrictions
            strict_bounds: Restrict results to location bounds
            session_token: Session token for billing

        Returns:
            List[Dict[str, Any]]: Autocomplete predictions

        Raises:
            AutocompleteError: If autocomplete fails
        """
        try:
            if not input_text or not input_text.strip():
                raise AutocompleteError(
                    input_text=input_text,
                    message="Input text cannot be empty",
                )

            # Prepare parameters
            params: Dict[str, Any] = {
                "input": input_text.strip(),
                "language": language,
            }

            if location:
                if isinstance(location, dict):
                    location = GeoLocation.model_validate(location)
                params["location"] = location

            if radius:
                params["radius"] = radius
            if types:
                params["types"] = "|".join(types)
            if components:
                params["components"] = "|".join(components)
            if strict_bounds:
                params["strictbounds"] = strict_bounds
            if session_token:
                params["sessiontoken"] = session_token

            # Make API request with proper parameter types
            client = await self._get_client()

            # Build kwargs for client method
            client_params: Dict[str, Any] = {"language": language}
            if location:
                client_params["location"] = location
            if radius:
                client_params["radius"] = radius
            if types:
                client_params["types"] = "|".join(types)
            if components:
                client_params["components"] = "|".join(components)
            if strict_bounds:
                client_params["strictbounds"] = strict_bounds
            if session_token:
                client_params["sessiontoken"] = session_token

            response = await client.place_autocomplete(
                input=input_text, **client_params
            )

            return response.get("predictions", [])

        except Exception as e:
            if isinstance(e, AutocompleteError):
                raise

            logger.error(
                "Place autocomplete failed",
                input_text=input_text,
                error=str(e),
                exc_info=True,
            )
            raise AutocompleteError(
                input_text=input_text,
                message="Place autocomplete failed",
            ) from e

    async def get_place_photo(
        self,
        photo_reference: str,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> bytes:
        """Get place photo data.

        Args:
            photo_reference: Photo reference from place details
            max_width: Maximum photo width
            max_height: Maximum photo height

        Returns:
            bytes: Photo image data

        Raises:
            PlacePhotoError: If photo retrieval fails
        """
        try:
            if not photo_reference or not photo_reference.strip():
                raise PlacePhotoError(
                    photo_reference=photo_reference,
                    message="Photo reference cannot be empty",
                )

            # Make API request
            client = await self._get_client()
            photo_data = await client.place_photo(
                photo_reference=photo_reference.strip(),
                max_width=max_width,
                max_height=max_height,
            )

            return photo_data

        except Exception as e:
            if isinstance(e, PlacePhotoError):
                raise

            logger.error(
                "Place photo retrieval failed",
                photo_reference=photo_reference,
                error=str(e),
                exc_info=True,
            )
            raise PlacePhotoError(
                photo_reference=photo_reference,
                message="Place photo retrieval failed",
            ) from e

    async def find_popular_places(
        self,
        location: Union[GeoLocation, Dict[str, float]],
        radius: int = 5000,
        place_types: Optional[List[PlaceType]] = None,
        min_rating: float = 4.0,
        language: str = "en",
    ) -> List[PlaceSearchResult]:
        """Find popular places near a location.

        Args:
            location: Center location for search
            radius: Search radius in meters
            place_types: Types of places to search for
            min_rating: Minimum rating threshold
            language: Response language

        Returns:
            List[PlaceSearchResult]: Popular places sorted by rating

        Raises:
            PlaceSearchError: If search fails
        """
        try:
            if not place_types:
                place_types = [
                    PlaceType.TOURIST_ATTRACTION,
                    PlaceType.RESTAURANT,
                    PlaceType.MUSEUM,
                    PlaceType.PARK,
                    PlaceType.AMUSEMENT_PARK,
                ]

            all_results = []

            # Search for each place type
            for place_type in place_types:
                try:
                    response = await self.search_nearby(
                        location=location,
                        radius=radius,
                        place_type=place_type,
                        language=language,
                    )

                    # Filter by rating
                    for result in response.results:
                        if result.rating and result.rating >= min_rating:
                            all_results.append(result)

                except PlaceSearchError as e:
                    logger.warning(
                        "Failed to search for place type",
                        place_type=place_type,
                        error=str(e),
                    )
                    continue

            # Sort by rating (descending) and user ratings total
            all_results.sort(
                key=lambda x: (x.rating or 0, x.user_ratings_total or 0), reverse=True
            )

            # Remove duplicates by place_id
            seen_places = set()
            unique_results = []
            for result in all_results:
                if result.place_id not in seen_places:
                    seen_places.add(result.place_id)
                    unique_results.append(result)

            return unique_results

        except Exception as e:
            logger.error(
                "Popular places search failed",
                location=str(location),
                error=str(e),
                exc_info=True,
            )
            raise PlaceSearchError(
                search_params={
                    "location": str(location),
                    "radius": radius,
                    "min_rating": min_rating,
                },
                search_type="popular",
                message="Popular places search failed",
            ) from e


# Global service instance
_places_service: Optional[PlacesService] = None


def get_places_service() -> PlacesService:
    """Get or create global Places service instance."""
    global _places_service

    if _places_service is None:
        _places_service = PlacesService()

    return _places_service
