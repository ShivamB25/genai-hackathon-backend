"""Geocoding Service for Google Maps API Integration.

This module provides high-level geocoding and reverse geocoding functionality
with proper error handling, validation, and response parsing.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from src.core.logging import get_logger
from src.maps_services.exceptions import (
    GeocodingError,
    InvalidLocationError,
    ReverseGeocodingError,
)
from src.maps_services.maps_client import get_maps_client
from src.maps_services.schemas import (
    AddressComponent,
    GeocodingResponse,
    GeoLocation,
)

logger = get_logger(__name__)


class GeocodingService:
    """Service for Google Maps Geocoding API operations."""

    def __init__(self) -> None:
        """Initialize Geocoding service."""
        self._client = None

    async def _get_client(self):
        """Get Maps API client instance."""
        if not self._client:
            self._client = await get_maps_client()
        return self._client

    def _parse_address_components(
        self, components_data: List[Dict[str, Any]]
    ) -> List[AddressComponent]:
        """Parse address components from API response."""
        components = []
        for component_data in components_data:
            try:
                component = AddressComponent.model_validate(component_data)
                components.append(component)
            except ValidationError as e:
                logger.warning(
                    "Failed to parse address component",
                    component_data=component_data,
                    validation_errors=e.errors(),
                )
        return components

    def _parse_geocoding_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single geocoding result."""
        try:
            # Parse geometry
            if "geometry" in result_data and "location" in result_data["geometry"]:
                location_data = result_data["geometry"]["location"]
                result_data["geometry"]["location"] = GeoLocation.model_validate(
                    location_data
                )

            # Parse address components
            if "address_components" in result_data:
                result_data["address_components"] = self._parse_address_components(
                    result_data["address_components"]
                )

            return result_data

        except Exception as e:
            logger.warning(
                "Failed to parse geocoding result",
                place_id=result_data.get("place_id"),
                error=str(e),
            )
            return result_data

    async def geocode_address(
        self,
        address: str,
        language: str = "en",
        region: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
        bounds: Optional[Dict[str, GeoLocation]] = None,
    ) -> GeocodingResponse:
        """Convert address to geographic coordinates.

        Args:
            address: Address to geocode
            language: Response language
            region: Region bias (country code)
            components: Component filters (country, administrative_area, etc.)
            bounds: Viewport bias bounds

        Returns:
            GeocodingResponse: Geocoding results with coordinates

        Raises:
            GeocodingError: If geocoding fails
            InvalidLocationError: If address format is invalid
        """
        try:
            if not address or not address.strip():
                raise InvalidLocationError(
                    location=address,
                    message="Address cannot be empty",
                )

            # Build parameters
            client_params: Dict[str, Any] = {
                "language": language,
            }

            if region:
                client_params["region"] = region
            if components:
                # Convert components dict to string format
                components_list = [
                    f"{key}:{value}" for key, value in components.items()
                ]
                client_params["components"] = "|".join(components_list)
            if bounds:
                # Format bounds as "lat,lng|lat,lng"
                northeast = bounds.get("northeast")
                southwest = bounds.get("southwest")
                if northeast and southwest:
                    client_params["bounds"] = (
                        f"{southwest.to_string()}|{northeast.to_string()}"
                    )

            # Make API request
            client = await self._get_client()
            response = await client.geocode(address=address.strip(), **client_params)

            # Parse results
            parsed_results = []
            for result_data in response.get("results", []):
                parsed_result = self._parse_geocoding_result(result_data)
                parsed_results.append(parsed_result)

            response["results"] = parsed_results

            # Validate and return response
            return GeocodingResponse.model_validate(response)

        except Exception as e:
            if isinstance(e, GeocodingError | InvalidLocationError):
                raise

            logger.error(
                "Address geocoding failed",
                address=address,
                error=str(e),
                exc_info=True,
            )
            raise GeocodingError(
                address=address,
                message="Address geocoding failed",
            ) from e

    async def reverse_geocode(
        self,
        location: Union[GeoLocation, Dict[str, float]],
        language: str = "en",
        result_type: Optional[List[str]] = None,
        location_type: Optional[List[str]] = None,
    ) -> GeocodingResponse:
        """Convert coordinates to address information.

        Args:
            location: Geographic coordinates
            language: Response language
            result_type: Filter by result type (street_address, route, etc.)
            location_type: Filter by location type (ROOFTOP, RANGE_INTERPOLATED, etc.)

        Returns:
            GeocodingResponse: Reverse geocoding results with addresses

        Raises:
            ReverseGeocodingError: If reverse geocoding fails
            InvalidLocationError: If coordinates are invalid
        """
        try:
            # Validate location
            if isinstance(location, dict):
                location = GeoLocation.model_validate(location)

            # Build parameters
            client_params: Dict[str, Any] = {
                "language": language,
            }

            if result_type:
                client_params["result_type"] = "|".join(result_type)
            if location_type:
                client_params["location_type"] = "|".join(location_type)

            # Make API request
            client = await self._get_client()
            response = await client.geocode(location=location, **client_params)

            # Parse results
            parsed_results = []
            for result_data in response.get("results", []):
                parsed_result = self._parse_geocoding_result(result_data)
                parsed_results.append(parsed_result)

            response["results"] = parsed_results

            # Validate and return response
            return GeocodingResponse.model_validate(response)

        except Exception as e:
            if isinstance(e, ReverseGeocodingError | InvalidLocationError):
                raise

            logger.error(
                "Reverse geocoding failed",
                location=str(location),
                error=str(e),
                exc_info=True,
            )

            # Extract coordinates for error reporting
            lat = (
                location.latitude
                if isinstance(location, GeoLocation)
                else location.get("lat")
            )
            lng = (
                location.longitude
                if isinstance(location, GeoLocation)
                else location.get("lng")
            )

            raise ReverseGeocodingError(
                latitude=lat,
                longitude=lng,
                message="Reverse geocoding failed",
            ) from e

    async def geocode_place_id(
        self,
        place_id: str,
        language: str = "en",
    ) -> GeocodingResponse:
        """Convert place ID to address and coordinates.

        Args:
            place_id: Place ID from Google Maps
            language: Response language

        Returns:
            GeocodingResponse: Place geocoding results

        Raises:
            GeocodingError: If place ID geocoding fails
        """
        try:
            if not place_id or not place_id.strip():
                raise GeocodingError(
                    message="Place ID cannot be empty",
                )

            # Make API request
            client = await self._get_client()
            response = await client.geocode(
                place_id=place_id.strip(),
                language=language,
            )

            # Parse results
            parsed_results = []
            for result_data in response.get("results", []):
                parsed_result = self._parse_geocoding_result(result_data)
                parsed_results.append(parsed_result)

            response["results"] = parsed_results

            # Validate and return response
            return GeocodingResponse.model_validate(response)

        except Exception as e:
            if isinstance(e, GeocodingError):
                raise

            logger.error(
                "Place ID geocoding failed",
                place_id=place_id,
                error=str(e),
                exc_info=True,
            )
            raise GeocodingError(
                message=f"Place ID geocoding failed: {place_id}",
            ) from e

    async def batch_geocode(
        self,
        addresses: List[str],
        language: str = "en",
        region: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
    ) -> List[GeocodingResponse]:
        """Geocode multiple addresses in batch.

        Args:
            addresses: List of addresses to geocode
            language: Response language
            region: Region bias (country code)
            components: Component filters

        Returns:
            List[GeocodingResponse]: List of geocoding results

        Raises:
            GeocodingError: If batch processing fails
        """
        try:
            if not addresses:
                return []

            results = []
            failed_addresses = []

            for address in addresses:
                try:
                    result = await self.geocode_address(
                        address=address,
                        language=language,
                        region=region,
                        components=components,
                    )
                    results.append(result)

                except Exception as e:
                    logger.warning(
                        "Failed to geocode address in batch",
                        address=address,
                        error=str(e),
                    )
                    failed_addresses.append(address)

                    # Create empty result for failed address
                    empty_result = GeocodingResponse(
                        status="ZERO_RESULTS",
                        results=[],
                        error_message=f"Failed to geocode: {e!s}",
                        info_messages=None,
                        next_page_token=None,
                    )
                    results.append(empty_result)

            if failed_addresses:
                logger.info(
                    "Batch geocoding completed with failures",
                    total_addresses=len(addresses),
                    failed_count=len(failed_addresses),
                    failed_addresses=failed_addresses[:5],  # Log first 5 failures
                )

            return results

        except Exception as e:
            logger.error(
                "Batch geocoding failed",
                addresses_count=len(addresses),
                error=str(e),
                exc_info=True,
            )
            raise GeocodingError(
                message="Batch geocoding failed",
            ) from e

    async def batch_reverse_geocode(
        self,
        locations: List[Union[GeoLocation, Dict[str, float]]],
        language: str = "en",
        result_type: Optional[List[str]] = None,
    ) -> List[GeocodingResponse]:
        """Reverse geocode multiple locations in batch.

        Args:
            locations: List of coordinates to reverse geocode
            language: Response language
            result_type: Filter by result type

        Returns:
            List[GeocodingResponse]: List of reverse geocoding results

        Raises:
            ReverseGeocodingError: If batch processing fails
        """
        try:
            if not locations:
                return []

            results = []
            failed_locations = []

            for location in locations:
                try:
                    result = await self.reverse_geocode(
                        location=location,
                        language=language,
                        result_type=result_type,
                    )
                    results.append(result)

                except Exception as e:
                    logger.warning(
                        "Failed to reverse geocode location in batch",
                        location=str(location),
                        error=str(e),
                    )
                    failed_locations.append(str(location))

                    # Create empty result for failed location
                    empty_result = GeocodingResponse(
                        status="ZERO_RESULTS",
                        results=[],
                        error_message=f"Failed to reverse geocode: {e!s}",
                        info_messages=None,
                        next_page_token=None,
                    )
                    results.append(empty_result)

            if failed_locations:
                logger.info(
                    "Batch reverse geocoding completed with failures",
                    total_locations=len(locations),
                    failed_count=len(failed_locations),
                    failed_locations=failed_locations[:5],  # Log first 5 failures
                )

            return results

        except Exception as e:
            logger.error(
                "Batch reverse geocoding failed",
                locations_count=len(locations),
                error=str(e),
                exc_info=True,
            )
            raise ReverseGeocodingError(
                message="Batch reverse geocoding failed",
            ) from e

    async def validate_address(
        self,
        address: str,
        language: str = "en",
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate and format an address.

        Args:
            address: Address to validate
            language: Response language
            region: Region bias (country code)

        Returns:
            Dict[str, Any]: Address validation results

        Raises:
            GeocodingError: If validation fails
        """
        try:
            geocoding_result = await self.geocode_address(
                address=address,
                language=language,
                region=region,
            )

            if not geocoding_result.results:
                return {
                    "original_address": address,
                    "is_valid": False,
                    "confidence": "ZERO_RESULTS",
                    "formatted_address": None,
                    "components": [],
                    "location": None,
                }

            # Use the first (best) result
            result = geocoding_result.results[0]
            geometry = result.get("geometry", {})
            location = geometry.get("location")

            # Determine confidence based on location type
            location_type = geometry.get("location_type", "UNKNOWN")
            confidence_map = {
                "ROOFTOP": "HIGH",
                "RANGE_INTERPOLATED": "MEDIUM",
                "GEOMETRIC_CENTER": "LOW",
                "APPROXIMATE": "LOW",
            }
            confidence = confidence_map.get(location_type, "UNKNOWN")

            return {
                "original_address": address,
                "is_valid": True,
                "confidence": confidence,
                "location_type": location_type,
                "formatted_address": result.get("formatted_address"),
                "components": result.get("address_components", []),
                "location": (
                    {
                        "lat": location.latitude if location else None,
                        "lng": location.longitude if location else None,
                    }
                    if location
                    else None
                ),
                "place_id": result.get("place_id"),
                "types": result.get("types", []),
            }

        except Exception as e:
            if isinstance(e, GeocodingError):
                raise

            logger.error(
                "Address validation failed",
                address=address,
                error=str(e),
                exc_info=True,
            )
            raise GeocodingError(
                address=address,
                message="Address validation failed",
            ) from e

    async def find_nearest_address(
        self,
        location: Union[GeoLocation, Dict[str, float]],
        language: str = "en",
    ) -> Optional[Dict[str, Any]]:
        """Find the nearest address to given coordinates.

        Args:
            location: Geographic coordinates
            language: Response language

        Returns:
            Optional[Dict[str, Any]]: Nearest address information

        Raises:
            ReverseGeocodingError: If search fails
        """
        try:
            # Use reverse geocoding with street address filter
            geocoding_result = await self.reverse_geocode(
                location=location,
                language=language,
                result_type=["street_address", "premise", "subpremise"],
            )

            if not geocoding_result.results:
                return None

            # Return the first (nearest) result
            result = geocoding_result.results[0]
            geometry = result.get("geometry", {})
            result_location = geometry.get("location")

            return {
                "formatted_address": result.get("formatted_address"),
                "components": result.get("address_components", []),
                "location": (
                    {
                        "lat": result_location.latitude if result_location else None,
                        "lng": result_location.longitude if result_location else None,
                    }
                    if result_location
                    else None
                ),
                "place_id": result.get("place_id"),
                "types": result.get("types", []),
                "location_type": geometry.get("location_type"),
            }

        except Exception as e:
            if isinstance(e, ReverseGeocodingError):
                raise

            logger.error(
                "Nearest address search failed",
                location=str(location),
                error=str(e),
                exc_info=True,
            )

            # Extract coordinates for error reporting
            lat = (
                location.latitude
                if isinstance(location, GeoLocation)
                else location.get("lat")
            )
            lng = (
                location.longitude
                if isinstance(location, GeoLocation)
                else location.get("lng")
            )

            raise ReverseGeocodingError(
                latitude=lat,
                longitude=lng,
                message="Nearest address search failed",
            ) from e

    async def get_location_info(
        self,
        query: Union[str, GeoLocation, Dict[str, float]],
        language: str = "en",
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive location information.

        Args:
            query: Address string or coordinates
            language: Response language
            region: Region bias (country code)

        Returns:
            Dict[str, Any]: Comprehensive location information

        Raises:
            GeocodingError: If information retrieval fails
        """
        try:
            if isinstance(query, str):
                # Geocode address
                result = await self.geocode_address(
                    address=query,
                    language=language,
                    region=region,
                )
                if not result.results:
                    return {"query": query, "found": False}

                geocode_result = result.results[0]

            else:
                # Reverse geocode coordinates
                if isinstance(query, dict):
                    query = GeoLocation.model_validate(query)

                result = await self.reverse_geocode(
                    location=query,
                    language=language,
                )
                if not result.results:
                    return {"query": str(query), "found": False}

                geocode_result = result.results[0]

            # Extract location information
            geometry = geocode_result.get("geometry", {})
            location = geometry.get("location")
            components = geocode_result.get("address_components", [])

            # Parse address components into structured data
            address_info = {}
            for component in components:
                types = component.get("types", [])
                long_name = component.get("long_name")
                short_name = component.get("short_name")

                if "country" in types:
                    address_info["country"] = {"long": long_name, "short": short_name}
                elif "administrative_area_level_1" in types:
                    address_info["state"] = {"long": long_name, "short": short_name}
                elif "administrative_area_level_2" in types:
                    address_info["county"] = {"long": long_name, "short": short_name}
                elif "locality" in types:
                    address_info["city"] = {"long": long_name, "short": short_name}
                elif "postal_code" in types:
                    address_info["postal_code"] = long_name
                elif "street_number" in types:
                    address_info["street_number"] = long_name
                elif "route" in types:
                    address_info["street"] = long_name

            return {
                "query": str(query),
                "found": True,
                "formatted_address": geocode_result.get("formatted_address"),
                "location": (
                    {
                        "lat": location.latitude if location else None,
                        "lng": location.longitude if location else None,
                    }
                    if location
                    else None
                ),
                "address_components": address_info,
                "place_id": geocode_result.get("place_id"),
                "types": geocode_result.get("types", []),
                "location_type": geometry.get("location_type"),
                "bounds": geometry.get("bounds"),
                "viewport": geometry.get("viewport"),
            }

        except Exception as e:
            logger.error(
                "Location information retrieval failed",
                query=str(query),
                error=str(e),
                exc_info=True,
            )
            raise GeocodingError(
                message=f"Location information retrieval failed: {query}",
            ) from e


# Global service instance
_geocoding_service: Optional[GeocodingService] = None


def get_geocoding_service() -> GeocodingService:
    """Get or create global Geocoding service instance."""
    global _geocoding_service

    if _geocoding_service is None:
        _geocoding_service = GeocodingService()

    return _geocoding_service
