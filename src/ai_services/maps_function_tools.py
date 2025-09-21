"""Maps Function Tools for AI-Powered Trip Planner Backend.

This module provides function tools that integrate Google Maps services
with the AI function calling framework for trip planning and location services.
"""

from datetime import datetime
from typing import Any, Dict

from src.ai_services.exceptions import FunctionCallError
from src.ai_services.function_tools import (
    FunctionTool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    get_tool_registry,
)
from src.core.logging import get_logger
from src.maps_services.directions_service import get_directions_service
from src.maps_services.exceptions import (
    GeocodingError,
    PlaceNotFoundError,
    RouteNotFoundError,
)
from src.maps_services.geocoding_service import get_geocoding_service
from src.maps_services.places_service import get_places_service
from src.maps_services.schemas import GeoLocation, PlaceType, PriceLevel, TravelMode

logger = get_logger(__name__)


class FindPlacesTool(FunctionTool):
    """Tool to search for places using text query."""

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Find places based on search query."""
        try:
            validated_params = self.validate_parameters(kwargs)

            query = validated_params["query"]
            location = validated_params.get("location")
            radius = validated_params.get("radius")
            place_type = validated_params.get("place_type")
            min_price = validated_params.get("min_price")
            max_price = validated_params.get("max_price")
            open_now = validated_params.get("open_now")
            language = validated_params.get("language", "en")

            # Convert location if provided
            if location:
                if isinstance(location, str):
                    # If it looks like coordinates, convert to GeoLocation
                    if "," in location:
                        try:
                            lat, lng = map(float, location.split(","))
                            location = GeoLocation(lat=lat, lng=lng)
                        except ValueError:
                            # Keep as string if not valid coordinates - will be geocoded by service
                            pass
                elif isinstance(location, dict):
                    location = GeoLocation.model_validate(location)

            # Convert enums if provided
            if place_type and isinstance(place_type, str):
                try:
                    place_type = PlaceType(place_type.lower())
                except ValueError:
                    place_type = None

            if min_price is not None:
                min_price = PriceLevel(min_price)
            if max_price is not None:
                max_price = PriceLevel(max_price)

            places_service = get_places_service()

            # Only pass location if it's a valid type for the service
            search_params = {
                "query": query,
                "place_type": place_type,
                "min_price": min_price,
                "max_price": max_price,
                "open_now": open_now,
                "language": language,
            }

            # Add location and radius only if location is properly converted
            if location and isinstance(location, GeoLocation | dict):
                search_params["location"] = location
                if radius:
                    search_params["radius"] = radius

            response = await places_service.search_text(**search_params)

            # Format results for AI consumption
            places = []
            for result in response.results[:10]:  # Limit to top 10 results
                place_info = {
                    "name": result.name,
                    "address": result.formatted_address,
                    "location": (
                        {
                            "lat": result.geometry.location.latitude,
                            "lng": result.geometry.location.longitude,
                        }
                        if result.geometry and result.geometry.location
                        else None
                    ),
                    "rating": result.rating,
                    "user_ratings_total": result.user_ratings_total,
                    "price_level": (
                        result.price_level.value if result.price_level else None
                    ),
                    "types": result.types,
                    "place_id": result.place_id,
                    "open_now": (
                        result.opening_hours.open_now if result.opening_hours else None
                    ),
                }
                places.append(place_info)

            return {
                "success": True,
                "query": query,
                "places_found": len(places),
                "places": places,
                "search_location": str(location) if location else None,
            }

        except PlaceNotFoundError:
            return {
                "success": True,
                "query": kwargs.get("query", ""),
                "places_found": 0,
                "places": [],
                "message": "No places found for the given query",
            }
        except Exception as e:
            logger.exception("Find places tool failed", error=str(e), params=kwargs)
            raise FunctionCallError(f"Failed to find places: {e}") from e


class GetDirectionsTool(FunctionTool):
    """Tool to get directions between locations."""

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get directions between origin and destination."""
        try:
            validated_params = self.validate_parameters(kwargs)

            origin = validated_params["origin"]
            destination = validated_params["destination"]
            mode = validated_params.get("mode", "driving")
            language = validated_params.get("language", "en")

            # Convert travel mode
            if isinstance(mode, str):
                try:
                    mode = TravelMode(mode.lower())
                except ValueError:
                    mode = TravelMode.DRIVING

            directions_service = get_directions_service()
            response = await directions_service.get_directions(
                origin=origin,
                destination=destination,
                mode=mode,
                language=language,
            )

            if not response.routes:
                return {
                    "success": False,
                    "message": "No route found between the specified locations",
                    "origin": str(origin),
                    "destination": str(destination),
                }

            # Use the first (best) route
            route = response.routes[0]

            # Extract key information
            total_distance = 0
            total_duration = 0
            steps = []

            for leg in route.legs:
                total_distance += leg.distance.value
                total_duration += leg.duration.value

                for step in leg.steps[:5]:  # Limit steps for AI consumption
                    steps.append(
                        {
                            "instruction": step.html_instructions,
                            "distance": step.distance.text,
                            "duration": step.duration.text,
                            "travel_mode": step.travel_mode.value,
                        }
                    )

            return {
                "success": True,
                "origin": str(origin),
                "destination": str(destination),
                "mode": mode.value,
                "total_distance": {
                    "text": f"{total_distance / 1000:.1f} km",
                    "meters": total_distance,
                },
                "total_duration": {
                    "text": f"{total_duration // 60} minutes",
                    "seconds": total_duration,
                },
                "summary": route.summary,
                "steps": steps,
                "warnings": route.warnings,
            }

        except RouteNotFoundError as e:
            return {
                "success": False,
                "message": "No route found between the specified locations",
                "origin": str(kwargs.get("origin", "")),
                "destination": str(kwargs.get("destination", "")),
                "error": str(e),
            }
        except Exception as e:
            logger.exception("Get directions tool failed", error=str(e), params=kwargs)
            raise FunctionCallError(f"Failed to get directions: {e}") from e


class GeocodeTool(FunctionTool):
    """Tool to convert addresses to coordinates or vice versa."""

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Geocode address or reverse geocode coordinates."""
        try:
            validated_params = self.validate_parameters(kwargs)

            address = validated_params.get("address")
            location = validated_params.get("location")
            language = validated_params.get("language", "en")

            geocoding_service = get_geocoding_service()

            if address:
                # Forward geocoding (address to coordinates)
                response = await geocoding_service.geocode_address(
                    address=address,
                    language=language,
                )

                if not response.results:
                    return {
                        "success": False,
                        "message": "Address not found",
                        "address": address,
                    }

                result = response.results[0]
                geometry = result.get("geometry", {})
                result_location = geometry.get("location")

                return {
                    "success": True,
                    "type": "geocoding",
                    "address": address,
                    "formatted_address": result.get("formatted_address"),
                    "location": (
                        {
                            "lat": result_location.latitude,
                            "lng": result_location.longitude,
                        }
                        if result_location
                        else None
                    ),
                    "place_id": result.get("place_id"),
                    "location_type": geometry.get("location_type"),
                    "types": result.get("types", []),
                }

            elif location:
                # Reverse geocoding (coordinates to address)
                if isinstance(location, str):
                    try:
                        lat, lng = map(float, location.split(","))
                        location = GeoLocation(lat=lat, lng=lng)
                    except ValueError:
                        raise FunctionCallError(
                            "Invalid location format. Use 'lat,lng' format."
                        ) from None
                elif isinstance(location, dict):
                    location = GeoLocation.model_validate(location)

                response = await geocoding_service.reverse_geocode(
                    location=location,
                    language=language,
                )

                if not response.results:
                    return {
                        "success": False,
                        "message": "No address found for the given coordinates",
                        "location": str(location),
                    }

                result = response.results[0]
                return {
                    "success": True,
                    "type": "reverse_geocoding",
                    "location": {
                        "lat": location.latitude,
                        "lng": location.longitude,
                    },
                    "formatted_address": result.get("formatted_address"),
                    "place_id": result.get("place_id"),
                    "types": result.get("types", []),
                }

            else:
                raise FunctionCallError(
                    "Either 'address' or 'location' parameter is required"
                )

        except GeocodingError as e:
            return {
                "success": False,
                "message": str(e),
                "address": kwargs.get("address"),
                "location": kwargs.get("location"),
            }
        except Exception as e:
            logger.exception("Geocode tool failed", error=str(e), params=kwargs)
            raise FunctionCallError(f"Failed to geocode: {e}") from e


class FindNearbyPlacesTool(FunctionTool):
    """Tool to find places near a specific location."""

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Find places near a given location."""
        try:
            validated_params = self.validate_parameters(kwargs)

            location = validated_params["location"]
            radius = validated_params.get("radius", 1000)
            place_type = validated_params.get("place_type")
            keyword = validated_params.get("keyword")
            min_rating = validated_params.get("min_rating", 0.0)
            language = validated_params.get("language", "en")

            # Convert location
            if isinstance(location, str):
                try:
                    lat, lng = map(float, location.split(","))
                    location = GeoLocation(lat=lat, lng=lng)
                except ValueError:
                    raise FunctionCallError(
                        "Invalid location format. Use 'lat,lng' format."
                    ) from None
            elif isinstance(location, dict):
                location = GeoLocation.model_validate(location)

            # Convert place type
            if place_type and isinstance(place_type, str):
                try:
                    place_type = PlaceType(place_type.lower())
                except ValueError:
                    place_type = None

            places_service = get_places_service()
            response = await places_service.search_nearby(
                location=location,
                radius=radius,
                place_type=place_type,
                keyword=keyword,
                language=language,
            )

            # Filter by minimum rating and format results
            places = []
            for result in response.results:
                if result.rating and result.rating >= min_rating:
                    place_info = {
                        "name": result.name,
                        "address": result.formatted_address,
                        "location": (
                            {
                                "lat": result.geometry.location.latitude,
                                "lng": result.geometry.location.longitude,
                            }
                            if result.geometry and result.geometry.location
                            else None
                        ),
                        "rating": result.rating,
                        "user_ratings_total": result.user_ratings_total,
                        "price_level": (
                            result.price_level.value if result.price_level else None
                        ),
                        "types": result.types,
                        "place_id": result.place_id,
                        "open_now": (
                            result.opening_hours.open_now
                            if result.opening_hours
                            else None
                        ),
                    }
                    places.append(place_info)

            # Sort by rating
            places.sort(key=lambda x: x["rating"] or 0, reverse=True)

            return {
                "success": True,
                "search_location": {
                    "lat": location.latitude,
                    "lng": location.longitude,
                },
                "radius_meters": radius,
                "place_type": place_type.value if place_type else None,
                "keyword": keyword,
                "min_rating": min_rating,
                "places_found": len(places),
                "places": places[:15],  # Limit to top 15 results
            }

        except PlaceNotFoundError:
            return {
                "success": True,
                "search_location": str(kwargs.get("location", "")),
                "places_found": 0,
                "places": [],
                "message": "No places found near the specified location",
            }
        except Exception as e:
            logger.exception(
                "Find nearby places tool failed", error=str(e), params=kwargs
            )
            raise FunctionCallError(f"Failed to find nearby places: {e}") from e


class GetTravelTimeTool(FunctionTool):
    """Tool to get travel time between two locations."""

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get travel time and distance between locations."""
        try:
            validated_params = self.validate_parameters(kwargs)

            origin = validated_params["origin"]
            destination = validated_params["destination"]
            mode = validated_params.get("mode", "driving")
            departure_time = validated_params.get("departure_time")

            # Convert travel mode
            if isinstance(mode, str):
                try:
                    mode = TravelMode(mode.lower())
                except ValueError:
                    mode = TravelMode.DRIVING

            # Convert departure time if provided
            if departure_time and isinstance(departure_time, str):
                try:
                    departure_time = datetime.fromisoformat(departure_time)
                except ValueError:
                    departure_time = None

            directions_service = get_directions_service()
            result = await directions_service.get_travel_time(
                origin=origin,
                destination=destination,
                mode=mode,
                departure_time=departure_time,
            )

            return {
                "success": True,
                "origin": str(origin),
                "destination": str(destination),
                "mode": mode.value,
                "distance": result["distance"],
                "duration": result["duration"],
                "duration_in_traffic": result.get("duration_in_traffic"),
                "departure_time": (
                    departure_time.isoformat() if departure_time else None
                ),
            }

        except RouteNotFoundError:
            return {
                "success": False,
                "message": "No travel route found between the specified locations",
                "origin": str(kwargs.get("origin", "")),
                "destination": str(kwargs.get("destination", "")),
            }
        except Exception as e:
            logger.exception("Get travel time tool failed", error=str(e), params=kwargs)
            raise FunctionCallError(f"Failed to get travel time: {e}") from e


class ValidateLocationTool(FunctionTool):
    """Tool to validate and format addresses or coordinates."""

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Validate location input and return standardized format."""
        try:
            validated_params = self.validate_parameters(kwargs)

            location = validated_params["location"]
            language = validated_params.get("language", "en")

            geocoding_service = get_geocoding_service()

            # Check if it's coordinates or address
            if isinstance(location, str) and "," in location:
                try:
                    # Try to parse as coordinates
                    lat, lng = map(float, location.split(","))
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        geo_location = GeoLocation(lat=lat, lng=lng)
                        result = await geocoding_service.find_nearest_address(
                            location=geo_location,
                            language=language,
                        )

                        return {
                            "success": True,
                            "input_type": "coordinates",
                            "is_valid": True,
                            "location": {
                                "lat": lat,
                                "lng": lng,
                            },
                            "nearest_address": (
                                result["formatted_address"] if result else None
                            ),
                            "place_id": result["place_id"] if result else None,
                        }
                except ValueError:
                    pass

            # Treat as address
            validation_result = await geocoding_service.validate_address(
                address=location,
                language=language,
            )

            return {
                "success": True,
                "input_type": "address",
                "original_address": location,
                "is_valid": validation_result["is_valid"],
                "confidence": validation_result["confidence"],
                "formatted_address": validation_result["formatted_address"],
                "location": validation_result["location"],
                "place_id": validation_result["place_id"],
                "address_components": validation_result.get("components", []),
            }

        except Exception as e:
            logger.exception(
                "Validate location tool failed", error=str(e), params=kwargs
            )
            raise FunctionCallError(f"Failed to validate location: {e}") from e


class GetPlaceDetailsTool(FunctionTool):
    """Tool to get detailed information about a specific place."""

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get detailed information about a place."""
        try:
            validated_params = self.validate_parameters(kwargs)

            place_id = validated_params["place_id"]
            language = validated_params.get("language", "en")

            places_service = get_places_service()
            place_details = await places_service.get_place_details(
                place_id=place_id,
                language=language,
            )

            return {
                "success": True,
                "place_id": place_id,
                "name": place_details.name,
                "formatted_address": place_details.formatted_address,
                "location": (
                    {
                        "lat": place_details.geometry.location.latitude,
                        "lng": place_details.geometry.location.longitude,
                    }
                    if place_details.geometry and place_details.geometry.location
                    else None
                ),
                "rating": place_details.rating,
                "user_ratings_total": place_details.user_ratings_total,
                "price_level": (
                    place_details.price_level.value
                    if place_details.price_level
                    else None
                ),
                "phone_number": place_details.formatted_phone_number,
                "website": place_details.website,
                "opening_hours": (
                    {
                        "open_now": place_details.opening_hours.open_now,
                        "weekday_text": place_details.opening_hours.weekday_text,
                    }
                    if place_details.opening_hours
                    else None
                ),
                "types": place_details.types,
                "business_status": (
                    place_details.business_status.value
                    if place_details.business_status
                    else None
                ),
                "reviews_count": (
                    len(place_details.reviews) if place_details.reviews else 0
                ),
                "photos_count": (
                    len(place_details.photos) if place_details.photos else 0
                ),
            }

        except Exception as e:
            logger.exception(
                "Get place details tool failed", error=str(e), params=kwargs
            )
            raise FunctionCallError(f"Failed to get place details: {e}") from e


def create_find_places_tool() -> FunctionTool:
    """Create the find places function tool."""
    metadata = ToolMetadata(
        name="find_places",
        description="Search for places using text query with optional filters",
        category=ToolCategory.MAPS,
        parameters=[
            ToolParameter(
                name="query",
                type_hint=str,
                description="Search query for places (e.g., 'restaurants in Mumbai', 'tourist attractions')",
                required=True,
                example="restaurants in Mumbai",
            ),
            ToolParameter(
                name="location",
                type_hint=str,
                description="Location to bias search around (address or 'lat,lng')",
                required=False,
                example="Mumbai, India",
            ),
            ToolParameter(
                name="radius",
                type_hint=int,
                description="Search radius in meters (max 50000)",
                required=False,
                default=5000,
                example=2000,
            ),
            ToolParameter(
                name="place_type",
                type_hint=str,
                description="Type of place to search for",
                required=False,
                example="restaurant",
            ),
            ToolParameter(
                name="min_price",
                type_hint=int,
                description="Minimum price level (0-4)",
                required=False,
                example=1,
            ),
            ToolParameter(
                name="max_price",
                type_hint=int,
                description="Maximum price level (0-4)",
                required=False,
                example=3,
            ),
            ToolParameter(
                name="open_now",
                type_hint=bool,
                description="Only return places that are currently open",
                required=False,
                example=True,
            ),
            ToolParameter(
                name="language",
                type_hint=str,
                description="Response language code",
                required=False,
                default="en",
                example="en",
            ),
        ],
        examples=[
            {"query": "restaurants in Mumbai", "radius": 2000, "open_now": True},
            {"query": "tourist attractions", "location": "Mumbai, India"},
        ],
        tags=["places", "search", "location", "maps"],
    )
    return FindPlacesTool(metadata)


def create_get_directions_tool() -> FunctionTool:
    """Create the get directions function tool."""
    metadata = ToolMetadata(
        name="get_directions",
        description="Get directions and route information between two locations",
        category=ToolCategory.MAPS,
        parameters=[
            ToolParameter(
                name="origin",
                type_hint=str,
                description="Starting location (address or 'lat,lng')",
                required=True,
                example="Mumbai Airport",
            ),
            ToolParameter(
                name="destination",
                type_hint=str,
                description="Destination location (address or 'lat,lng')",
                required=True,
                example="Gateway of India, Mumbai",
            ),
            ToolParameter(
                name="mode",
                type_hint=str,
                description="Transportation mode (driving, walking, bicycling, transit)",
                required=False,
                default="driving",
                example="driving",
            ),
            ToolParameter(
                name="language",
                type_hint=str,
                description="Response language code",
                required=False,
                default="en",
                example="en",
            ),
        ],
        examples=[
            {
                "origin": "Mumbai Airport",
                "destination": "Gateway of India",
                "mode": "driving",
            },
            {
                "origin": "19.0760,72.8777",
                "destination": "18.9220,72.8347",
                "mode": "transit",
            },
        ],
        tags=["directions", "route", "navigation", "maps"],
    )
    return GetDirectionsTool(metadata)


def create_geocode_tool() -> FunctionTool:
    """Create the geocoding function tool."""
    metadata = ToolMetadata(
        name="geocode_location",
        description="Convert addresses to coordinates or coordinates to addresses",
        category=ToolCategory.MAPS,
        parameters=[
            ToolParameter(
                name="address",
                type_hint=str,
                description="Address to convert to coordinates",
                required=False,
                example="Gateway of India, Mumbai",
            ),
            ToolParameter(
                name="location",
                type_hint=str,
                description="Coordinates to convert to address ('lat,lng')",
                required=False,
                example="19.0760,72.8777",
            ),
            ToolParameter(
                name="language",
                type_hint=str,
                description="Response language code",
                required=False,
                default="en",
                example="en",
            ),
        ],
        examples=[
            {"address": "Gateway of India, Mumbai"},
            {"location": "19.0760,72.8777"},
        ],
        tags=["geocoding", "coordinates", "address", "maps"],
    )
    return GeocodeTool(metadata)


def create_find_nearby_places_tool() -> FunctionTool:
    """Create the find nearby places function tool."""
    metadata = ToolMetadata(
        name="find_nearby_places",
        description="Find places near a specific location with filters",
        category=ToolCategory.MAPS,
        parameters=[
            ToolParameter(
                name="location",
                type_hint=str,
                description="Center location for search ('lat,lng' or address)",
                required=True,
                example="19.0760,72.8777",
            ),
            ToolParameter(
                name="radius",
                type_hint=int,
                description="Search radius in meters (max 50000)",
                required=False,
                default=1000,
                example=2000,
            ),
            ToolParameter(
                name="place_type",
                type_hint=str,
                description="Type of place to search for",
                required=False,
                example="restaurant",
            ),
            ToolParameter(
                name="keyword",
                type_hint=str,
                description="Keyword to filter places",
                required=False,
                example="vegetarian",
            ),
            ToolParameter(
                name="min_rating",
                type_hint=float,
                description="Minimum rating filter (0-5)",
                required=False,
                default=0.0,
                example=4.0,
            ),
            ToolParameter(
                name="language",
                type_hint=str,
                description="Response language code",
                required=False,
                default="en",
                example="en",
            ),
        ],
        examples=[
            {"location": "19.0760,72.8777", "place_type": "restaurant", "radius": 1000},
            {"location": "Mumbai", "keyword": "vegetarian", "min_rating": 4.0},
        ],
        tags=["nearby", "places", "search", "location", "maps"],
    )
    return FindNearbyPlacesTool(metadata)


def create_get_travel_time_tool() -> FunctionTool:
    """Create the get travel time function tool."""
    metadata = ToolMetadata(
        name="get_travel_time",
        description="Get travel time and distance between two locations",
        category=ToolCategory.TRAVEL,
        parameters=[
            ToolParameter(
                name="origin",
                type_hint=str,
                description="Starting location (address or 'lat,lng')",
                required=True,
                example="Mumbai Airport",
            ),
            ToolParameter(
                name="destination",
                type_hint=str,
                description="Destination location (address or 'lat,lng')",
                required=True,
                example="Gateway of India, Mumbai",
            ),
            ToolParameter(
                name="mode",
                type_hint=str,
                description="Transportation mode (driving, walking, bicycling, transit)",
                required=False,
                default="driving",
                example="driving",
            ),
            ToolParameter(
                name="departure_time",
                type_hint=str,
                description="Departure time in ISO format for traffic calculations",
                required=False,
                example="2024-01-15T10:00:00",
            ),
        ],
        examples=[
            {
                "origin": "Mumbai Airport",
                "destination": "Gateway of India",
                "mode": "driving",
            },
            {
                "origin": "19.0760,72.8777",
                "destination": "18.9220,72.8347",
                "departure_time": "2024-01-15T10:00:00",
            },
        ],
        tags=["travel", "time", "distance", "route", "maps"],
    )
    return GetTravelTimeTool(metadata)


def create_validate_location_tool() -> FunctionTool:
    """Create the validate location function tool."""
    metadata = ToolMetadata(
        name="validate_location",
        description="Validate and standardize location input (address or coordinates)",
        category=ToolCategory.VALIDATION,
        parameters=[
            ToolParameter(
                name="location",
                type_hint=str,
                description="Location to validate (address or 'lat,lng')",
                required=True,
                example="Gateway of India, Mumbai",
            ),
            ToolParameter(
                name="language",
                type_hint=str,
                description="Response language code",
                required=False,
                default="en",
                example="en",
            ),
        ],
        examples=[
            {"location": "Gateway of India, Mumbai"},
            {"location": "19.0760,72.8777"},
        ],
        tags=["validation", "location", "address", "coordinates", "maps"],
    )
    return ValidateLocationTool(metadata)


def create_get_place_details_tool() -> FunctionTool:
    """Create the get place details function tool."""
    metadata = ToolMetadata(
        name="get_place_details",
        description="Get detailed information about a specific place using its place ID",
        category=ToolCategory.PLACES,
        parameters=[
            ToolParameter(
                name="place_id",
                type_hint=str,
                description="Google Places ID of the place",
                required=True,
                example="ChIJbU60yXAWrjsR4jNMNtgzDjI",
            ),
            ToolParameter(
                name="language",
                type_hint=str,
                description="Response language code",
                required=False,
                default="en",
                example="en",
            ),
        ],
        examples=[
            {"place_id": "ChIJbU60yXAWrjsR4jNMNtgzDjI"},
        ],
        tags=["place", "details", "information", "maps"],
    )
    return GetPlaceDetailsTool(metadata)


def register_maps_tools() -> None:
    """Register all Maps function tools with the tool registry."""
    try:
        registry = get_tool_registry()

        tools = [
            create_find_places_tool(),
            create_get_directions_tool(),
            create_geocode_tool(),
            create_find_nearby_places_tool(),
            create_get_travel_time_tool(),
            create_validate_location_tool(),
            create_get_place_details_tool(),
        ]

        for tool in tools:
            try:
                registry.register_tool(tool)
                logger.debug("Registered Maps tool", tool_name=tool.metadata.name)
            except Exception as e:
                logger.exception(
                    "Failed to register Maps tool",
                    tool_name=tool.metadata.name,
                    error=str(e),
                )

        logger.info(
            "Maps function tools registered successfully",
            tool_count=len(tools),
            categories=[ToolCategory.MAPS, ToolCategory.TRAVEL, ToolCategory.PLACES],
        )

    except Exception as e:
        logger.exception(
            "Failed to register Maps function tools",
            error=str(e),
        )


# Auto-register tools when module is imported
try:
    register_maps_tools()
except Exception as e:
    logger.warning(
        "Failed to auto-register Maps tools on import",
        error=str(e),
    )
