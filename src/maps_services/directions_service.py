"""Directions Service for Google Maps API Integration.

This module provides high-level directions and route planning functionality
with proper error handling, validation, and response parsing.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from src.core.logging import get_logger
from src.maps_services.exceptions import (
    DirectionsError,
    DistanceMatrixError,
    InvalidWaypointError,
    RouteNotFoundError,
)
from src.maps_services.maps_client import get_maps_client
from src.maps_services.schemas import (
    DirectionsResponse,
    Distance,
    DistanceMatrixResponse,
    Duration,
    GeoLocation,
    RouteInfo,
    RouteLeg,
    RouteStep,
    TravelMode,
)

logger = get_logger(__name__)


class DirectionsService:
    """Service for Google Maps Directions API operations."""

    def __init__(self) -> None:
        """Initialize Directions service."""
        self._client = None

    async def _get_client(self):
        """Get Maps API client instance."""
        if not self._client:
            self._client = await get_maps_client()
        return self._client

    def _parse_distance(self, distance_data: Dict[str, Any]) -> Distance:
        """Parse distance data into Distance model."""
        try:
            return Distance.model_validate(distance_data)
        except ValidationError as e:
            logger.warning(
                "Failed to parse distance data",
                distance_data=distance_data,
                validation_errors=e.errors(),
            )
            # Return a basic distance object with available data
            return Distance(
                text=distance_data.get("text", "Unknown"),
                value=distance_data.get("value", 0),
            )

    def _parse_duration(self, duration_data: Dict[str, Any]) -> Duration:
        """Parse duration data into Duration model."""
        try:
            return Duration.model_validate(duration_data)
        except ValidationError as e:
            logger.warning(
                "Failed to parse duration data",
                duration_data=duration_data,
                validation_errors=e.errors(),
            )
            # Return a basic duration object with available data
            return Duration(
                text=duration_data.get("text", "Unknown"),
                value=duration_data.get("value", 0),
            )

    def _parse_geo_location(self, location_data: Dict[str, Any]) -> GeoLocation:
        """Parse location data into GeoLocation model."""
        try:
            return GeoLocation.model_validate(location_data)
        except ValidationError as e:
            logger.warning(
                "Failed to parse location data",
                location_data=location_data,
                validation_errors=e.errors(),
            )
            # Try to extract lat/lng with different key formats
            lat = location_data.get("lat") or location_data.get("latitude")
            lng = location_data.get("lng") or location_data.get("longitude")
            if lat is not None and lng is not None:
                return GeoLocation(lat=lat, lng=lng)
            else:
                raise DirectionsError("Invalid location data in route response") from e

    def _parse_route_step(self, step_data: Dict[str, Any]) -> RouteStep:
        """Parse route step data into RouteStep model."""
        try:
            # Parse nested objects
            if "distance" in step_data:
                step_data["distance"] = self._parse_distance(step_data["distance"])
            if "duration" in step_data:
                step_data["duration"] = self._parse_duration(step_data["duration"])
            if "start_location" in step_data:
                step_data["start_location"] = self._parse_geo_location(
                    step_data["start_location"]
                )
            if "end_location" in step_data:
                step_data["end_location"] = self._parse_geo_location(
                    step_data["end_location"]
                )

            # Handle nested steps recursively
            if step_data.get("steps"):
                nested_steps = []
                for nested_step in step_data["steps"]:
                    try:
                        nested_steps.append(self._parse_route_step(nested_step))
                    except Exception as e:
                        logger.warning(
                            "Failed to parse nested route step",
                            error=str(e),
                        )
                step_data["steps"] = nested_steps if nested_steps else None

            return RouteStep.model_validate(step_data)

        except ValidationError as e:
            logger.warning(
                "Failed to parse route step",
                validation_errors=e.errors(),
            )
            raise DirectionsError("Failed to parse route step data") from e

    def _parse_route_leg(self, leg_data: Dict[str, Any]) -> RouteLeg:
        """Parse route leg data into RouteLeg model."""
        try:
            # Parse nested objects
            if "distance" in leg_data:
                leg_data["distance"] = self._parse_distance(leg_data["distance"])
            if "duration" in leg_data:
                leg_data["duration"] = self._parse_duration(leg_data["duration"])
            if leg_data.get("duration_in_traffic"):
                leg_data["duration_in_traffic"] = self._parse_duration(
                    leg_data["duration_in_traffic"]
                )
            if "start_location" in leg_data:
                leg_data["start_location"] = self._parse_geo_location(
                    leg_data["start_location"]
                )
            if "end_location" in leg_data:
                leg_data["end_location"] = self._parse_geo_location(
                    leg_data["end_location"]
                )

            # Parse steps
            if "steps" in leg_data:
                steps = []
                for step_data in leg_data["steps"]:
                    try:
                        steps.append(self._parse_route_step(step_data))
                    except Exception as e:
                        logger.warning(
                            "Failed to parse route step in leg",
                            error=str(e),
                        )
                leg_data["steps"] = steps

            return RouteLeg.model_validate(leg_data)

        except ValidationError as e:
            logger.warning(
                "Failed to parse route leg",
                validation_errors=e.errors(),
            )
            raise DirectionsError("Failed to parse route leg data") from e

    def _parse_route_info(self, route_data: Dict[str, Any]) -> RouteInfo:
        """Parse route data into RouteInfo model."""
        try:
            # Parse bounds
            if "bounds" in route_data:
                bounds_data = route_data["bounds"]
                if "northeast" in bounds_data and "southwest" in bounds_data:
                    route_data["bounds"] = {
                        "northeast": self._parse_geo_location(bounds_data["northeast"]),
                        "southwest": self._parse_geo_location(bounds_data["southwest"]),
                    }

            # Parse legs
            if "legs" in route_data:
                legs = []
                for leg_data in route_data["legs"]:
                    try:
                        legs.append(self._parse_route_leg(leg_data))
                    except Exception as e:
                        logger.warning(
                            "Failed to parse route leg",
                            error=str(e),
                        )
                route_data["legs"] = legs

            return RouteInfo.model_validate(route_data)

        except ValidationError as e:
            logger.warning(
                "Failed to parse route info",
                validation_errors=e.errors(),
            )
            raise DirectionsError("Failed to parse route data") from e

    def _parse_directions_response(
        self, response_data: Dict[str, Any]
    ) -> DirectionsResponse:
        """Parse directions response data into DirectionsResponse model."""
        try:
            # Parse routes
            if "routes" in response_data:
                routes = []
                for route_data in response_data["routes"]:
                    try:
                        routes.append(self._parse_route_info(route_data))
                    except Exception as e:
                        logger.warning(
                            "Failed to parse route in response",
                            error=str(e),
                        )
                response_data["routes"] = routes

            return DirectionsResponse.model_validate(response_data)

        except ValidationError as e:
            logger.exception(
                "Failed to parse directions response",
            )
            raise DirectionsError("Failed to parse directions response") from e

    async def get_directions(
        self,
        origin: Union[str, GeoLocation, Dict[str, float]],
        destination: Union[str, GeoLocation, Dict[str, float]],
        waypoints: Optional[List[Union[str, GeoLocation, Dict[str, float]]]] = None,
        mode: TravelMode = TravelMode.DRIVING,
        alternatives: bool = False,
        avoid: Optional[List[str]] = None,
        language: str = "en",
        region: Optional[str] = None,
        units: str = "metric",
        departure_time: Optional[Union[datetime, str]] = None,
        arrival_time: Optional[Union[datetime, str]] = None,
        traffic_model: str = "best_guess",
        optimize_waypoints: bool = False,
    ) -> DirectionsResponse:
        """Get directions between locations.

        Args:
            origin: Starting location
            destination: Ending location
            waypoints: Intermediate waypoints
            mode: Transportation mode
            alternatives: Include alternative routes
            avoid: Features to avoid (tolls, highways, ferries, indoor)
            language: Response language
            region: Region bias (country code)
            units: Unit system (metric/imperial)
            departure_time: Departure time for traffic calculations
            arrival_time: Arrival time for transit planning
            traffic_model: Traffic model (best_guess, pessimistic, optimistic)
            optimize_waypoints: Optimize waypoint order

        Returns:
            DirectionsResponse: Directions with routes and turn-by-turn navigation

        Raises:
            DirectionsError: If directions calculation fails
            RouteNotFoundError: If no route can be found
            InvalidWaypointError: If waypoints are invalid
        """
        try:
            # Validate inputs
            if isinstance(origin, dict):
                origin = GeoLocation.model_validate(origin)
            if isinstance(destination, dict):
                destination = GeoLocation.model_validate(destination)

            if waypoints:
                validated_waypoints = []
                for waypoint in waypoints:
                    if isinstance(waypoint, dict):
                        validated_waypoints.append(GeoLocation.model_validate(waypoint))
                    else:
                        validated_waypoints.append(waypoint)
                waypoints = validated_waypoints

            # Build parameters
            client_params: Dict[str, Any] = {
                "mode": mode.value,
                "language": language,
                "units": units,
            }

            if waypoints:
                client_params["waypoints"] = waypoints
            if alternatives:
                client_params["alternatives"] = alternatives
            if avoid:
                client_params["avoid"] = "|".join(avoid)
            if region:
                client_params["region"] = region
            if departure_time:
                if isinstance(departure_time, datetime):
                    client_params["departure_time"] = int(departure_time.timestamp())
                else:
                    client_params["departure_time"] = departure_time
            if arrival_time:
                if isinstance(arrival_time, datetime):
                    client_params["arrival_time"] = int(arrival_time.timestamp())
                else:
                    client_params["arrival_time"] = arrival_time
            if traffic_model != "best_guess":
                client_params["traffic_model"] = traffic_model
            if optimize_waypoints:
                client_params["optimize"] = optimize_waypoints

            # Make API request
            client = await self._get_client()
            response = await client.directions(
                origin=origin, destination=destination, **client_params
            )

            # Check if any routes were found
            if response.get("status") == "ZERO_RESULTS":
                raise RouteNotFoundError(
                    origin=str(origin),
                    destination=str(destination),
                    travel_mode=mode.value,
                )

            # Parse and return response
            return self._parse_directions_response(response)

        except Exception as e:
            if isinstance(
                e, DirectionsError | RouteNotFoundError | InvalidWaypointError
            ):
                raise

            logger.error(
                "Directions calculation failed",
                origin=str(origin),
                destination=str(destination),
                mode=mode.value if isinstance(mode, TravelMode) else mode,
                error=str(e),
                exc_info=True,
            )
            raise DirectionsError(
                origin=str(origin),
                destination=str(destination),
                message="Directions calculation failed",
            ) from e

    async def get_distance_matrix(
        self,
        origins: List[Union[str, GeoLocation, Dict[str, float]]],
        destinations: List[Union[str, GeoLocation, Dict[str, float]]],
        mode: TravelMode = TravelMode.DRIVING,
        language: str = "en",
        region: Optional[str] = None,
        avoid: Optional[List[str]] = None,
        units: str = "metric",
        departure_time: Optional[Union[datetime, str]] = None,
        arrival_time: Optional[Union[datetime, str]] = None,
        traffic_model: str = "best_guess",
    ) -> DistanceMatrixResponse:
        """Get distance matrix between multiple origins and destinations.

        Args:
            origins: List of origin locations
            destinations: List of destination locations
            mode: Transportation mode
            language: Response language
            region: Region bias (country code)
            avoid: Features to avoid
            units: Unit system (metric/imperial)
            departure_time: Departure time for traffic calculations
            arrival_time: Arrival time for transit planning
            traffic_model: Traffic model

        Returns:
            DistanceMatrixResponse: Distance and time matrix

        Raises:
            DistanceMatrixError: If matrix calculation fails
        """
        try:
            if not origins or not destinations:
                raise DistanceMatrixError(
                    origins=[str(o) for o in origins] if origins else [],
                    destinations=[str(d) for d in destinations] if destinations else [],
                    message="Origins and destinations cannot be empty",
                )

            # Validate and convert locations
            validated_origins = []
            for origin in origins:
                if isinstance(origin, dict):
                    validated_origins.append(GeoLocation.model_validate(origin))
                else:
                    validated_origins.append(origin)

            validated_destinations = []
            for destination in destinations:
                if isinstance(destination, dict):
                    validated_destinations.append(
                        GeoLocation.model_validate(destination)
                    )
                else:
                    validated_destinations.append(destination)

            # Build parameters
            client_params: Dict[str, Any] = {
                "mode": mode.value,
                "language": language,
                "units": units,
            }

            if region:
                client_params["region"] = region
            if avoid:
                client_params["avoid"] = "|".join(avoid)
            if departure_time:
                if isinstance(departure_time, datetime):
                    client_params["departure_time"] = int(departure_time.timestamp())
                else:
                    client_params["departure_time"] = departure_time
            if arrival_time:
                if isinstance(arrival_time, datetime):
                    client_params["arrival_time"] = int(arrival_time.timestamp())
                else:
                    client_params["arrival_time"] = arrival_time
            if traffic_model != "best_guess":
                client_params["traffic_model"] = traffic_model

            # Make API request
            client = await self._get_client()
            response = await client.distance_matrix(
                origins=validated_origins,
                destinations=validated_destinations,
                **client_params,
            )

            # Parse and return response
            return DistanceMatrixResponse.model_validate(response)

        except Exception as e:
            if isinstance(e, DistanceMatrixError):
                raise

            logger.error(
                "Distance matrix calculation failed",
                origins_count=len(origins) if origins else 0,
                destinations_count=len(destinations) if destinations else 0,
                mode=mode.value if isinstance(mode, TravelMode) else mode,
                error=str(e),
                exc_info=True,
            )
            raise DistanceMatrixError(
                origins=[str(o) for o in origins] if origins else [],
                destinations=[str(d) for d in destinations] if destinations else [],
                message="Distance matrix calculation failed",
            ) from e

    async def get_travel_time(
        self,
        origin: Union[str, GeoLocation, Dict[str, float]],
        destination: Union[str, GeoLocation, Dict[str, float]],
        mode: TravelMode = TravelMode.DRIVING,
        departure_time: Optional[datetime] = None,
        traffic_model: str = "best_guess",
    ) -> Dict[str, Any]:
        """Get travel time and distance between two locations.

        Args:
            origin: Starting location
            destination: Ending location
            mode: Transportation mode
            departure_time: Departure time for traffic calculations
            traffic_model: Traffic model

        Returns:
            Dict[str, Any]: Travel time and distance information

        Raises:
            DirectionsError: If calculation fails
        """
        try:
            # Use distance matrix for simple travel time calculation
            matrix_response = await self.get_distance_matrix(
                origins=[origin],
                destinations=[destination],
                mode=mode,
                departure_time=departure_time,
                traffic_model=traffic_model,
            )

            if not matrix_response.rows or not matrix_response.rows[0].get("elements"):
                raise DirectionsError(
                    origin=str(origin),
                    destination=str(destination),
                    message="No travel time data available",
                )

            element = matrix_response.rows[0]["elements"][0]

            if element.get("status") != "OK":
                raise DirectionsError(
                    origin=str(origin),
                    destination=str(destination),
                    message=f"Travel time calculation failed: {element.get('status')}",
                )

            return {
                "origin": str(origin),
                "destination": str(destination),
                "mode": mode.value,
                "distance": element.get("distance", {}),
                "duration": element.get("duration", {}),
                "duration_in_traffic": element.get("duration_in_traffic"),
                "status": element.get("status"),
            }

        except Exception as e:
            if isinstance(e, DirectionsError):
                raise

            logger.error(
                "Travel time calculation failed",
                origin=str(origin),
                destination=str(destination),
                mode=mode.value,
                error=str(e),
                exc_info=True,
            )
            raise DirectionsError(
                origin=str(origin),
                destination=str(destination),
                message="Travel time calculation failed",
            ) from e

    async def optimize_route(
        self,
        start: Union[str, GeoLocation, Dict[str, float]],
        waypoints: List[Union[str, GeoLocation, Dict[str, float]]],
        end: Optional[Union[str, GeoLocation, Dict[str, float]]] = None,
        mode: TravelMode = TravelMode.DRIVING,
        language: str = "en",
    ) -> DirectionsResponse:
        """Optimize route order for multiple waypoints.

        Args:
            start: Starting location
            waypoints: List of waypoints to visit
            end: Ending location (defaults to start if not provided)
            mode: Transportation mode
            language: Response language

        Returns:
            DirectionsResponse: Optimized route with reordered waypoints

        Raises:
            DirectionsError: If optimization fails
            InvalidWaypointError: If waypoints are invalid
        """
        try:
            if not waypoints:
                raise InvalidWaypointError(
                    invalid_waypoints=["empty waypoints list"],
                    message="Waypoints list cannot be empty for route optimization",
                )

            if len(waypoints) > 25:  # Google Maps API limit
                raise InvalidWaypointError(
                    invalid_waypoints=[f"waypoint_count_{len(waypoints)}"],
                    message="Maximum 25 waypoints allowed for route optimization",
                )

            destination = end if end is not None else start

            # Get optimized directions
            response = await self.get_directions(
                origin=start,
                destination=destination,
                waypoints=waypoints,
                mode=mode,
                language=language,
                optimize_waypoints=True,
            )

            return response

        except Exception as e:
            if isinstance(e, DirectionsError | InvalidWaypointError):
                raise

            logger.error(
                "Route optimization failed",
                start=str(start),
                waypoints_count=len(waypoints) if waypoints else 0,
                end=str(end) if end else None,
                error=str(e),
                exc_info=True,
            )
            raise DirectionsError(
                origin=str(start),
                destination=str(end) if end else str(start),
                message="Route optimization failed",
            ) from e

    async def get_route_alternatives(
        self,
        origin: Union[str, GeoLocation, Dict[str, float]],
        destination: Union[str, GeoLocation, Dict[str, float]],
        mode: TravelMode = TravelMode.DRIVING,
        departure_time: Optional[datetime] = None,
        language: str = "en",
    ) -> List[RouteInfo]:
        """Get alternative routes between two locations.

        Args:
            origin: Starting location
            destination: Ending location
            mode: Transportation mode
            departure_time: Departure time for traffic calculations
            language: Response language

        Returns:
            List[RouteInfo]: List of alternative routes sorted by duration

        Raises:
            DirectionsError: If route calculation fails
        """
        try:
            response = await self.get_directions(
                origin=origin,
                destination=destination,
                mode=mode,
                alternatives=True,
                departure_time=departure_time,
                language=language,
            )

            # Sort routes by duration
            routes = response.routes
            if routes:
                routes.sort(
                    key=lambda r: (
                        r.legs[0].duration.value
                        if r.legs and r.legs[0].duration
                        else float("inf")
                    )
                )

            return routes

        except Exception as e:
            if isinstance(e, DirectionsError):
                raise

            logger.error(
                "Alternative routes calculation failed",
                origin=str(origin),
                destination=str(destination),
                mode=mode.value,
                error=str(e),
                exc_info=True,
            )
            raise DirectionsError(
                origin=str(origin),
                destination=str(destination),
                message="Alternative routes calculation failed",
            ) from e


# Global service instance
_directions_service: Optional[DirectionsService] = None


def get_directions_service() -> DirectionsService:
    """Get or create global Directions service instance."""
    global _directions_service

    if _directions_service is None:
        _directions_service = DirectionsService()

    return _directions_service
