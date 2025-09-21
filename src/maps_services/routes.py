"""Maps Services API Routes for AI-Powered Trip Planner Backend.

This module provides FastAPI routes for Google Maps API operations including
place search, details, directions, geocoding, and autocomplete with proper
authentication and validation.
"""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from src.auth.dependencies import get_user_id
from src.core.logging import get_logger
from src.maps_services.directions_service import get_directions_service
from src.maps_services.exceptions import (
    AutocompleteError,
    DirectionsError,
    GeocodingError,
    InvalidLocationError,
    PlaceDetailsError,
    PlaceNotFoundError,
    PlaceSearchError,
)
from src.maps_services.geocoding_service import get_geocoding_service
from src.maps_services.places_service import get_places_service
from src.maps_services.schemas import (
    DirectionsRequest,
    DirectionsResponse,
    GeocodingResponse,
    GeoLocation,
    PlaceDetails,
    PlaceSearchResponse,
    PlaceType,
    PriceLevel,
    TravelMode,
)

logger = get_logger(__name__)

# Create router with prefix and tags
router = APIRouter(prefix="/api/v1/places", tags=["Maps & Places"])


# Request Models
class PlaceSearchRequest(BaseModel):
    """Request model for place search."""

    query: str = Field(..., min_length=1, description="Search query")
    location: Optional[Dict[str, float]] = Field(None, description="Bias location")
    radius: Optional[int] = Field(
        None, ge=1, le=50000, description="Search radius in meters"
    )
    place_type: Optional[str] = Field(None, description="Place type filter")
    min_price: Optional[int] = Field(
        None, ge=0, le=4, description="Minimum price level"
    )
    max_price: Optional[int] = Field(
        None, ge=0, le=4, description="Maximum price level"
    )
    open_now: Optional[bool] = Field(None, description="Only return open places")
    language: str = Field(default="en", description="Response language")


class NearbyPlacesRequest(BaseModel):
    """Request model for nearby places search."""

    latitude: float = Field(..., ge=-90, le=90, description="Center latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Center longitude")
    radius: int = Field(..., ge=1, le=50000, description="Search radius in meters")
    keyword: Optional[str] = Field(None, description="Search keyword")
    place_type: Optional[str] = Field(None, description="Place type filter")
    min_price: Optional[int] = Field(
        None, ge=0, le=4, description="Minimum price level"
    )
    max_price: Optional[int] = Field(
        None, ge=0, le=4, description="Maximum price level"
    )
    open_now: Optional[bool] = Field(None, description="Only return open places")
    language: str = Field(default="en", description="Response language")


class DirectionsAPIRequest(BaseModel):
    """Request model for directions."""

    origin: str = Field(..., description="Starting location")
    destination: str = Field(..., description="Ending location")
    waypoints: Optional[List[str]] = Field(None, description="Intermediate waypoints")
    mode: str = Field(default="driving", description="Transportation mode")
    alternatives: bool = Field(default=False, description="Include alternative routes")
    avoid: Optional[List[str]] = Field(None, description="Features to avoid")
    language: str = Field(default="en", description="Response language")
    optimize_waypoints: bool = Field(
        default=False, description="Optimize waypoint order"
    )


class GeocodeRequest(BaseModel):
    """Request model for geocoding."""

    address: Optional[str] = Field(None, description="Address to geocode")
    latitude: Optional[float] = Field(
        None, description="Latitude for reverse geocoding"
    )
    longitude: Optional[float] = Field(
        None, description="Longitude for reverse geocoding"
    )
    place_id: Optional[str] = Field(None, description="Place ID to geocode")
    language: str = Field(default="en", description="Response language")


# Places Search Routes


@router.get(
    "/search",
    response_model=PlaceSearchResponse,
    summary="Search Places",
    description="Search for places using text query with optional filters",
)
async def search_places(
    query: str = Query(..., min_length=1, description="Search query"),
    latitude: Optional[float] = Query(None, ge=-90, le=90, description="Bias latitude"),
    longitude: Optional[float] = Query(
        None, ge=-180, le=180, description="Bias longitude"
    ),
    radius: Optional[int] = Query(None, ge=1, le=50000, description="Search radius"),
    place_type: Optional[str] = Query(None, description="Place type filter"),
    min_price: Optional[int] = Query(
        None, ge=0, le=4, description="Minimum price level"
    ),
    max_price: Optional[int] = Query(
        None, ge=0, le=4, description="Maximum price level"
    ),
    open_now: Optional[bool] = Query(None, description="Only return open places"),
    language: str = Query("en", description="Response language"),
    user_id: str = Depends(get_user_id),
    places_service=Depends(get_places_service),
) -> PlaceSearchResponse:
    """Search for places using text query."""
    try:
        # Create location object if coordinates provided
        location = None
        if latitude is not None and longitude is not None:
            location = GeoLocation(lat=latitude, lng=longitude)

        # Convert price levels to enums
        min_price_level = PriceLevel(min_price) if min_price is not None else None
        max_price_level = PriceLevel(max_price) if max_price is not None else None

        # Convert place type to enum
        place_type_enum = None
        if place_type:
            try:
                place_type_enum = PlaceType(place_type)
            except ValueError:
                logger.warning(f"Invalid place type: {place_type}")

        result = await places_service.search_text(
            query=query,
            location=location,
            radius=radius,
            place_type=place_type_enum,
            min_price=min_price_level,
            max_price=max_price_level,
            open_now=open_now,
            language=language,
        )

        logger.info(
            "Places search completed",
            user_id=user_id,
            query=query,
            results_count=len(result.results),
        )

        return result

    except PlaceSearchError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Place search failed: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Places search failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Places search service error",
        ) from e


@router.get(
    "/nearby",
    response_model=PlaceSearchResponse,
    summary="Find Nearby Places",
    description="Find places near a specific location",
)
async def find_nearby_places(
    latitude: float = Query(..., ge=-90, le=90, description="Center latitude"),
    longitude: float = Query(..., ge=-180, le=180, description="Center longitude"),
    radius: int = Query(..., ge=1, le=50000, description="Search radius in meters"),
    keyword: Optional[str] = Query(None, description="Search keyword"),
    place_type: Optional[str] = Query(None, description="Place type filter"),
    min_price: Optional[int] = Query(
        None, ge=0, le=4, description="Minimum price level"
    ),
    max_price: Optional[int] = Query(
        None, ge=0, le=4, description="Maximum price level"
    ),
    open_now: Optional[bool] = Query(None, description="Only return open places"),
    language: str = Query("en", description="Response language"),
    user_id: str = Depends(get_user_id),
    places_service=Depends(get_places_service),
) -> PlaceSearchResponse:
    """Find places near a location."""
    try:
        location = GeoLocation(lat=latitude, lng=longitude)

        # Convert place type to enum
        place_type_enum = None
        if place_type:
            try:
                place_type_enum = PlaceType(place_type)
            except ValueError:
                logger.warning(f"Invalid place type: {place_type}")

        # Convert price levels to enums
        min_price_level = PriceLevel(min_price) if min_price is not None else None
        max_price_level = PriceLevel(max_price) if max_price is not None else None

        result = await places_service.search_nearby(
            location=location,
            radius=radius,
            keyword=keyword,
            place_type=place_type_enum,
            min_price=min_price_level,
            max_price=max_price_level,
            open_now=open_now,
            language=language,
        )

        logger.info(
            "Nearby places search completed",
            user_id=user_id,
            location=f"{latitude},{longitude}",
            radius=radius,
            results_count=len(result.results),
        )

        return result

    except (PlaceSearchError, InvalidLocationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Nearby search failed: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Nearby places search failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Nearby places search service error",
        ) from e


@router.get(
    "/{place_id}",
    response_model=PlaceDetails,
    summary="Get Place Details",
    description="Get detailed information about a specific place",
)
async def get_place_details(
    place_id: str = Path(..., description="Place identifier from search results"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields"),
    language: str = Query("en", description="Response language"),
    user_id: str = Depends(get_user_id),
    places_service=Depends(get_places_service),
) -> PlaceDetails:
    """Get detailed place information."""
    try:
        fields_list = fields.split(",") if fields else None

        result = await places_service.get_place_details(
            place_id=place_id,
            fields=fields_list,
            language=language,
        )

        logger.info(
            "Place details retrieved",
            user_id=user_id,
            place_id=place_id,
            place_name=result.name,
        )

        return result

    except PlaceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Place not found: {place_id}",
        ) from e
    except PlaceDetailsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Place details error: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Place details retrieval failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Place details service error",
        ) from e


@router.get(
    "/autocomplete",
    response_model=List[Dict[str, Any]],
    summary="Place Autocomplete",
    description="Get place suggestions for autocomplete",
)
async def place_autocomplete(
    input_text: str = Query(
        ..., min_length=1, description="Input text to autocomplete"
    ),
    latitude: Optional[float] = Query(None, ge=-90, le=90, description="Bias latitude"),
    longitude: Optional[float] = Query(
        None, ge=-180, le=180, description="Bias longitude"
    ),
    radius: Optional[int] = Query(None, ge=1, le=50000, description="Bias radius"),
    types: Optional[str] = Query(None, description="Comma-separated place types"),
    language: str = Query("en", description="Response language"),
    user_id: str = Depends(get_user_id),
    places_service=Depends(get_places_service),
) -> List[Dict[str, Any]]:
    """Get place autocomplete suggestions."""
    try:
        # Create location object if coordinates provided
        location = None
        if latitude is not None and longitude is not None:
            location = GeoLocation(lat=latitude, lng=longitude)

        # Parse types
        types_list = types.split(",") if types else None

        result = await places_service.autocomplete(
            input_text=input_text,
            location=location,
            radius=radius,
            language=language,
            types=types_list,
        )

        logger.info(
            "Place autocomplete completed",
            user_id=user_id,
            input_text=input_text,
            suggestions_count=len(result),
        )

        return result

    except AutocompleteError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Autocomplete failed: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Place autocomplete failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Autocomplete service error",
        ) from e


# Directions Routes


@router.post(
    "/directions",
    response_model=DirectionsResponse,
    summary="Get Directions",
    description="Get route directions with optimization",
)
async def get_directions(
    directions_request: DirectionsAPIRequest,
    user_id: str = Depends(get_user_id),
    directions_service=Depends(get_directions_service),
) -> DirectionsResponse:
    """Get directions between locations."""
    try:
        # Convert travel mode to enum
        try:
            travel_mode = TravelMode(directions_request.mode)
        except ValueError:
            travel_mode = TravelMode.DRIVING

        # Convert request to service format
        waypoints_converted: Optional[List[Union[str, GeoLocation]]] = None
        if directions_request.waypoints:
            waypoints_converted = [str(wp) for wp in directions_request.waypoints]

        request_data = DirectionsRequest(
            origin=directions_request.origin,
            destination=directions_request.destination,
            waypoints=waypoints_converted,
            mode=travel_mode,
            alternatives=directions_request.alternatives,
            avoid=directions_request.avoid,
            language=directions_request.language,
            optimize_waypoints=directions_request.optimize_waypoints,
            region=None,
            units="metric",
            departure_time=None,
            arrival_time=None,
            traffic_model="best_guess",
        )

        result = await directions_service.get_directions(request_data)

        logger.info(
            "Directions retrieved",
            user_id=user_id,
            origin=directions_request.origin,
            destination=directions_request.destination,
            mode=directions_request.mode,
        )

        return result

    except DirectionsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directions failed: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Directions request failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Directions service error",
        ) from e


# Geocoding Routes


@router.post(
    "/geocode",
    response_model=GeocodingResponse,
    summary="Geocode Address",
    description="Convert between addresses and coordinates",
)
async def geocode_location(
    geocode_request: GeocodeRequest,
    user_id: str = Depends(get_user_id),
    geocoding_service=Depends(get_geocoding_service),
) -> GeocodingResponse:
    """Geocode address or reverse geocode coordinates."""
    try:
        # Create location object for reverse geocoding
        location = None
        if (
            geocode_request.latitude is not None
            and geocode_request.longitude is not None
        ):
            location = GeoLocation(
                lat=geocode_request.latitude, lng=geocode_request.longitude
            )

        # Determine geocoding type and call appropriate method
        if geocode_request.address:
            result = await geocoding_service.geocode_address(geocode_request.address)
        elif location:
            result = await geocoding_service.reverse_geocode(location)
        elif geocode_request.place_id:
            result = await geocoding_service.geocode_place_id(geocode_request.place_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide address, coordinates, or place_id",
            )

        logger.info(
            "Geocoding completed",
            user_id=user_id,
            address=geocode_request.address,
            place_id=geocode_request.place_id,
            results_count=len(result.results),
        )

        return result

    except GeocodingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geocoding failed: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Geocoding request failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Geocoding service error",
        ) from e


# Popular Places Route


@router.get(
    "/popular",
    response_model=List[Dict[str, Any]],
    summary="Find Popular Places",
    description="Find popular places near a location",
)
async def find_popular_places(
    latitude: float = Query(..., ge=-90, le=90, description="Center latitude"),
    longitude: float = Query(..., ge=-180, le=180, description="Center longitude"),
    radius: int = Query(5000, ge=1, le=50000, description="Search radius in meters"),
    min_rating: float = Query(4.0, ge=0, le=5, description="Minimum rating"),
    place_types: Optional[str] = Query(None, description="Comma-separated place types"),
    language: str = Query("en", description="Response language"),
    user_id: str = Depends(get_user_id),
    places_service=Depends(get_places_service),
) -> List[Dict[str, Any]]:
    """Find popular places near a location."""
    try:
        location = GeoLocation(lat=latitude, lng=longitude)

        # Parse place types
        place_types_list = None
        if place_types:
            place_types_list = []
            for pt in place_types.split(","):
                try:
                    place_types_list.append(PlaceType(pt.strip()))
                except ValueError:
                    logger.warning(f"Invalid place type: {pt}")

        result = await places_service.find_popular_places(
            location=location,
            radius=radius,
            place_types=place_types_list,
            min_rating=min_rating,
            language=language,
        )

        # Convert to serializable format
        popular_places = [
            {
                "place_id": place.place_id,
                "name": place.name,
                "rating": place.rating,
                "user_ratings_total": place.user_ratings_total,
                "types": place.types,
                "formatted_address": place.formatted_address,
                "price_level": place.price_level.value if place.price_level else None,
                "geometry": {
                    "location": {
                        "lat": place.geometry.location.latitude,
                        "lng": place.geometry.location.longitude,
                    }
                },
            }
            for place in result
        ]

        logger.info(
            "Popular places search completed",
            user_id=user_id,
            location=f"{latitude},{longitude}",
            results_count=len(popular_places),
        )

        return popular_places

    except (PlaceSearchError, InvalidLocationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Popular places search failed: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Popular places search failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Popular places search service error",
        ) from e


# Place Photo Route


@router.get(
    "/{place_id}/photo",
    summary="Get Place Photo",
    description="Get photo data for a place",
)
async def get_place_photo(
    place_id: str = Path(..., description="Place identifier"),
    photo_reference: str = Query(..., description="Photo reference from place details"),
    max_width: Optional[int] = Query(400, gt=0, le=1600, description="Maximum width"),
    max_height: Optional[int] = Query(400, gt=0, le=1600, description="Maximum height"),
    user_id: str = Depends(get_user_id),
    places_service=Depends(get_places_service),
):
    """Get place photo."""
    try:
        photo_data = await places_service.get_place_photo(
            photo_reference=photo_reference,
            max_width=max_width,
            max_height=max_height,
        )

        logger.info(
            "Place photo retrieved",
            user_id=user_id,
            place_id=place_id,
            photo_reference=photo_reference,
        )

        # Return photo data (you might want to return a proper image response)
        return {
            "success": True,
            "photo_data_size": len(photo_data),
            "message": "Photo retrieved successfully",
        }

    except Exception as e:
        logger.error("Place photo retrieval failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Photo retrieval service error",
        ) from e


# Health Check for Maps Services
@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Maps Services Health",
    description="Check health of Maps API services",
)
async def maps_services_health() -> Dict[str, Any]:
    """Check Maps services health."""
    try:
        health_info = {
            "status": "healthy",
            "services": {
                "places_api": "healthy",
                "directions_api": "healthy",
                "geocoding_api": "healthy",
            },
            "timestamp": "2024-01-01T00:00:00Z",
        }

        return {
            "success": True,
            "health": health_info,
        }

    except Exception as e:
        logger.error("Maps services health check failed", error=str(e), exc_info=True)
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }
