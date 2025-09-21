"""Maps Service Schemas for Google Maps API Integration.

This module provides Pydantic schemas for Google Maps API data models including
geolocation, place details, directions, and search parameters with validation.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TravelMode(str, Enum):
    """Transportation modes for directions and distance calculations."""

    DRIVING = "driving"
    WALKING = "walking"
    BICYCLING = "bicycling"
    TRANSIT = "transit"


class PlaceType(str, Enum):
    """Google Maps Place Types for filtering searches."""

    # Establishment types
    RESTAURANT = "restaurant"
    LODGING = "lodging"
    TOURIST_ATTRACTION = "tourist_attraction"
    MUSEUM = "museum"
    AMUSEMENT_PARK = "amusement_park"
    AQUARIUM = "aquarium"
    ART_GALLERY = "art_gallery"
    ZOO = "zoo"
    PARK = "park"

    # Services
    GAS_STATION = "gas_station"
    ATM = "atm"
    BANK = "bank"
    HOSPITAL = "hospital"
    PHARMACY = "pharmacy"

    # Shopping
    SHOPPING_MALL = "shopping_mall"
    STORE = "store"
    SUPERMARKET = "supermarket"

    # Transportation
    AIRPORT = "airport"
    BUS_STATION = "bus_station"
    SUBWAY_STATION = "subway_station"
    TRAIN_STATION = "train_station"

    # Entertainment
    MOVIE_THEATER = "movie_theater"
    NIGHT_CLUB = "night_club"
    BAR = "bar"
    CASINO = "casino"


class PriceLevel(int, Enum):
    """Price level indicators (0-4 scale)."""

    FREE = 0
    INEXPENSIVE = 1
    MODERATE = 2
    EXPENSIVE = 3
    VERY_EXPENSIVE = 4


class BusinessStatus(str, Enum):
    """Business operational status."""

    OPERATIONAL = "OPERATIONAL"
    CLOSED_TEMPORARILY = "CLOSED_TEMPORARILY"
    CLOSED_PERMANENTLY = "CLOSED_PERMANENTLY"


class GeoLocation(BaseModel):
    """Geographic location with latitude and longitude coordinates."""

    latitude: float = Field(
        ..., alias="lat", ge=-90.0, le=90.0, description="Latitude in decimal degrees"
    )
    longitude: float = Field(
        ...,
        alias="lng",
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees",
    )

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within valid range."""
        if not -90.0 <= v <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within valid range."""
        if not -180.0 <= v <= 180.0:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v

    def to_string(self) -> str:
        """Convert to comma-separated string format."""
        return f"{self.latitude},{self.longitude}"


class Bounds(BaseModel):
    """Geographic bounds defined by northeast and southwest corners."""

    northeast: GeoLocation = Field(..., description="Northeast corner of bounds")
    southwest: GeoLocation = Field(..., description="Southwest corner of bounds")

    @model_validator(mode="after")
    def validate_bounds(self) -> "Bounds":
        """Validate that bounds are logically consistent."""
        if self.northeast.latitude <= self.southwest.latitude:
            raise ValueError(
                "Northeast latitude must be greater than southwest latitude"
            )
        if self.northeast.longitude <= self.southwest.longitude:
            raise ValueError(
                "Northeast longitude must be greater than southwest longitude"
            )
        return self


class Geometry(BaseModel):
    """Place geometry information."""

    location: GeoLocation = Field(..., description="Primary location coordinates")
    location_type: Optional[str] = Field(None, description="Location type precision")
    viewport: Optional[Bounds] = Field(None, description="Recommended viewport bounds")
    bounds: Optional[Bounds] = Field(None, description="Bounding box for location")


class AddressComponent(BaseModel):
    """Individual component of a structured address."""

    long_name: str = Field(..., description="Full name of the address component")
    short_name: str = Field(..., description="Abbreviated name of the component")
    types: List[str] = Field(..., description="Types that apply to this component")


class OpeningHours(BaseModel):
    """Business opening hours information."""

    open_now: Optional[bool] = Field(None, description="Whether the place is open now")
    periods: Optional[List[dict]] = Field(None, description="Opening periods")
    weekday_text: Optional[List[str]] = Field(
        None, description="Human-readable opening hours"
    )


class Photo(BaseModel):
    """Place photo reference information."""

    photo_reference: str = Field(..., description="Photo reference for API requests")
    height: int = Field(..., gt=0, description="Photo height in pixels")
    width: int = Field(..., gt=0, description="Photo width in pixels")
    html_attributions: List[str] = Field(
        default_factory=list, description="Required attributions"
    )


class Review(BaseModel):
    """Place review information."""

    author_name: str = Field(..., description="Reviewer name")
    author_url: Optional[str] = Field(None, description="Reviewer profile URL")
    language: Optional[str] = Field(None, description="Review language")
    profile_photo_url: Optional[str] = Field(None, description="Reviewer photo URL")
    rating: int = Field(..., ge=1, le=5, description="Review rating (1-5)")
    relative_time_description: str = Field(
        ..., description="Human-readable time description"
    )
    text: str = Field(..., description="Review text content")
    time: int = Field(..., description="Review time as Unix timestamp")


class PlaceDetails(BaseModel):
    """Comprehensive place details from Google Maps."""

    place_id: str = Field(..., description="Unique place identifier")
    name: str = Field(..., description="Place name")
    formatted_address: Optional[str] = Field(None, description="Human-readable address")
    address_components: Optional[List[AddressComponent]] = Field(
        None, description="Structured address"
    )
    geometry: Geometry = Field(..., description="Place location and bounds")

    # Business information
    business_status: Optional[BusinessStatus] = Field(
        None, description="Business operational status"
    )
    types: List[str] = Field(default_factory=list, description="Place types")
    price_level: Optional[PriceLevel] = Field(None, description="Price level (0-4)")
    rating: Optional[float] = Field(
        None, ge=0, le=5, description="Average rating (0-5)"
    )
    user_ratings_total: Optional[int] = Field(
        None, ge=0, description="Total number of ratings"
    )

    # Contact information
    formatted_phone_number: Optional[str] = Field(
        None, description="Formatted phone number"
    )
    international_phone_number: Optional[str] = Field(
        None, description="International phone number"
    )
    website: Optional[str] = Field(None, description="Official website URL")
    url: Optional[str] = Field(None, description="Google Maps URL")

    # Hours and accessibility
    opening_hours: Optional[OpeningHours] = Field(
        None, description="Opening hours information"
    )
    wheelchair_accessible_entrance: Optional[bool] = Field(
        None, description="Wheelchair accessibility"
    )

    # Media and reviews
    photos: Optional[List[Photo]] = Field(None, description="Place photos")
    reviews: Optional[List[Review]] = Field(None, description="Place reviews")

    # Additional metadata
    plus_code: Optional[dict] = Field(None, description="Plus code location")
    utc_offset: Optional[int] = Field(None, description="UTC offset in minutes")


class PlaceSearchResult(BaseModel):
    """Simplified place information from search results."""

    place_id: str = Field(..., description="Unique place identifier")
    name: str = Field(..., description="Place name")
    formatted_address: Optional[str] = Field(None, description="Human-readable address")
    geometry: Geometry = Field(..., description="Place location")
    types: List[str] = Field(default_factory=list, description="Place types")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average rating")
    user_ratings_total: Optional[int] = Field(None, ge=0, description="Total ratings")
    price_level: Optional[PriceLevel] = Field(None, description="Price level")
    opening_hours: Optional[OpeningHours] = Field(None, description="Opening hours")
    photos: Optional[List[Photo]] = Field(None, description="Place photos")
    business_status: Optional[BusinessStatus] = Field(
        None, description="Business status"
    )


class Distance(BaseModel):
    """Distance information with text and numeric values."""

    text: str = Field(..., description="Human-readable distance")
    value: int = Field(..., ge=0, description="Distance in meters")


class Duration(BaseModel):
    """Duration information with text and numeric values."""

    text: str = Field(..., description="Human-readable duration")
    value: int = Field(..., ge=0, description="Duration in seconds")


class TransitDetails(BaseModel):
    """Transit-specific route information."""

    arrival_stop: Optional[dict] = Field(None, description="Arrival transit stop")
    departure_stop: Optional[dict] = Field(None, description="Departure transit stop")
    arrival_time: Optional[dict] = Field(None, description="Arrival time information")
    departure_time: Optional[dict] = Field(
        None, description="Departure time information"
    )
    headsign: Optional[str] = Field(None, description="Transit line destination")
    headway: Optional[int] = Field(None, description="Transit frequency in seconds")
    line: Optional[dict] = Field(None, description="Transit line information")
    num_stops: Optional[int] = Field(None, description="Number of stops")


class RouteStep(BaseModel):
    """Individual step in a route."""

    distance: Distance = Field(..., description="Step distance")
    duration: Duration = Field(..., description="Step duration")
    end_location: GeoLocation = Field(..., description="Step end coordinates")
    start_location: GeoLocation = Field(..., description="Step start coordinates")
    html_instructions: str = Field(..., description="HTML-formatted instructions")
    polyline: dict = Field(..., description="Encoded polyline points")
    travel_mode: TravelMode = Field(..., description="Transportation mode")
    maneuver: Optional[str] = Field(None, description="Driving maneuver type")
    transit_details: Optional[TransitDetails] = Field(
        None, description="Transit information"
    )
    steps: Optional[List["RouteStep"]] = Field(
        None, description="Sub-steps for detailed navigation"
    )


class RouteLeg(BaseModel):
    """Route segment between waypoints."""

    distance: Distance = Field(..., description="Leg total distance")
    duration: Duration = Field(..., description="Leg total duration")
    duration_in_traffic: Optional[Duration] = Field(
        None, description="Duration considering traffic"
    )
    end_address: str = Field(..., description="End point address")
    start_address: str = Field(..., description="Start point address")
    end_location: GeoLocation = Field(..., description="End point coordinates")
    start_location: GeoLocation = Field(..., description="Start point coordinates")
    steps: List[RouteStep] = Field(..., description="Turn-by-turn directions")
    traffic_speed_entry: Optional[List] = Field(
        None, description="Traffic speed information"
    )
    via_waypoint: Optional[List] = Field(None, description="Via waypoints on this leg")


class RouteInfo(BaseModel):
    """Complete route information from directions API."""

    bounds: Bounds = Field(..., description="Route bounding box")
    copyrights: str = Field(..., description="Route data copyrights")
    legs: List[RouteLeg] = Field(..., description="Route segments")
    overview_polyline: dict = Field(..., description="Route overview polyline")
    summary: str = Field(..., description="Route summary description")
    warnings: List[str] = Field(default_factory=list, description="Route warnings")
    waypoint_order: List[int] = Field(
        default_factory=list, description="Optimized waypoint order"
    )
    fare: Optional[dict] = Field(None, description="Transit fare information")


class DirectionsResponse(BaseModel):
    """Complete directions API response."""

    status: str = Field(..., description="Response status")
    routes: List[RouteInfo] = Field(..., description="Available routes")
    geocoded_waypoints: Optional[List[dict]] = Field(
        None, description="Geocoded waypoint information"
    )
    available_travel_modes: Optional[List[TravelMode]] = Field(
        None, description="Available travel modes"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if status != OK"
    )


class PlaceSearchFilters(BaseModel):
    """Filters for place search requests."""

    location: Optional[GeoLocation] = Field(None, description="Search center location")
    radius: Optional[int] = Field(
        None, ge=1, le=50000, description="Search radius in meters"
    )
    keyword: Optional[str] = Field(None, description="Search keyword")
    language: Optional[str] = Field("en", description="Response language")
    min_price: Optional[PriceLevel] = Field(None, description="Minimum price level")
    max_price: Optional[PriceLevel] = Field(None, description="Maximum price level")
    type: Optional[PlaceType] = Field(None, description="Place type filter")
    open_now: Optional[bool] = Field(None, description="Only return places open now")

    @model_validator(mode="after")
    def validate_price_range(self) -> "PlaceSearchFilters":
        """Validate price range is logical."""
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError(
                "Minimum price level cannot be greater than maximum price level"
            )
        return self


class NearbySearchRequest(BaseModel):
    """Request parameters for nearby places search."""

    location: GeoLocation = Field(..., description="Center location for search")
    radius: int = Field(..., ge=1, le=50000, description="Search radius in meters")
    keyword: Optional[str] = Field(None, description="Search keyword")
    language: Optional[str] = Field("en", description="Response language")
    type: Optional[PlaceType] = Field(None, description="Place type filter")
    min_price: Optional[PriceLevel] = Field(None, description="Minimum price level")
    max_price: Optional[PriceLevel] = Field(None, description="Maximum price level")
    open_now: Optional[bool] = Field(
        None, description="Only return currently open places"
    )


class TextSearchRequest(BaseModel):
    """Request parameters for text-based place search."""

    query: str = Field(..., min_length=1, description="Search query text")
    location: Optional[GeoLocation] = Field(None, description="Search bias location")
    radius: Optional[int] = Field(
        None, ge=1, le=50000, description="Search radius in meters"
    )
    language: Optional[str] = Field("en", description="Response language")
    type: Optional[PlaceType] = Field(None, description="Place type filter")
    min_price: Optional[PriceLevel] = Field(None, description="Minimum price level")
    max_price: Optional[PriceLevel] = Field(None, description="Maximum price level")
    open_now: Optional[bool] = Field(
        None, description="Only return currently open places"
    )


class PlaceAutocompleteRequest(BaseModel):
    """Request parameters for place autocomplete."""

    input: str = Field(..., min_length=1, description="Autocomplete input text")
    location: Optional[GeoLocation] = Field(None, description="Bias location")
    radius: Optional[int] = Field(
        None, ge=1, le=50000, description="Bias radius in meters"
    )
    language: Optional[str] = Field("en", description="Response language")
    types: Optional[List[str]] = Field(None, description="Place type restrictions")
    components: Optional[List[str]] = Field(
        None, description="Country component restrictions"
    )
    strict_bounds: Optional[bool] = Field(False, description="Strict location bounds")


class GeocodingRequest(BaseModel):
    """Request parameters for geocoding."""

    address: Optional[str] = Field(None, description="Address to geocode")
    location: Optional[GeoLocation] = Field(
        None, description="Coordinates for reverse geocoding"
    )
    place_id: Optional[str] = Field(None, description="Place ID to geocode")
    language: Optional[str] = Field("en", description="Response language")
    region: Optional[str] = Field(None, description="Region bias")
    components: Optional[dict] = Field(None, description="Component filters")

    @model_validator(mode="after")
    def validate_geocoding_input(self) -> "GeocodingRequest":
        """Validate that exactly one input parameter is provided."""
        inputs = [self.address, self.location, self.place_id]
        provided_inputs = [inp for inp in inputs if inp is not None]

        if len(provided_inputs) != 1:
            raise ValueError(
                "Exactly one of address, location, or place_id must be provided"
            )
        return self


class DirectionsRequest(BaseModel):
    """Request parameters for directions."""

    origin: Union[str, GeoLocation] = Field(..., description="Starting location")
    destination: Union[str, GeoLocation] = Field(..., description="Ending location")
    waypoints: Optional[List[Union[str, GeoLocation]]] = Field(
        None, description="Intermediate waypoints"
    )
    mode: TravelMode = Field(TravelMode.DRIVING, description="Transportation mode")
    alternatives: Optional[bool] = Field(
        False, description="Include alternative routes"
    )
    avoid: Optional[List[str]] = Field(None, description="Features to avoid")
    language: Optional[str] = Field("en", description="Response language")
    region: Optional[str] = Field(None, description="Region bias")
    units: Optional[str] = Field("metric", description="Unit system")
    departure_time: Optional[Union[datetime, str]] = Field(
        None, description="Departure time"
    )
    arrival_time: Optional[Union[datetime, str]] = Field(
        None, description="Arrival time"
    )
    traffic_model: Optional[str] = Field("best_guess", description="Traffic model")
    optimize_waypoints: Optional[bool] = Field(
        False, description="Optimize waypoint order"
    )

    @model_validator(mode="after")
    def validate_time_parameters(self) -> "DirectionsRequest":
        """Validate that departure_time and arrival_time are not both specified."""
        if self.departure_time is not None and self.arrival_time is not None:
            raise ValueError("Cannot specify both departure_time and arrival_time")
        return self


class DistanceMatrixRequest(BaseModel):
    """Request parameters for distance matrix."""

    origins: List[Union[str, GeoLocation]] = Field(
        ..., min_length=1, description="Origin locations"
    )
    destinations: List[Union[str, GeoLocation]] = Field(
        ..., min_length=1, description="Destination locations"
    )
    mode: TravelMode = Field(TravelMode.DRIVING, description="Transportation mode")
    language: Optional[str] = Field("en", description="Response language")
    region: Optional[str] = Field(None, description="Region bias")
    avoid: Optional[List[str]] = Field(None, description="Features to avoid")
    units: Optional[str] = Field("metric", description="Unit system")
    departure_time: Optional[Union[datetime, str]] = Field(
        None, description="Departure time"
    )
    arrival_time: Optional[Union[datetime, str]] = Field(
        None, description="Arrival time"
    )
    traffic_model: Optional[str] = Field("best_guess", description="Traffic model")


class MapsAPIResponse(BaseModel):
    """Base response model for Maps API calls."""

    status: str = Field(..., description="API response status")
    error_message: Optional[str] = Field(
        None, description="Error message if applicable"
    )
    info_messages: Optional[List[str]] = Field(
        None, description="Informational messages"
    )
    next_page_token: Optional[str] = Field(None, description="Token for pagination")


class PlaceSearchResponse(MapsAPIResponse):
    """Response model for place search requests."""

    results: List[PlaceSearchResult] = Field(
        default_factory=list, description="Search results"
    )
    html_attributions: List[str] = Field(
        default_factory=list, description="Required attributions"
    )


class GeocodingResponse(MapsAPIResponse):
    """Response model for geocoding requests."""

    results: List[dict] = Field(default_factory=list, description="Geocoding results")


class DistanceMatrixResponse(MapsAPIResponse):
    """Response model for distance matrix requests."""

    origin_addresses: List[str] = Field(
        default_factory=list, description="Origin addresses"
    )
    destination_addresses: List[str] = Field(
        default_factory=list, description="Destination addresses"
    )
    rows: List[dict] = Field(default_factory=list, description="Distance matrix rows")


# Enable forward references for recursive models
RouteStep.model_rebuild()
