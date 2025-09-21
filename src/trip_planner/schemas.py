"""Trip Planning Schemas for AI-Powered Trip Planner Backend - Google ADK Multi-Agent System.

This module provides comprehensive Pydantic models for trip planning including TripRequest
with comprehensive trip parameters, TripItinerary with day-by-day plans and activities,
Activity, Place, and Transportation models, Budget and preference models, and Agent
response and intermediate result schemas.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.logging import get_logger

logger = get_logger(__name__)


class TripType(str, Enum):
    """Types of trips."""

    LEISURE = "leisure"
    BUSINESS = "business"
    ADVENTURE = "adventure"
    CULTURAL = "cultural"
    ROMANTIC = "romantic"
    FAMILY = "family"
    SOLO = "solo"
    GROUP = "group"
    EDUCATIONAL = "educational"
    WELLNESS = "wellness"
    LUXURY = "luxury"
    BUDGET = "budget"


class AccommodationType(str, Enum):
    """Types of accommodations."""

    HOTEL = "hotel"
    RESORT = "resort"
    GUESTHOUSE = "guesthouse"
    HOSTEL = "hostel"
    APARTMENT = "apartment"
    VILLA = "villa"
    HOMESTAY = "homestay"
    CAMPING = "camping"
    BOUTIQUE = "boutique"
    LUXURY = "luxury"


class TransportationMode(str, Enum):
    """Transportation modes."""

    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR = "car"
    TAXI = "taxi"
    RIDESHARE = "rideshare"
    METRO = "metro"
    FERRY = "ferry"
    WALKING = "walking"
    CYCLING = "cycling"
    SCOOTER = "scooter"
    AUTO_RICKSHAW = "auto_rickshaw"


class ActivityType(str, Enum):
    """Types of activities."""

    SIGHTSEEING = "sightseeing"
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    RELAXATION = "relaxation"
    DINING = "dining"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    NATURE = "nature"
    HISTORICAL = "historical"
    RELIGIOUS = "religious"
    NIGHTLIFE = "nightlife"
    WELLNESS = "wellness"
    EDUCATIONAL = "educational"


class PriceRange(str, Enum):
    """Price ranges for activities and services."""

    FREE = "free"
    BUDGET = "budget"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"
    LUXURY = "luxury"


class DifficultyLevel(str, Enum):
    """Difficulty levels for activities."""

    EASY = "easy"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    EXTREME = "extreme"


class GeoLocation(BaseModel):
    """Geographic location model."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    accuracy: Optional[float] = Field(None, description="Location accuracy in meters")

    def __str__(self) -> str:
        return f"{self.latitude},{self.longitude}"


class Address(BaseModel):
    """Address information model."""

    formatted_address: str = Field(..., description="Full formatted address")
    street_number: Optional[str] = Field(None, description="Street number")
    street_name: Optional[str] = Field(None, description="Street name")
    neighborhood: Optional[str] = Field(None, description="Neighborhood")
    city: str = Field(..., description="City name")
    state: Optional[str] = Field(None, description="State or province")
    country: str = Field(..., description="Country name")
    postal_code: Optional[str] = Field(None, description="Postal code")
    location: Optional[GeoLocation] = Field(None, description="Geographic coordinates")


class Contact(BaseModel):
    """Contact information model."""

    phone_number: Optional[str] = Field(None, description="Phone number")
    email: Optional[str] = Field(None, description="Email address")
    website: Optional[str] = Field(None, description="Website URL")
    social_media: Dict[str, str] = Field(
        default_factory=dict, description="Social media links"
    )


class OpeningHours(BaseModel):
    """Opening hours information."""

    is_open_now: Optional[bool] = Field(None, description="Currently open status")
    weekday_text: List[str] = Field(
        default_factory=list, description="Weekday hours text"
    )
    periods: List[Dict[str, Any]] = Field(
        default_factory=list, description="Opening periods"
    )
    special_hours: Dict[str, str] = Field(
        default_factory=dict, description="Special hours"
    )


class Place(BaseModel):
    """Place information model."""

    place_id: str = Field(..., description="Unique place identifier")
    name: str = Field(..., description="Place name")
    description: Optional[str] = Field(None, description="Place description")
    address: Address = Field(..., description="Place address")
    contact: Optional[Contact] = Field(None, description="Contact information")
    opening_hours: Optional[OpeningHours] = Field(None, description="Opening hours")
    rating: Optional[float] = Field(None, ge=0, le=5, description="User rating")
    review_count: Optional[int] = Field(None, ge=0, description="Number of reviews")
    price_level: Optional[PriceRange] = Field(None, description="Price range")
    place_types: List[str] = Field(default_factory=list, description="Place types")
    amenities: List[str] = Field(
        default_factory=list, description="Available amenities"
    )
    accessibility: Dict[str, bool] = Field(
        default_factory=dict, description="Accessibility features"
    )
    photos: List[str] = Field(default_factory=list, description="Photo URLs")
    reviews: List[Dict[str, Any]] = Field(
        default_factory=list, description="Recent reviews"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Transportation(BaseModel):
    """Transportation information model."""

    transport_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Transport ID"
    )
    mode: TransportationMode = Field(..., description="Transportation mode")
    origin: Union[str, GeoLocation] = Field(..., description="Origin location")
    destination: Union[str, GeoLocation] = Field(
        ..., description="Destination location"
    )
    departure_time: Optional[datetime] = Field(None, description="Departure time")
    arrival_time: Optional[datetime] = Field(None, description="Arrival time")
    duration: Optional[int] = Field(None, description="Travel duration in minutes")
    distance: Optional[float] = Field(None, description="Distance in kilometers")
    cost: Optional[Decimal] = Field(None, description="Transportation cost")
    currency: str = Field(default="INR", description="Currency code")
    provider: Optional[str] = Field(None, description="Service provider")
    booking_reference: Optional[str] = Field(None, description="Booking reference")
    route_details: Dict[str, Any] = Field(
        default_factory=dict, description="Route information"
    )
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alternative options"
    )


class Activity(BaseModel):
    """Activity information model."""

    activity_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Activity ID"
    )
    name: str = Field(..., description="Activity name")
    description: str = Field(..., description="Activity description")
    activity_type: ActivityType = Field(..., description="Type of activity")
    location: Place = Field(..., description="Activity location")
    start_time: Optional[datetime] = Field(None, description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    duration: int = Field(..., gt=0, description="Duration in minutes")
    cost: Optional[Decimal] = Field(None, ge=0, description="Activity cost")
    currency: str = Field(default="INR", description="Currency code")
    price_range: Optional[PriceRange] = Field(None, description="Price category")
    difficulty_level: Optional[DifficultyLevel] = Field(
        None, description="Difficulty level"
    )
    age_restrictions: Optional[Dict[str, int]] = Field(
        None, description="Age restrictions"
    )
    group_size_limits: Optional[Dict[str, int]] = Field(
        None, description="Group size limits"
    )
    booking_required: bool = Field(default=False, description="Booking required")
    cancellation_policy: Optional[str] = Field(None, description="Cancellation policy")
    inclusions: List[str] = Field(default_factory=list, description="What's included")
    exclusions: List[str] = Field(default_factory=list, description="What's excluded")
    requirements: List[str] = Field(
        default_factory=list, description="Special requirements"
    )
    tips: List[str] = Field(
        default_factory=list, description="Tips and recommendations"
    )
    alternatives: List[str] = Field(
        default_factory=list, description="Alternative activities"
    )
    weather_dependent: bool = Field(
        default=False, description="Weather dependent activity"
    )
    indoor_activity: bool = Field(default=False, description="Indoor activity")
    accessibility: Dict[str, bool] = Field(
        default_factory=dict, description="Accessibility features"
    )


class DayPlan(BaseModel):
    """Daily itinerary plan model."""

    day_number: int = Field(..., ge=1, description="Day number in trip")
    plan_date: date = Field(..., description="Date of the day")
    theme: Optional[str] = Field(None, description="Day theme or focus")
    activities: List[Activity] = Field(
        default_factory=list, description="Planned activities"
    )
    transportation: List[Transportation] = Field(
        default_factory=list, description="Transportation"
    )
    meals: List[Dict[str, Any]] = Field(
        default_factory=list, description="Meal recommendations"
    )
    accommodation: Optional[Place] = Field(
        None, description="Accommodation for the night"
    )
    total_cost: Optional[Decimal] = Field(None, ge=0, description="Total day cost")
    estimated_walking: Optional[float] = Field(
        None, description="Estimated walking distance (km)"
    )
    estimated_travel_time: Optional[int] = Field(
        None, description="Total travel time (minutes)"
    )
    weather_forecast: Optional[Dict[str, Any]] = Field(
        None, description="Weather forecast"
    )
    local_events: List[Dict[str, Any]] = Field(
        default_factory=list, description="Local events"
    )
    emergency_contacts: List[Dict[str, str]] = Field(
        default_factory=list, description="Emergency contacts"
    )
    notes: List[str] = Field(
        default_factory=list, description="Special notes for the day"
    )
    backup_plans: List[str] = Field(
        default_factory=list, description="Backup activity plans"
    )


class Budget(BaseModel):
    """Budget information model."""

    total_budget: Decimal = Field(..., gt=0, description="Total trip budget")
    currency: str = Field(default="INR", description="Budget currency")
    breakdown: Dict[str, Decimal] = Field(
        default_factory=dict, description="Budget breakdown"
    )
    contingency_percentage: float = Field(
        default=0.15, ge=0, le=0.5, description="Contingency buffer"
    )
    spent_amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Amount already spent"
    )
    remaining_amount: Optional[Decimal] = Field(None, description="Remaining budget")
    daily_budget: Optional[Decimal] = Field(None, description="Daily budget allocation")
    cost_optimization_tips: List[str] = Field(
        default_factory=list, description="Budget tips"
    )
    payment_methods: List[str] = Field(
        default_factory=list, description="Accepted payment methods"
    )

    @model_validator(mode="after")
    def calculate_remaining_budget(self):
        """Calculate remaining budget."""
        if self.total_budget and self.spent_amount:
            self.remaining_amount = self.total_budget - self.spent_amount
        return self


class TravelerProfile(BaseModel):
    """Traveler profile and preferences."""

    traveler_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Traveler ID"
    )
    name: Optional[str] = Field(None, description="Traveler name")
    age_group: Optional[str] = Field(None, description="Age group")
    interests: List[str] = Field(default_factory=list, description="Travel interests")
    activity_preferences: List[ActivityType] = Field(
        default_factory=list, description="Preferred activities"
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list, description="Dietary restrictions"
    )
    accessibility_needs: List[str] = Field(
        default_factory=list, description="Accessibility requirements"
    )
    fitness_level: Optional[DifficultyLevel] = Field(None, description="Fitness level")
    budget_preference: Optional[PriceRange] = Field(
        None, description="Budget preference"
    )
    accommodation_preference: Optional[AccommodationType] = Field(
        None, description="Accommodation preference"
    )
    transport_preferences: List[TransportationMode] = Field(
        default_factory=list, description="Transport preferences"
    )
    language_preferences: List[str] = Field(
        default_factory=list, description="Language preferences"
    )
    cultural_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Cultural preferences"
    )
    special_occasions: List[str] = Field(
        default_factory=list, description="Special occasions"
    )


class TripRequest(BaseModel):
    """Comprehensive trip request model."""

    request_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Request ID"
    )
    user_id: str = Field(..., description="User identifier")

    # Basic trip information
    destination: str = Field(..., description="Primary destination")
    additional_destinations: List[str] = Field(
        default_factory=list, description="Additional destinations"
    )
    start_date: date = Field(..., description="Trip start date")
    end_date: date = Field(..., description="Trip end date")
    duration_days: int = Field(..., gt=0, description="Trip duration in days")

    # Traveler information
    traveler_count: int = Field(
        default=1, gt=0, le=50, description="Number of travelers"
    )
    traveler_profiles: List[TravelerProfile] = Field(
        default_factory=list, description="Traveler profiles"
    )

    # Trip preferences
    trip_type: TripType = Field(default=TripType.LEISURE, description="Type of trip")
    budget: Optional[Budget] = Field(None, description="Budget information")
    preferred_activities: List[ActivityType] = Field(
        default_factory=list, description="Preferred activities"
    )
    accommodation_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Accommodation preferences"
    )
    transportation_preferences: List[TransportationMode] = Field(
        default_factory=list, description="Transport preferences"
    )

    # Special requirements
    special_requirements: List[str] = Field(
        default_factory=list, description="Special requirements"
    )
    accessibility_needs: List[str] = Field(
        default_factory=list, description="Accessibility needs"
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list, description="Dietary restrictions"
    )
    medical_considerations: List[str] = Field(
        default_factory=list, description="Medical considerations"
    )

    # Flexibility and constraints
    flexibility: Dict[str, Any] = Field(
        default_factory=dict, description="Flexibility options"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Hard constraints"
    )
    must_include: List[str] = Field(
        default_factory=list, description="Must-include activities/places"
    )
    avoid: List[str] = Field(default_factory=list, description="Things to avoid")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request creation time",
    )
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    priority: str = Field(default="normal", description="Request priority")
    source: str = Field(default="web", description="Request source")
    session_id: Optional[str] = Field(None, description="Session identifier")

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v, info):
        """Validate that end date is after start date."""
        if info.data and "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("End date must be after start date")
        return v

    @field_validator("duration_days")
    @classmethod
    def validate_duration(cls, v, info):
        """Validate duration matches date range."""
        if info.data and "start_date" in info.data and "end_date" in info.data:
            calculated_duration = (
                info.data["end_date"] - info.data["start_date"]
            ).days + 1
            if v != calculated_duration:
                raise ValueError(
                    f"Duration {v} does not match date range {calculated_duration} days"
                )
        return v


class AgentResponse(BaseModel):
    """Agent response model for multi-agent system."""

    response_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Response ID"
    )
    agent_id: str = Field(..., description="Agent identifier")
    agent_role: str = Field(..., description="Agent role")
    request_id: str = Field(..., description="Original request ID")

    # Response content
    response_type: str = Field(..., description="Type of response")
    content: Union[str, Dict[str, Any]] = Field(..., description="Response content")
    confidence_score: float = Field(
        default=0.8, ge=0, le=1, description="Confidence in response"
    )

    # Execution information
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed")
    function_calls: List[str] = Field(
        default_factory=list, description="Functions called"
    )

    # Quality metrics
    relevance_score: Optional[float] = Field(
        None, ge=0, le=1, description="Response relevance"
    )
    completeness_score: Optional[float] = Field(
        None, ge=0, le=1, description="Response completeness"
    )
    accuracy_score: Optional[float] = Field(
        None, ge=0, le=1, description="Response accuracy"
    )

    # Context and dependencies
    context_used: Dict[str, Any] = Field(
        default_factory=dict, description="Context used"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Dependent responses"
    )
    follow_up_required: bool = Field(default=False, description="Follow-up needed")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response time",
    )
    expires_at: Optional[datetime] = Field(None, description="Response expiration")

    # Error handling
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")


class TripItinerary(BaseModel):
    """Comprehensive trip itinerary model."""

    itinerary_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Itinerary ID"
    )
    request_id: str = Field(..., description="Original request ID")
    user_id: str = Field(..., description="User identifier")

    # Basic information
    title: str = Field(..., description="Itinerary title")
    description: str = Field(..., description="Itinerary description")
    destination: str = Field(..., description="Primary destination")
    start_date: date = Field(..., description="Trip start date")
    end_date: date = Field(..., description="Trip end date")
    duration_days: int = Field(..., gt=0, description="Trip duration")
    traveler_count: int = Field(..., gt=0, description="Number of travelers")

    # Itinerary content
    daily_plans: List[DayPlan] = Field(..., description="Day-by-day plans")
    overall_budget: Budget = Field(..., description="Trip budget")
    transportation_summary: List[Transportation] = Field(
        default_factory=list, description="Transportation overview"
    )
    accommodation_summary: List[Place] = Field(
        default_factory=list, description="Accommodations"
    )

    # Optimization metrics
    optimization_score: Dict[str, float] = Field(
        default_factory=dict, description="Optimization scores"
    )
    efficiency_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Efficiency metrics"
    )

    # Practical information
    packing_suggestions: List[str] = Field(
        default_factory=list, description="Packing suggestions"
    )
    local_customs: List[str] = Field(
        default_factory=list, description="Local customs and etiquette"
    )
    safety_tips: List[str] = Field(
        default_factory=list, description="Safety recommendations"
    )
    emergency_information: Dict[str, Any] = Field(
        default_factory=dict, description="Emergency contacts/info"
    )

    # Agent contributions
    agent_responses: List[AgentResponse] = Field(
        default_factory=list, description="Agent contributions"
    )
    creation_workflow: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow used"
    )

    # Status and metadata
    status: str = Field(default="draft", description="Itinerary status")
    version: int = Field(default=1, description="Itinerary version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time",
    )
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    created_by_agents: List[str] = Field(
        default_factory=list, description="Contributing agents"
    )

    @field_validator("daily_plans")
    @classmethod
    def validate_daily_plans(cls, v, info):
        """Validate daily plans match trip duration."""
        if (
            info.data
            and "duration_days" in info.data
            and len(v) != info.data["duration_days"]
        ):
            raise ValueError(
                f"Daily plans count {len(v)} does not match duration {info.data['duration_days']}"
            )
        return v

    def get_total_cost(self) -> Decimal:
        """Calculate total trip cost."""
        total = Decimal("0")

        for day_plan in self.daily_plans:
            if day_plan.total_cost:
                total += day_plan.total_cost

        return total

    def get_activity_summary(self) -> Dict[str, int]:
        """Get summary of activities by type."""
        activity_counts = {}

        for day_plan in self.daily_plans:
            for activity in day_plan.activities:
                activity_type = activity.activity_type.value
                activity_counts[activity_type] = (
                    activity_counts.get(activity_type, 0) + 1
                )

        return activity_counts


class WorkflowResult(BaseModel):
    """Result of workflow execution."""

    workflow_id: str = Field(..., description="Workflow identifier")
    execution_id: str = Field(..., description="Execution identifier")
    request_id: str = Field(..., description="Original request ID")

    # Results
    success: bool = Field(..., description="Execution success")
    itinerary: Optional[TripItinerary] = Field(None, description="Generated itinerary")
    agent_responses: List[AgentResponse] = Field(
        default_factory=list, description="Agent responses"
    )

    # Execution metrics
    execution_time: float = Field(..., ge=0, description="Total execution time")
    agents_used: List[str] = Field(default_factory=list, description="Agents involved")
    function_calls_total: int = Field(default=0, description="Total function calls")
    tokens_consumed: int = Field(default=0, description="Total tokens used")

    # Quality assessment
    quality_scores: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )
    user_satisfaction_predicted: Optional[float] = Field(
        None, description="Predicted satisfaction"
    )

    # Error handling
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result creation time",
    )
    workflow_version: str = Field(default="1.0", description="Workflow version used")


class ItineraryOptimization(BaseModel):
    """Itinerary optimization suggestions."""

    optimization_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Optimization ID"
    )
    itinerary_id: str = Field(..., description="Target itinerary ID")

    # Optimization criteria
    criteria: List[str] = Field(..., description="Optimization criteria")
    weights: Dict[str, float] = Field(
        default_factory=dict, description="Criteria weights"
    )

    # Suggestions
    time_optimizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Time optimizations"
    )
    cost_optimizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Cost optimizations"
    )
    experience_optimizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Experience optimizations"
    )
    route_optimizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Route optimizations"
    )

    # Impact assessment
    estimated_improvements: Dict[str, str] = Field(
        default_factory=dict, description="Expected improvements"
    )
    implementation_difficulty: Dict[str, str] = Field(
        default_factory=dict, description="Implementation difficulty"
    )

    # Scores
    current_scores: Dict[str, float] = Field(
        default_factory=dict, description="Current performance scores"
    )
    optimized_scores: Dict[str, float] = Field(
        default_factory=dict, description="Projected optimized scores"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time",
    )


class AgentCapabilityAssessment(BaseModel):
    """Assessment of agent capabilities for specific tasks."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_role: str = Field(..., description="Agent role")
    task_type: str = Field(..., description="Task type being assessed")

    # Capability scores
    domain_expertise: float = Field(
        ..., ge=0, le=1, description="Domain expertise score"
    )
    function_availability: float = Field(
        ..., ge=0, le=1, description="Required functions available"
    )
    context_compatibility: float = Field(
        ..., ge=0, le=1, description="Context compatibility"
    )
    performance_history: float = Field(
        ..., ge=0, le=1, description="Historical performance"
    )

    # Resource requirements
    estimated_execution_time: int = Field(
        ..., gt=0, description="Estimated execution time (seconds)"
    )
    estimated_token_usage: int = Field(
        default=0, description="Estimated token consumption"
    )
    required_functions: List[str] = Field(
        default_factory=list, description="Required function tools"
    )

    # Suitability assessment
    overall_suitability: float = Field(
        ..., ge=0, le=1, description="Overall suitability score"
    )
    recommended: bool = Field(..., description="Agent recommended for task")
    reasoning: str = Field(..., description="Assessment reasoning")

    # Alternative suggestions
    alternative_agents: List[str] = Field(
        default_factory=list, description="Alternative agent suggestions"
    )
    capability_gaps: List[str] = Field(
        default_factory=list, description="Identified capability gaps"
    )

    assessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Assessment time",
    )


class TripPlanningSession(BaseModel):
    """Trip planning session state model."""

    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")

    # Session state
    current_request: Optional[TripRequest] = Field(
        None, description="Current trip request"
    )
    active_workflows: List[str] = Field(
        default_factory=list, description="Active workflow IDs"
    )
    completed_workflows: List[str] = Field(
        default_factory=list, description="Completed workflow IDs"
    )

    # Generated content
    generated_itineraries: List[TripItinerary] = Field(
        default_factory=list, description="Generated itineraries"
    )
    optimization_suggestions: List[ItineraryOptimization] = Field(
        default_factory=list, description="Optimizations"
    )
    agent_assessments: List[AgentCapabilityAssessment] = Field(
        default_factory=list, description="Agent assessments"
    )

    # Session context
    conversation_context: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation context"
    )
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Learned preferences"
    )
    session_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Session metadata"
    )

    # Performance tracking
    total_execution_time: float = Field(
        default=0.0, description="Total session execution time"
    )
    total_tokens_used: int = Field(default=0, description="Total tokens consumed")
    function_calls_made: int = Field(default=0, description="Total function calls")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session start",
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last activity",
    )
    expires_at: Optional[datetime] = Field(None, description="Session expiration")

    def add_itinerary(self, itinerary: TripItinerary) -> None:
        """Add generated itinerary to session."""
        self.generated_itineraries.append(itinerary)
        self.last_activity = datetime.now(timezone.utc)

    def get_latest_itinerary(self) -> Optional[TripItinerary]:
        """Get the most recently generated itinerary."""
        if self.generated_itineraries:
            return max(self.generated_itineraries, key=lambda x: x.created_at)
        return None

    def update_session_stats(
        self, execution_time: float, tokens_used: int, function_calls: int
    ) -> None:
        """Update session performance statistics."""
        self.total_execution_time += execution_time
        self.total_tokens_used += tokens_used
        self.function_calls_made += function_calls
        self.last_activity = datetime.now(timezone.utc)


class TripRecommendation(BaseModel):
    """Trip recommendation model."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Recommendation ID"
    )
    source_agent: str = Field(..., description="Agent that generated recommendation")

    # Recommendation content
    recommendation_type: str = Field(..., description="Type of recommendation")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    rationale: str = Field(..., description="Why this is recommended")

    # Recommendation data
    recommended_places: List[Place] = Field(
        default_factory=list, description="Recommended places"
    )
    recommended_activities: List[Activity] = Field(
        default_factory=list, description="Recommended activities"
    )
    recommended_routes: List[Transportation] = Field(
        default_factory=list, description="Recommended routes"
    )

    # Scoring and ranking
    relevance_score: float = Field(
        ..., ge=0, le=1, description="Relevance to user preferences"
    )
    popularity_score: float = Field(
        default=0.5, ge=0, le=1, description="General popularity"
    )
    uniqueness_score: float = Field(
        default=0.5, ge=0, le=1, description="Uniqueness factor"
    )
    value_score: float = Field(default=0.5, ge=0, le=1, description="Value for money")

    # Practical information
    best_time_to_visit: Optional[str] = Field(None, description="Best time to visit")
    estimated_duration: Optional[int] = Field(
        None, description="Estimated time needed (minutes)"
    )
    estimated_cost: Optional[Decimal] = Field(None, description="Estimated cost")
    booking_requirements: List[str] = Field(
        default_factory=list, description="Booking requirements"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time",
    )
    tags: List[str] = Field(default_factory=list, description="Recommendation tags")

    def get_overall_score(self) -> float:
        """Calculate overall recommendation score."""
        weights = {
            "relevance": 0.4,
            "popularity": 0.2,
            "uniqueness": 0.2,
            "value": 0.2,
        }

        return (
            self.relevance_score * weights["relevance"]
            + self.popularity_score * weights["popularity"]
            + self.uniqueness_score * weights["uniqueness"]
            + self.value_score * weights["value"]
        )


# Utility functions for schema operations


def create_trip_request_from_dict(data: Dict[str, Any]) -> TripRequest:
    """Create TripRequest from dictionary data."""
    try:
        return TripRequest.model_validate(data)
    except Exception as e:
        logger.exception("Failed to create TripRequest from data")
        raise ValueError(f"Invalid trip request data: {e}") from e


def merge_agent_responses(responses: List[AgentResponse]) -> Dict[str, Any]:
    """Merge multiple agent responses into consolidated data."""

    merged_data = {
        "response_count": len(responses),
        "agents_involved": [r.agent_id for r in responses],
        "agent_roles": [r.agent_role for r in responses],
        "total_execution_time": sum(r.execution_time for r in responses),
        "total_tokens": sum(r.tokens_used for r in responses),
        "average_confidence": (
            sum(r.confidence_score for r in responses) / len(responses)
            if responses
            else 0
        ),
        "consolidated_content": {},
        "all_function_calls": [],
        "combined_warnings": [],
        "combined_errors": [],
    }

    # Merge content by agent role
    for response in responses:
        role = response.agent_role
        merged_data["consolidated_content"][role] = response.content
        merged_data["all_function_calls"].extend(response.function_calls)
        merged_data["combined_warnings"].extend(response.warnings)
        merged_data["combined_errors"].extend(response.errors)

    return merged_data


def calculate_itinerary_metrics(itinerary: TripItinerary) -> Dict[str, Any]:
    """Calculate comprehensive metrics for an itinerary."""

    metrics = {
        "total_cost": float(itinerary.get_total_cost()),
        "activity_distribution": itinerary.get_activity_summary(),
        "daily_averages": {
            "activities_per_day": len(
                [a for day in itinerary.daily_plans for a in day.activities]
            )
            / itinerary.duration_days,
            "cost_per_day": float(itinerary.get_total_cost()) / itinerary.duration_days,
        },
        "transportation_modes": {},
        "accommodation_types": {},
        "coverage_analysis": {
            "destinations_covered": len(
                set(
                    [
                        day.accommodation.address.city
                        for day in itinerary.daily_plans
                        if day.accommodation
                    ]
                )
            ),
            "activity_types_covered": len(
                set(
                    [
                        a.activity_type.value
                        for day in itinerary.daily_plans
                        for a in day.activities
                    ]
                )
            ),
        },
    }

    # Analyze transportation modes
    for day in itinerary.daily_plans:
        for transport in day.transportation:
            mode = transport.mode.value
            metrics["transportation_modes"][mode] = (
                metrics["transportation_modes"].get(mode, 0) + 1
            )

    return metrics


def validate_itinerary_completeness(itinerary: TripItinerary) -> Dict[str, Any]:
    """Validate itinerary completeness and identify gaps."""

    validation_result = {
        "is_complete": True,
        "completeness_score": 1.0,
        "missing_elements": [],
        "warnings": [],
        "recommendations": [],
    }

    issues = []

    # Check if all days have plans
    if len(itinerary.daily_plans) != itinerary.duration_days:
        issues.append("Daily plans count mismatch")
        validation_result["missing_elements"].append("daily_plans")

    # Check if each day has activities
    empty_days = [
        i for i, day in enumerate(itinerary.daily_plans) if not day.activities
    ]
    if empty_days:
        issues.append(f"Days without activities: {empty_days}")
        validation_result["missing_elements"].append("activities")

    # Check accommodation coverage
    days_without_accommodation = [
        i
        for i, day in enumerate(itinerary.daily_plans)
        if not day.accommodation
        and i < len(itinerary.daily_plans) - 1  # Last day might not need accommodation
    ]
    if days_without_accommodation:
        issues.append(f"Days without accommodation: {days_without_accommodation}")
        validation_result["warnings"].append("accommodation_gaps")

    # Check budget coverage
    if not itinerary.overall_budget.total_budget:
        issues.append("Missing budget information")
        validation_result["missing_elements"].append("budget")

    # Calculate completeness score
    total_checks = 4  # Number of validation checks
    passed_checks = total_checks - len(
        [e for e in validation_result["missing_elements"]]
    )
    validation_result["completeness_score"] = passed_checks / total_checks
    validation_result["is_complete"] = validation_result["completeness_score"] >= 0.8

    if issues:
        validation_result["warnings"].extend(issues)

    return validation_result
