"""Database models for AI-Powered Trip Planner Backend.

This module defines dataclass models for Firestore documents with proper
type annotations and validation for consistent data structure.
"""

import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Self


# Enums for consistent values
class TripStatus(str, Enum):
    """Trip status enumeration."""

    DRAFT = "draft"
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ActivityType(str, Enum):
    """Activity type enumeration."""

    SIGHTSEEING = "sightseeing"
    RESTAURANT = "restaurant"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    OUTDOOR = "outdoor"
    CULTURAL = "cultural"
    NIGHTLIFE = "nightlife"
    TRANSPORTATION = "transportation"
    ACCOMMODATION = "accommodation"
    OTHER = "other"


class TransportationMode(str, Enum):
    """Transportation mode enumeration."""

    WALKING = "walking"
    DRIVING = "driving"
    TRANSIT = "transit"
    BICYCLING = "bicycling"
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    TAXI = "taxi"
    RIDESHARE = "rideshare"


class SessionStatus(str, Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


# Base dataclass for all documents
@dataclass
class BaseDocument:
    """Base document class with common fields."""

    id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary for Firestore."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                if isinstance(field_value, datetime):
                    result[field_name] = field_value
                elif isinstance(field_value, Enum):
                    result[field_name] = field_value.value
                elif isinstance(field_value, list | dict):
                    result[field_name] = field_value
                else:
                    result[field_name] = field_value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], doc_id: str | None = None) -> Self:
        """Create dataclass instance from Firestore document."""
        if doc_id:
            data["id"] = doc_id

        # Convert datetime strings back to datetime objects
        for field_name in [
            "created_at",
            "updated_at",
            "last_login_at",
            "expires_at",
            "start_date",
            "end_date",
        ]:
            if field_name in data and isinstance(data[field_name], str):
                with contextlib.suppress(ValueError, AttributeError):
                    data[field_name] = datetime.fromisoformat(
                        data[field_name].replace("Z", "+00:00")
                    )

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# User Preferences
@dataclass
class UserPreferencesDocument:
    """User preferences document model."""

    currency: str = "INR"
    timezone: str = "Asia/Kolkata"
    language: str = "en"
    country: str = "India"

    # Travel preferences
    travel_style: str | None = None
    accommodation_type: str | None = None
    transportation_mode: str | None = None
    food_preferences: list[str] = field(default_factory=list)
    activity_interests: list[str] = field(default_factory=list)

    # Trip planning preferences
    default_trip_duration: int = 7
    budget_range: str | None = None
    group_size_preference: int | None = None

    # Notification preferences
    email_notifications: bool = True
    push_notifications: bool = True
    marketing_emails: bool = False


# User Document
@dataclass
class UserDocument(BaseDocument):
    """User document model for Firestore."""

    uid: str = ""
    email: str | None = None
    email_verified: bool = False

    # Profile information
    display_name: str | None = None
    photo_url: str | None = None
    phone_number: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: datetime | None = None
    bio: str | None = None

    # User preferences
    preferences: UserPreferencesDocument = field(
        default_factory=UserPreferencesDocument
    )

    # Account status
    profile_complete: bool = False
    terms_accepted: bool = False
    privacy_policy_accepted: bool = False

    # Metadata
    last_login_at: datetime | None = None
    login_count: int = 0
    is_admin: bool = False
    is_active: bool = True

    # Statistics
    trips_created: int = 0
    trips_completed: int = 0
    favorite_destinations: list[str] = field(default_factory=list)


# Session Document
@dataclass
class SessionDocument(BaseDocument):
    """Session document model for Firestore."""

    user_id: str = ""
    session_token: str = ""
    firebase_token: str = ""

    # Session metadata
    status: SessionStatus = SessionStatus.ACTIVE
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Client information
    client_ip: str | None = None
    user_agent: str | None = None
    device_info: dict[str, Any] | None = None

    # Activity tracking
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    activity_count: int = 0


# Location Information
@dataclass
class LocationDocument:
    """Location document model."""

    name: str = ""
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    place_id: str | None = None  # Google Places API ID

    # Location metadata
    city: str | None = None
    state: str | None = None
    country: str | None = None
    postal_code: str | None = None

    # Additional information
    rating: float | None = None
    price_level: int | None = None
    types: list[str] = field(default_factory=list)
    photos: list[str] = field(default_factory=list)


# Activity/Place Information
@dataclass
class ActivityDocument:
    """Activity document model."""

    name: str = ""
    description: str | None = None
    activity_type: ActivityType = ActivityType.OTHER

    # Location information
    location: LocationDocument = field(default_factory=LocationDocument)

    # Timing information
    duration_minutes: int | None = None
    best_time_of_day: str | None = None  # morning, afternoon, evening, night
    seasonal_info: str | None = None

    # Cost information
    estimated_cost: float | None = None
    currency: str = "INR"
    cost_notes: str | None = None

    # Booking information
    booking_required: bool = False
    booking_url: str | None = None
    contact_info: dict[str, str] | None = None

    # Additional metadata
    tags: list[str] = field(default_factory=list)
    ai_generated: bool = True
    user_added: bool = False


# Daily Itinerary
@dataclass
class DayItineraryDocument:
    """Daily itinerary document model."""

    day_number: int = 1
    date: datetime | None = None

    # Activities for the day
    activities: list[ActivityDocument] = field(default_factory=list)

    # Transportation between activities
    transportation: list[dict[str, Any]] = field(default_factory=list)

    # Daily summary
    total_duration_minutes: int | None = None
    estimated_cost: float | None = None
    notes: str | None = None

    # Status
    completed: bool = False
    user_modified: bool = False


# Trip Document
@dataclass
class TripDocument(BaseDocument):
    """Trip document model for Firestore."""

    user_id: str = ""
    title: str = ""
    description: str | None = None

    # Trip basic information
    destination: str = ""
    start_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_days: int = 1

    # Trip preferences
    budget: float | None = None
    currency: str = "INR"
    group_size: int = 1
    travel_style: str | None = None

    # Trip status and metadata
    status: TripStatus = TripStatus.DRAFT

    # Itinerary
    daily_itineraries: list[DayItineraryDocument] = field(default_factory=list)

    # AI generation metadata
    ai_prompt: str | None = None
    ai_model_used: str | None = None
    generation_timestamp: datetime | None = None
    regeneration_count: int = 0

    # User interaction
    user_rating: int | None = None  # 1-5 stars
    user_feedback: str | None = None
    shared: bool = False
    shared_url: str | None = None

    # Trip completion tracking
    completion_percentage: float = 0.0
    completed_activities: int = 0
    total_activities: int = 0


# Chat/Conversation Document for AI interactions
@dataclass
class ChatMessageDocument:
    """Chat message document model."""

    message_id: str = ""
    role: str = "user"  # user, assistant, system
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Message metadata
    tokens_used: int | None = None
    model_used: str | None = None
    processing_time: float | None = None

    # Message context
    trip_id: str | None = None
    context_data: dict[str, Any] | None = None


@dataclass
class ConversationDocument(BaseDocument):
    """Conversation document model for AI chat sessions."""

    user_id: str = ""
    title: str | None = None

    # Conversation metadata
    messages: list[ChatMessageDocument] = field(default_factory=list)
    total_messages: int = 0
    total_tokens: int = 0

    # Related trip information
    related_trip_id: str | None = None
    conversation_type: str = "general"  # general, trip_planning, modification

    # Status
    is_active: bool = True
    last_message_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# Feedback Document
@dataclass
class FeedbackDocument(BaseDocument):
    """Feedback document model."""

    user_id: str = ""
    type: str = "general"  # general, trip, bug_report, feature_request

    # Feedback content
    title: str | None = None
    content: str = ""
    rating: int | None = None  # 1-5 stars

    # Context information
    trip_id: str | None = None
    page_url: str | None = None
    user_agent: str | None = None

    # Status
    status: str = "pending"  # pending, reviewed, resolved
    admin_notes: str | None = None
    resolved_at: datetime | None = None


# Analytics Event Document
@dataclass
class AnalyticsEventDocument(BaseDocument):
    """Analytics event document model."""

    user_id: str | None = None
    session_id: str | None = None

    # Event information
    event_type: str = ""
    event_name: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    # Context
    page_url: str | None = None
    referrer: str | None = None
    user_agent: str | None = None
    client_ip: str | None = None

    # Device/browser info
    device_type: str | None = None
    browser: str | None = None
    os: str | None = None


# Error Log Document
@dataclass
class ErrorLogDocument(BaseDocument):
    """Error log document model."""

    error_id: str = ""

    # Error information
    error_type: str = ""
    error_message: str = ""
    stack_trace: str | None = None

    # Context information
    user_id: str | None = None
    request_id: str | None = None
    endpoint: str | None = None
    http_method: str | None = None

    # Client information
    user_agent: str | None = None
    client_ip: str | None = None

    # Additional context
    context_data: dict[str, Any] = field(default_factory=dict)

    # Status
    resolved: bool = False
    resolution_notes: str | None = None


# API Usage Document for rate limiting and analytics
@dataclass
class ApiUsageDocument(BaseDocument):
    """API usage document model."""

    user_id: str | None = None
    endpoint: str = ""
    http_method: str = ""

    # Usage metrics
    request_count: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0

    # Status codes
    status_codes: dict[str, int] = field(default_factory=dict)

    # Time window
    window_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    window_end: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Rate limiting
    rate_limited_count: int = 0
    last_rate_limit: datetime | None = None


# Collection name mappings
COLLECTION_NAMES = {
    UserDocument: "users",
    SessionDocument: "sessions",
    TripDocument: "trips",
    ConversationDocument: "conversations",
    FeedbackDocument: "feedback",
    AnalyticsEventDocument: "analytics_events",
    ErrorLogDocument: "error_logs",
    ApiUsageDocument: "api_usage",
}


def get_collection_name(model_class: type) -> str:
    """Get Firestore collection name for a model class.

    Args:
        model_class: The model class.

    Returns:
        str: The collection name.
    """
    return COLLECTION_NAMES.get(model_class) or model_class.__name__.lower()
