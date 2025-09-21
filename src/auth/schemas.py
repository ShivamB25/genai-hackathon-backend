"""User schemas and Pydantic models for AI-Powered Trip Planner Backend.

This module defines Pydantic models for user authentication, profiles, and preferences
with comprehensive validation and serialization.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from src.core.config import settings


class UserPreferences(BaseModel):
    """User preferences and settings model."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    # Location preferences
    currency: str = Field(
        default=settings.default_budget_currency,
        description="Preferred currency for budgeting",
        examples=["USD", "EUR", "INR"],
    )
    timezone: str = Field(
        default=settings.default_timezone,
        description="User's timezone",
        examples=["Asia/Kolkata", "America/New_York", "Europe/London"],
    )
    language: str = Field(
        default="en", description="Preferred language code", examples=["en", "hi", "es"]
    )
    country: str = Field(
        default=settings.default_country,
        description="Primary country of residence",
        examples=["India", "United States", "United Kingdom"],
    )

    # Travel preferences
    travel_style: str | None = Field(
        default=None,
        description="Preferred travel style",
        examples=["budget", "luxury", "adventure", "cultural", "relaxed"],
    )
    accommodation_type: str | None = Field(
        default=None,
        description="Preferred accommodation type",
        examples=["hotel", "hostel", "vacation_rental", "resort", "boutique"],
    )
    transportation_mode: str | None = Field(
        default=None,
        description="Preferred transportation mode",
        examples=["flight", "train", "car", "bus", "mixed"],
    )
    food_preferences: list[str] = Field(
        default_factory=list,
        description="Food preferences and dietary restrictions",
        examples=["vegetarian", "vegan", "halal", "kosher", "gluten_free"],
    )
    activity_interests: list[str] = Field(
        default_factory=list,
        description="Types of activities interested in",
        examples=["museums", "outdoor", "nightlife", "shopping", "cultural"],
    )

    # Trip planning preferences
    default_trip_duration: int = Field(
        default=settings.default_trip_duration_days,
        ge=settings.min_trip_duration_days,
        le=settings.max_trip_duration_days,
        description="Default trip duration in days",
    )
    budget_range: str | None = Field(
        default=None,
        description="Typical budget range",
        examples=["budget", "mid_range", "luxury"],
    )
    group_size_preference: int | None = Field(
        default=None, ge=1, le=20, description="Preferred group size for travel"
    )

    # Notification preferences
    email_notifications: bool = Field(
        default=True, description="Enable email notifications"
    )
    push_notifications: bool = Field(
        default=True, description="Enable push notifications"
    )
    marketing_emails: bool = Field(default=False, description="Enable marketing emails")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code."""
        if v not in settings.supported_languages:
            msg = f"Language '{v}' not supported. Supported: {settings.supported_languages}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("food_preferences", "activity_interests", mode="before")
    @classmethod
    def validate_lists(cls, v) -> list[str]:
        """Ensure lists contain unique lowercase strings."""
        if not isinstance(v, list):
            return []
        return list(set(str(item).lower().strip() for item in v if item))


class UserProfile(BaseModel):
    """Complete user profile model."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    # Firebase Auth data
    uid: str = Field(description="Firebase user UID")
    email: EmailStr | None = Field(default=None, description="User email address")
    email_verified: bool = Field(default=False, description="Email verification status")

    # Profile information
    display_name: str | None = Field(
        default=None, min_length=1, max_length=100, description="User's display name"
    )
    photo_url: str | None = Field(default=None, description="Profile photo URL")
    phone_number: str | None = Field(default=None, description="User's phone number")

    # Additional profile fields
    first_name: str | None = Field(
        default=None, min_length=1, max_length=50, description="First name"
    )
    last_name: str | None = Field(
        default=None, min_length=1, max_length=50, description="Last name"
    )
    date_of_birth: datetime | None = Field(default=None, description="Date of birth")
    bio: str | None = Field(default=None, max_length=500, description="User biography")

    # User preferences
    preferences: UserPreferences = Field(
        default_factory=UserPreferences, description="User preferences and settings"
    )

    # Account status
    profile_complete: bool = Field(
        default=False, description="Whether profile setup is complete"
    )
    terms_accepted: bool = Field(
        default=False, description="Terms of service acceptance"
    )
    privacy_policy_accepted: bool = Field(
        default=False, description="Privacy policy acceptance"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Account creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last profile update timestamp"
    )
    last_login_at: datetime | None = Field(
        default=None, description="Last login timestamp"
    )

    @field_validator("photo_url")
    @classmethod
    def validate_photo_url(cls, v: str | None) -> str | None:
        """Validate photo URL format."""
        if v is None:
            return v

        if not v.startswith(("http://", "https://")):
            msg = "Photo URL must start with http:// or https://"
            raise ValueError(msg)

        return v

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, v: datetime | None) -> datetime | None:
        """Validate date of birth is reasonable."""
        if v is None:
            return v

        now = datetime.utcnow()
        age = (now - v).days // 365

        if age < 13:
            msg = "User must be at least 13 years old"
            raise ValueError(msg)
        if age > 120:
            msg = "Invalid date of birth"
            raise ValueError(msg)

        return v


class AuthTokenClaims(BaseModel):
    """Firebase ID token claims model."""

    model_config = ConfigDict(extra="allow")

    # Standard JWT claims
    iss: str = Field(description="Token issuer")
    aud: str = Field(description="Token audience")
    auth_time: int = Field(description="Authentication time")
    user_id: str = Field(description="User ID", alias="uid")
    sub: str = Field(description="Subject (user ID)")
    iat: int = Field(description="Issued at time")
    exp: int = Field(description="Expiration time")

    # Firebase-specific claims
    email: str | None = Field(default=None, description="User email")
    email_verified: bool | None = Field(default=None, description="Email verified")
    name: str | None = Field(default=None, description="Display name")
    picture: str | None = Field(default=None, description="Profile picture URL")

    # Custom claims
    custom_claims: dict[str, Any] | None = Field(
        default_factory=dict, description="Custom user claims"
    )


class AuthToken(BaseModel):
    """Authentication token response model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    access_token: str = Field(description="Firebase ID token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int | None = Field(
        default=None, description="Token expiration time in seconds"
    )
    refresh_token: str | None = Field(
        default=None, description="Refresh token (if available)"
    )
    scope: str | None = Field(default=None, description="Token scope")


class LoginRequest(BaseModel):
    """Login request model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    id_token: str = Field(min_length=1, description="Firebase ID token from client")


class LoginResponse(BaseModel):
    """Login response model."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(default=True, description="Login success status")
    message: str = Field(default="Login successful", description="Response message")
    user: UserProfile = Field(description="User profile data")
    token: AuthToken | None = Field(
        default=None, description="Authentication token (if needed)"
    )


class RefreshTokenRequest(BaseModel):
    """Token refresh request model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    refresh_token: str = Field(min_length=1, description="Refresh token")


class LogoutRequest(BaseModel):
    """Logout request model."""

    model_config = ConfigDict(extra="forbid")

    revoke_all_tokens: bool = Field(default=False, description="Revoke all user tokens")


class UpdateProfileRequest(BaseModel):
    """Profile update request model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    display_name: str | None = Field(
        default=None, min_length=1, max_length=100, description="Display name"
    )
    first_name: str | None = Field(
        default=None, min_length=1, max_length=50, description="First name"
    )
    last_name: str | None = Field(
        default=None, min_length=1, max_length=50, description="Last name"
    )
    bio: str | None = Field(default=None, max_length=500, description="User biography")
    date_of_birth: datetime | None = Field(default=None, description="Date of birth")


class UpdatePreferencesRequest(BaseModel):
    """Preferences update request model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    preferences: UserPreferences = Field(description="Updated user preferences")


class ChangePasswordRequest(BaseModel):
    """Password change request model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    current_password: str = Field(min_length=8, description="Current password")
    new_password: str = Field(min_length=8, description="New password")
    confirm_password: str = Field(min_length=8, description="Password confirmation")

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, values):
        """Validate password confirmation matches."""
        if "new_password" in values.data and v != values.data["new_password"]:
            msg = "Passwords do not match"
            raise ValueError(msg)
        return v


class DeleteAccountRequest(BaseModel):
    """Account deletion request model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    confirm_deletion: bool = Field(description="Confirmation of account deletion")
    reason: str | None = Field(
        default=None, max_length=500, description="Reason for account deletion"
    )

    @field_validator("confirm_deletion")
    @classmethod
    def validate_confirmation(cls, v: bool) -> bool:
        """Ensure deletion is explicitly confirmed."""
        if not v:
            msg = "Account deletion must be explicitly confirmed"
            raise ValueError(msg)
        return v


class UserStatsResponse(BaseModel):
    """User statistics response model."""

    model_config = ConfigDict(extra="forbid")

    trips_created: int = Field(default=0, description="Number of trips created")
    trips_completed: int = Field(default=0, description="Number of completed trips")
    favorite_destinations: list[str] = Field(
        default_factory=list, description="User's favorite destinations"
    )
    total_distance_traveled: float | None = Field(
        default=None, description="Total distance traveled in kilometers"
    )
    account_age_days: int = Field(description="Account age in days")
    last_activity: datetime | None = Field(
        default=None, description="Last activity timestamp"
    )


class ApiResponse(BaseModel):
    """Generic API response model."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(description="Operation success status")
    message: str = Field(description="Response message")
    data: dict | None = Field(default=None, description="Response data")
    errors: list[str] | None = Field(default=None, description="List of error messages")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
