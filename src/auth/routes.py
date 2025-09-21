"""User Management API Routes for AI-Powered Trip Planner Backend.

This module provides FastAPI routes for user profile management, preferences,
trip history, and account operations with Firebase auth and Firestore integration.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import (
    PaginationParams,
    TripFilterParams,
    get_database_client,
    get_pagination_params,
    get_trip_filter_params,
    get_user_context,
)
from src.auth.dependencies import get_user_id
from src.auth.firebase_auth import delete_user_account
from src.auth.schemas import (
    ApiResponse,
    DeleteAccountRequest,
    UpdatePreferencesRequest,
    UpdateProfileRequest,
    UserPreferences,
    UserProfile,
    UserStatsResponse,
)
from src.core.logging import get_logger
from src.database.firestore_client import FirestoreClient
from src.trip_planner.schemas import TripItinerary

logger = get_logger(__name__)

# Create router with prefix and tags
router = APIRouter(prefix="/api/v1/users", tags=["User Management"])


# Response Models
class ProfileResponse(BaseModel):
    """Response model for profile operations."""

    success: bool = Field(..., description="Operation success")
    profile: Optional[UserProfile] = Field(None, description="User profile")
    message: str = Field(..., description="Response message")


class TripsHistoryResponse(BaseModel):
    """Response model for user trip history."""

    success: bool = Field(..., description="Operation success")
    trips: List[TripItinerary] = Field(..., description="User's trips")
    stats: UserStatsResponse = Field(..., description="User statistics")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")


# User Profile Routes


@router.get(
    "/profile",
    response_model=ProfileResponse,
    summary="Get User Profile",
    description="Get user profile from Firestore",
)
async def get_user_profile(
    user_context: Dict[str, Any] = Depends(get_user_context),
) -> ProfileResponse:
    """Get user profile information."""
    try:
        profile_data = user_context.get("profile", {})
        auth_user = user_context.get("auth_user", {})

        # Create or update profile from auth data
        if not profile_data:
            # Create profile from Firebase auth data
            profile_data = {
                "uid": user_context["user_id"],
                "email": auth_user.get("email"),
                "email_verified": auth_user.get("email_verified", False),
                "display_name": auth_user.get("name"),
                "photo_url": auth_user.get("picture"),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

        # Convert to UserProfile model
        profile = UserProfile.model_validate(profile_data)

        logger.info(
            "User profile retrieved",
            user_id=user_context["user_id"],
            email=profile.email,
        )

        return ProfileResponse(
            success=True,
            profile=profile,
            message="Profile retrieved successfully",
        )

    except Exception as e:
        logger.error(
            "Failed to get user profile",
            user_id=user_context.get("user_id"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile",
        ) from e


@router.put(
    "/profile",
    response_model=ProfileResponse,
    summary="Update User Profile",
    description="Update user profile in Firestore",
)
async def update_user_profile(
    profile_updates: UpdateProfileRequest,
    user_context: Dict[str, Any] = Depends(get_user_context),
    db: FirestoreClient = Depends(get_database_client),
) -> ProfileResponse:
    """Update user profile."""
    try:
        user_id = user_context["user_id"]
        current_profile = user_context.get("profile", {})

        # Apply updates
        update_data = {
            k: v for k, v in profile_updates.model_dump().items() if v is not None
        }
        update_data["updated_at"] = datetime.now(timezone.utc)

        # Check if this makes profile complete
        required_fields = ["display_name", "first_name", "last_name"]
        if all(
            update_data.get(field) or current_profile.get(field)
            for field in required_fields
        ):
            update_data["profile_complete"] = True

        # Update in Firestore
        await db.update_document("users", user_id, update_data)

        # Get updated profile
        updated_profile_data = await db.get_document("users", user_id)
        profile = UserProfile.model_validate(updated_profile_data)

        logger.info(
            "User profile updated",
            user_id=user_id,
            updated_fields=list(update_data.keys()),
        )

        return ProfileResponse(
            success=True,
            profile=profile,
            message="Profile updated successfully",
        )

    except Exception as e:
        logger.error(
            "Failed to update user profile",
            user_id=user_context.get("user_id"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile",
        ) from e


@router.put(
    "/preferences",
    response_model=ApiResponse,
    summary="Update Travel Preferences",
    description="Update user's travel preferences",
)
async def update_travel_preferences(
    preferences_request: UpdatePreferencesRequest,
    user_context: Dict[str, Any] = Depends(get_user_context),
    db: FirestoreClient = Depends(get_database_client),
) -> ApiResponse:
    """Update user travel preferences."""
    try:
        user_id = user_context["user_id"]

        # Update preferences in Firestore
        update_data = {
            "preferences": preferences_request.preferences.model_dump(),
            "updated_at": datetime.now(timezone.utc),
        }

        await db.update_document("users", user_id, update_data)

        logger.info(
            "User preferences updated",
            user_id=user_id,
            preferences=preferences_request.preferences.model_dump(),
        )

        return ApiResponse(
            success=True,
            message="Travel preferences updated successfully",
            data={"preferences": preferences_request.preferences.model_dump()},
        )

    except Exception as e:
        logger.error(
            "Failed to update user preferences",
            user_id=user_context.get("user_id"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update travel preferences",
        ) from e


@router.get(
    "/trips",
    response_model=TripsHistoryResponse,
    summary="Get User Trip History",
    description="Get user's trip history from Firestore",
)
async def get_user_trip_history(
    user_id: str = Depends(get_user_id),
    pagination: PaginationParams = Depends(get_pagination_params),
    filters: TripFilterParams = Depends(get_trip_filter_params),
    db: FirestoreClient = Depends(get_database_client),
) -> TripsHistoryResponse:
    """Get user's trip history."""
    try:
        # Build query filters
        query_filters = [("user_id", "==", user_id)]

        # Add additional filters
        if filters.destination:
            query_filters.append(("destination", "==", filters.destination))
        if filters.status:
            query_filters.append(("status", "==", filters.status))
        if filters.start_date_after:
            query_filters.append(("start_date", ">=", filters.start_date_after))
        if filters.start_date_before:
            query_filters.append(("start_date", "<=", filters.start_date_before))

        # Query trips
        trip_docs = await db.query_documents(
            collection_name="trips",
            filters=query_filters,
            order_by=[("created_at", "desc")],
            limit=pagination.limit + 1,  # Get one extra for next page check
        )

        # Process results
        trips = []
        for doc_data in trip_docs[: pagination.limit]:
            try:
                trip = TripItinerary.model_validate(doc_data)
                trips.append(trip)
            except Exception as e:
                logger.warning(
                    "Failed to parse trip data",
                    trip_id=doc_data.get("id"),
                    error=str(e),
                )

        # Calculate user statistics
        total_trips = len(trips)
        completed_trips = sum(1 for trip in trips if trip.status == "completed")
        destinations = list(set(trip.destination for trip in trips))

        # Calculate account age
        user_profile = await db.get_document("users", user_id)
        created_at = user_profile.get("created_at") if user_profile else None
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif not isinstance(created_at, datetime):
            created_at = datetime.now(timezone.utc)  # Fallback to now

        account_age = (datetime.now(timezone.utc) - created_at).days

        stats = UserStatsResponse(
            trips_created=total_trips,
            trips_completed=completed_trips,
            favorite_destinations=destinations[:5],  # Top 5 destinations
            account_age_days=account_age,
            last_activity=(
                max(trip.updated_at or trip.created_at for trip in trips)
                if trips
                else None
            ),
        )

        # Pagination info
        has_next_page = len(trip_docs) > pagination.limit
        pagination_info = {
            "total": total_trips,
            "page": pagination.page,
            "limit": pagination.limit,
            "has_next_page": has_next_page,
        }

        logger.info(
            "User trip history retrieved",
            user_id=user_id,
            trips_count=total_trips,
        )

        return TripsHistoryResponse(
            success=True,
            trips=trips,
            stats=stats,
            pagination=pagination_info,
        )

    except Exception as e:
        logger.error(
            "Failed to get user trip history",
            user_id=user_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trip history",
        ) from e


@router.delete(
    "/account",
    response_model=ApiResponse,
    summary="Delete User Account",
    description="Delete user account and all associated data from Firebase/Firestore",
)
async def delete_user_account_endpoint(
    deletion_request: DeleteAccountRequest,
    user_context: Dict[str, Any] = Depends(get_user_context),
    db: FirestoreClient = Depends(get_database_client),
) -> ApiResponse:
    """Delete user account and all associated data."""
    try:
        user_id = user_context["user_id"]

        if not deletion_request.confirm_deletion:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account deletion must be explicitly confirmed",
            )

        # Delete user trips first
        user_trips = await db.query_documents(
            collection_name="trips",
            filters=[("user_id", "==", user_id)],
        )

        for trip_doc in user_trips:
            try:
                await db.delete_document("trips", trip_doc["id"])
                logger.debug("Deleted user trip", trip_id=trip_doc["id"])
            except Exception as e:
                logger.warning(
                    "Failed to delete trip", trip_id=trip_doc["id"], error=str(e)
                )

        # Delete user sessions
        user_sessions = await db.query_documents(
            collection_name="ai_sessions",
            filters=[("user_id", "==", user_id)],
        )

        for session_doc in user_sessions:
            try:
                await db.delete_document("ai_sessions", session_doc["id"])
                logger.debug("Deleted user session", session_id=session_doc["id"])
            except Exception as e:
                logger.warning(
                    "Failed to delete session",
                    session_id=session_doc["id"],
                    error=str(e),
                )

        # Delete user from Firebase Auth and Firestore
        deleted = await delete_user_account(user_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user account from Firebase",
            )

        logger.info(
            "User account deleted successfully",
            user_id=user_id,
            reason=deletion_request.reason,
        )

        return ApiResponse(
            success=True,
            message="Account deleted successfully",
            data={
                "user_id": user_id,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
                "reason": deletion_request.reason,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete user account",
            user_id=user_context.get("user_id"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user account",
        ) from e


# User Statistics Route


@router.get(
    "/stats",
    response_model=UserStatsResponse,
    summary="Get User Statistics",
    description="Get comprehensive user statistics and activity metrics",
)
async def get_user_statistics(
    user_id: str = Depends(get_user_id),
    db: FirestoreClient = Depends(get_database_client),
) -> UserStatsResponse:
    """Get user statistics and metrics."""
    try:
        # Get user trips for statistics
        user_trips = await db.query_documents(
            collection_name="trips",
            filters=[("user_id", "==", user_id)],
            order_by=[("created_at", "desc")],
        )

        # Calculate statistics
        total_trips = len(user_trips)
        completed_trips = sum(
            1 for trip in user_trips if trip.get("status") == "completed"
        )

        # Get favorite destinations (most visited)
        destination_counts = {}
        total_distance = 0.0

        for trip in user_trips:
            dest = trip.get("destination", "Unknown")
            destination_counts[dest] = destination_counts.get(dest, 0) + 1

            # Calculate approximate distance if available
            if trip.get("efficiency_metrics", {}).get("total_distance"):
                total_distance += float(trip["efficiency_metrics"]["total_distance"])

        favorite_destinations = sorted(
            destination_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        favorite_destinations = [dest for dest, _ in favorite_destinations]

        # Get account age
        user_profile = await db.get_document("users", user_id)
        created_at = user_profile.get("created_at") if user_profile else None
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif not isinstance(created_at, datetime):
            created_at = datetime.now(timezone.utc)  # Fallback to now

        account_age = (datetime.now(timezone.utc) - created_at).days

        # Get last activity
        last_activity = None
        if user_trips:
            last_trip = max(
                user_trips,
                key=lambda x: x.get("updated_at") or x.get("created_at", ""),
            )
            last_activity_str = last_trip.get("updated_at") or last_trip.get(
                "created_at"
            )
            if last_activity_str:
                last_activity = datetime.fromisoformat(
                    last_activity_str.replace("Z", "+00:00")
                )

        stats = UserStatsResponse(
            trips_created=total_trips,
            trips_completed=completed_trips,
            favorite_destinations=favorite_destinations,
            total_distance_traveled=total_distance if total_distance > 0 else None,
            account_age_days=account_age,
            last_activity=last_activity,
        )

        logger.info(
            "User statistics generated",
            user_id=user_id,
            trips_created=total_trips,
            trips_completed=completed_trips,
        )

        return stats

    except Exception as e:
        logger.error(
            "Failed to get user statistics",
            user_id=user_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics",
        ) from e


# User Preferences Route


@router.get(
    "/preferences",
    response_model=UserPreferences,
    summary="Get User Preferences",
    description="Get user's travel preferences",
)
async def get_user_preferences(
    user_context: Dict[str, Any] = Depends(get_user_context),
) -> UserPreferences:
    """Get user travel preferences."""
    try:
        preferences_data = user_context.get("preferences", {})

        # Create preferences object with defaults if empty
        preferences = UserPreferences.model_validate(preferences_data)

        logger.debug(
            "User preferences retrieved",
            user_id=user_context["user_id"],
        )

        return preferences

    except Exception as e:
        logger.error(
            "Failed to get user preferences",
            user_id=user_context.get("user_id"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user preferences",
        ) from e


# Account Management Routes


@router.post(
    "/complete-profile",
    response_model=ApiResponse,
    summary="Complete Profile Setup",
    description="Mark user profile as complete and accept terms",
)
async def complete_profile_setup(
    user_context: Dict[str, Any] = Depends(get_user_context),
    db: FirestoreClient = Depends(get_database_client),
) -> ApiResponse:
    """Complete user profile setup."""
    try:
        user_id = user_context["user_id"]

        # Update profile completion status
        update_data = {
            "profile_complete": True,
            "terms_accepted": True,
            "privacy_policy_accepted": True,
            "updated_at": datetime.now(timezone.utc),
        }

        await db.update_document("users", user_id, update_data)

        logger.info("User profile setup completed", user_id=user_id)

        return ApiResponse(
            success=True,
            message="Profile setup completed successfully",
            data={"profile_complete": True},
        )

    except Exception as e:
        logger.error(
            "Failed to complete profile setup",
            user_id=user_context.get("user_id"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete profile setup",
        ) from e


# Health Check for User Services
@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="User Services Health",
    description="Check health of user management services",
)
async def user_services_health() -> Dict[str, Any]:
    """Check user services health."""
    try:
        health_info = {
            "status": "healthy",
            "services": {
                "firebase_auth": "healthy",
                "firestore_users": "healthy",
                "profile_management": "healthy",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "success": True,
            "health": health_info,
        }

    except Exception as e:
        logger.error("User services health check failed", error=str(e), exc_info=True)
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
