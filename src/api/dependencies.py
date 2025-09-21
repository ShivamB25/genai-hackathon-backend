"""FastAPI API Dependencies for AI-Powered Trip Planner Backend.

This module provides FastAPI dependencies for service injection, database connections,
pagination, filtering, and request validation using existing Firebase auth,
Firestore client, Maps services, and AI services.
"""

from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from src.ai_services.agent_orchestrator import (
    TripPlannerOrchestrator,
    get_trip_planner_orchestrator,
)
from src.ai_services.session_manager import SessionManager, get_session_manager
from src.auth.dependencies import get_current_user, get_user_id
from src.core.logging import get_logger
from src.database.firestore_client import FirestoreClient, get_firestore_client
from src.maps_services.directions_service import get_directions_service
from src.maps_services.geocoding_service import get_geocoding_service
from src.maps_services.places_service import get_places_service

logger = get_logger(__name__)


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters for API endpoints."""

    page: int = Field(default=1, ge=1, le=1000, description="Page number")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page")
    offset: Optional[int] = Field(
        default=None, description="Offset for cursor pagination"
    )

    @field_validator("offset")
    @classmethod
    def calculate_offset(cls, v, info):
        """Calculate offset from page if not provided."""
        if v is None and "page" in info.data and "limit" in info.data:
            return (info.data["page"] - 1) * info.data["limit"]
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database queries."""
        return {
            "page": self.page,
            "limit": self.limit,
            "offset": self.offset or (self.page - 1) * self.limit,
        }


class SortParams(BaseModel):
    """Sorting parameters for API endpoints."""

    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_order: str = Field(
        default="desc", pattern="^(asc|desc)$", description="Sort order"
    )

    def to_firestore_order(self) -> List[tuple]:
        """Convert to Firestore order_by format."""
        return [(self.sort_by, self.sort_order)]


class FilterParams(BaseModel):
    """Common filtering parameters for API endpoints."""

    search: Optional[str] = Field(None, description="Search query")
    status: Optional[str] = Field(None, description="Status filter")
    created_after: Optional[str] = Field(None, description="Created after date (ISO)")
    created_before: Optional[str] = Field(None, description="Created before date (ISO)")

    def to_firestore_filters(self) -> List[tuple]:
        """Convert to Firestore where filters format."""
        filters = []

        if self.status:
            filters.append(("status", "==", self.status))
        if self.created_after:
            filters.append(("created_at", ">=", self.created_after))
        if self.created_before:
            filters.append(("created_at", "<=", self.created_before))

        return filters


# Pagination Dependencies
async def get_pagination_params(
    page: int = Query(1, ge=1, le=1000, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: Optional[int] = Query(None, description="Offset for cursor pagination"),
) -> PaginationParams:
    """Get pagination parameters from query parameters."""
    return PaginationParams(page=page, limit=limit, offset=offset)


async def get_sort_params(
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query(
        "desc", pattern="^(asc|desc)$", description="Sort order (asc/desc)"
    ),
) -> SortParams:
    """Get sorting parameters from query parameters."""
    return SortParams(sort_by=sort_by, sort_order=sort_order)


async def get_filter_params(
    search: Optional[str] = Query(None, description="Search query"),
    status: Optional[str] = Query(None, description="Status filter"),
    created_after: Optional[str] = Query(
        None, description="Created after date (ISO format)"
    ),
    created_before: Optional[str] = Query(
        None, description="Created before date (ISO format)"
    ),
) -> FilterParams:
    """Get filtering parameters from query parameters."""
    return FilterParams(
        search=search,
        status=status,
        created_after=created_after,
        created_before=created_before,
    )


# Database Dependencies
async def get_database_client() -> FirestoreClient:
    """Get Firestore database client."""
    try:
        client = get_firestore_client()
        # Perform health check
        if not await client.health_check():
            logger.error("Firestore health check failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service unavailable",
            )
        return client
    except Exception as e:
        logger.error("Failed to get database client", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed",
        ) from e


# Service Dependencies
async def get_places_api_service():
    """Get Places API service."""
    try:
        return get_places_service()
    except Exception as e:
        logger.error("Failed to get Places service", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Places service unavailable",
        ) from e


async def get_directions_api_service():
    """Get Directions API service."""
    try:
        return get_directions_service()
    except Exception as e:
        logger.error("Failed to get Directions service", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Directions service unavailable",
        ) from e


async def get_geocoding_api_service():
    """Get Geocoding API service."""
    try:
        return get_geocoding_service()
    except Exception as e:
        logger.error("Failed to get Geocoding service", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Geocoding service unavailable",
        ) from e


async def get_ai_session_manager() -> SessionManager:
    """Get AI session manager."""
    try:
        return get_session_manager()
    except Exception as e:
        logger.error("Failed to get AI session manager", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI session service unavailable",
        ) from e


async def get_trip_orchestrator(
    user_id: str = Depends(get_user_id),
    session_manager: SessionManager = Depends(get_ai_session_manager),
) -> TripPlannerOrchestrator:
    """Get trip planner orchestrator for user session."""
    try:
        # Create or get user session
        session_id = f"user_{user_id}_trip_session"

        # Check if session exists
        try:
            await session_manager.get_session(session_id)
        except Exception:
            # Create new session if not found
            await session_manager.create_session(
                session_id=session_id,
                user_id=user_id,
                initial_context={"service": "trip_planning"},
            )

        # Get orchestrator for session
        orchestrator = get_trip_planner_orchestrator(session_id)

        logger.debug(
            "Trip orchestrator initialized",
            user_id=user_id,
            session_id=session_id,
        )

        return orchestrator
    except Exception as e:
        logger.error(
            "Failed to get trip orchestrator",
            user_id=user_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trip planning service unavailable",
        ) from e


# User Context Dependencies
async def get_user_context(
    user: Dict[str, Any] = Depends(get_current_user),
    db: FirestoreClient = Depends(get_database_client),
) -> Dict[str, Any]:
    """Get comprehensive user context including profile and preferences."""
    try:
        user_id = user.get("uid")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )

        # Get user profile from Firestore
        user_profile = await db.get_document("users", user_id) or {}

        # Combine auth user data with profile
        user_context = {
            "user_id": user_id,
            "auth_user": user.get("auth_user", {}),
            "profile": user_profile,
            "preferences": user_profile.get("travel_preferences", {}),
            "firebase_claims": user.get("token_claims", {}),
        }

        return user_context
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get user context",
            user_id=user.get("uid"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user context",
        ) from e


# Trip-specific Dependencies
class TripFilterParams(BaseModel):
    """Trip-specific filtering parameters."""

    destination: Optional[str] = Field(None, description="Destination filter")
    trip_type: Optional[str] = Field(None, description="Trip type filter")
    status: Optional[str] = Field(None, description="Trip status filter")
    start_date_after: Optional[str] = Field(
        None, description="Start date after (ISO format)"
    )
    start_date_before: Optional[str] = Field(
        None, description="Start date before (ISO format)"
    )
    budget_min: Optional[float] = Field(None, ge=0, description="Minimum budget")
    budget_max: Optional[float] = Field(None, ge=0, description="Maximum budget")

    def to_firestore_filters(self) -> List[tuple]:
        """Convert to Firestore where filters format."""
        filters = []

        if self.destination:
            filters.append(("destination", "==", self.destination))
        if self.trip_type:
            filters.append(("trip_type", "==", self.trip_type))
        if self.status:
            filters.append(("status", "==", self.status))
        if self.start_date_after:
            filters.append(("start_date", ">=", self.start_date_after))
        if self.start_date_before:
            filters.append(("start_date", "<=", self.start_date_before))
        if self.budget_min:
            filters.append(("overall_budget.total_budget", ">=", str(self.budget_min)))
        if self.budget_max:
            filters.append(("overall_budget.total_budget", "<=", str(self.budget_max)))

        return filters


async def get_trip_filter_params(
    destination: Optional[str] = Query(None, description="Destination filter"),
    trip_type: Optional[str] = Query(None, description="Trip type filter"),
    status: Optional[str] = Query(None, description="Trip status filter"),
    start_date_after: Optional[str] = Query(
        None, description="Start date after (ISO format)"
    ),
    start_date_before: Optional[str] = Query(
        None, description="Start date before (ISO format)"
    ),
    budget_min: Optional[float] = Query(None, ge=0, description="Minimum budget"),
    budget_max: Optional[float] = Query(None, ge=0, description="Maximum budget"),
) -> TripFilterParams:
    """Get trip-specific filtering parameters from query parameters."""
    return TripFilterParams(
        destination=destination,
        trip_type=trip_type,
        status=status,
        start_date_after=start_date_after,
        start_date_before=start_date_before,
        budget_min=budget_min,
        budget_max=budget_max,
    )


# Request Validation Dependencies
async def validate_trip_access(
    trip_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context),
    db: FirestoreClient = Depends(get_database_client),
) -> Dict[str, Any]:
    """Validate user has access to trip and return trip data."""
    try:
        # Get trip from database
        trip_data = await db.get_document("trips", trip_id)
        if not trip_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trip not found: {trip_id}",
            )

        # Check ownership or sharing permissions
        user_id = user_context["user_id"]
        trip_user_id = trip_data.get("user_id")

        if trip_user_id != user_id:
            # Check if trip is shared with user
            shared_with = trip_data.get("shared_with", [])
            if user_id not in shared_with:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this trip",
                )

        return trip_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to validate trip access",
            trip_id=trip_id,
            user_id=user_context.get("user_id"),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate trip access",
        ) from e


# Health Check Dependencies
async def check_service_health() -> Dict[str, Any]:
    """Check health of all dependent services."""
    health_status = {
        "status": "healthy",
        "services": {},
        "timestamp": "2024-01-01T00:00:00Z",  # Will be updated by calling code
    }

    try:
        # Check database
        try:
            await get_database_client()
            health_status["services"]["database"] = "healthy"
        except Exception as e:
            health_status["services"]["database"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"

        # Check AI services
        try:
            await get_ai_session_manager()
            health_status["services"]["ai_session"] = "healthy"
        except Exception as e:
            health_status["services"]["ai_session"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"

        # Check Maps services
        try:
            await get_places_api_service()
            health_status["services"]["places_api"] = "healthy"
        except Exception as e:
            health_status["services"]["places_api"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"

    except Exception as e:
        logger.error("Health check failed", error=str(e), exc_info=True)
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)

    return health_status


# Commonly used dependency combinations for convenience
DatabaseClient = Depends(get_database_client)
PlacesService = Depends(get_places_api_service)
DirectionsService = Depends(get_directions_api_service)
GeocodingService = Depends(get_geocoding_api_service)
AISessionManager = Depends(get_ai_session_manager)
TripOrchestrator = Depends(get_trip_orchestrator)
UserContext = Depends(get_user_context)
PaginationParams = Depends(get_pagination_params)
SortParams = Depends(get_sort_params)
FilterParams = Depends(get_filter_params)
TripFilterParams = Depends(get_trip_filter_params)
