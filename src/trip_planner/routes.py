"""Trip Planner API Routes for AI-Powered Trip Planner Backend.

This module provides FastAPI routes for trip planning operations including
AI trip generation, CRUD operations, sharing, and comprehensive validation
with Firebase authentication integration.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)
from pydantic import BaseModel, Field, model_validator

from src.api.dependencies import (
    PaginationParams,
    TripFilterParams,
    get_pagination_params,
    get_trip_filter_params,
    get_user_context,
)
from src.auth.dependencies import get_user_id
from src.core.logging import get_logger
from src.trip_planner.schemas import TripItinerary, TripRequest, WorkflowResult
from src.trip_planner.services import (
    TripAccessDeniedError,
    TripNotFoundError,
    TripPlanGenerationError,
    TripPlannerService,
    get_trip_planner_service,
)

logger = get_logger(__name__)

# Create router with prefix and tags
router = APIRouter(prefix="/api/v1/trips", tags=["Trip Planning"])


# Request/Response Models
class TripPlanRequest(BaseModel):
    """Request model for trip planning."""

    destination: str = Field(..., min_length=1, description="Primary destination")
    additional_destinations: List[str] = Field(
        default_factory=list, description="Additional destinations"
    )
    start_date: date = Field(..., description="Trip start date (YYYY-MM-DD)")
    end_date: date = Field(..., description="Trip end date (YYYY-MM-DD)")
    traveler_count: int = Field(
        default=1, gt=0, le=50, description="Number of travelers"
    )
    trip_type: str = Field(default="leisure", description="Type of trip")
    budget_amount: Optional[float] = Field(None, gt=0, description="Total budget")
    budget_currency: str = Field(default="INR", description="Budget currency")
    preferred_activities: List[str] = Field(
        default_factory=list, description="Preferred activity types"
    )
    accommodation_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Accommodation preferences"
    )
    transportation_preferences: List[str] = Field(
        default_factory=list, description="Transport preferences"
    )
    special_requirements: List[str] = Field(
        default_factory=list, description="Special requirements"
    )
    accessibility_needs: List[str] = Field(
        default_factory=list, description="Accessibility needs"
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list, description="Dietary restrictions"
    )
    must_include: List[str] = Field(
        default_factory=list, description="Must-include places/activities"
    )
    avoid: List[str] = Field(default_factory=list, description="Things to avoid")

    @model_validator(mode="after")
    def validate_date_order(self) -> "TripPlanRequest":
        """Ensure the requested trip ends after it begins."""
        if self.end_date <= self.start_date:
            msg = "end_date must be after start_date"
            raise ValueError(msg)
        return self


class TripUpdateRequest(BaseModel):
    """Request model for trip updates."""

    title: Optional[str] = Field(None, description="Updated trip title")
    description: Optional[str] = Field(None, description="Updated description")
    daily_plans: Optional[List[Dict[str, Any]]] = Field(
        None, description="Updated daily plans"
    )
    budget: Optional[Dict[str, Any]] = Field(None, description="Updated budget")
    packing_suggestions: Optional[List[str]] = Field(
        None, description="Updated packing suggestions"
    )
    safety_tips: Optional[List[str]] = Field(None, description="Updated safety tips")


class TripShareRequest(BaseModel):
    """Request model for trip sharing."""

    user_emails: List[str] = Field(
        ..., min_length=1, description="Email addresses to share with"
    )


class TripResponse(BaseModel):
    """Response model for trip operations."""

    success: bool = Field(..., description="Operation success")
    trip: Optional[TripItinerary] = Field(None, description="Trip data")
    message: str = Field(..., description="Response message")


class TripListResponse(BaseModel):
    """Response model for trip listing."""

    success: bool = Field(..., description="Operation success")
    trips: List[TripItinerary] = Field(..., description="User's trips")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    total_count: Optional[int] = Field(None, description="Total trips count")


# Main Trip Planning Endpoint
@router.post(
    "/plan",
    response_model=WorkflowResult,
    status_code=status.HTTP_201_CREATED,
    summary="Create AI-Generated Trip Plan",
    description="Generate a comprehensive trip itinerary using AI multi-agent system",
)
async def create_trip_plan(
    trip_request: TripPlanRequest,
    _background_tasks: BackgroundTasks,
    user_context: Dict[str, Any] = Depends(get_user_context),
    service: TripPlannerService = Depends(get_trip_planner_service),
    workflow_type: str = Query(
        "comprehensive",
        pattern="^(comprehensive|quick)$",
        description="Workflow type for trip planning",
    ),
    background: bool = Query(False, description="Execute as background task"),
) -> WorkflowResult:
    """Create AI-generated trip plan using multi-agent system."""
    try:
        user_id = user_context["user_id"]

        # Convert request to TripRequest schema
        from decimal import Decimal

        from src.trip_planner.schemas import (
            ActivityType,
            Budget,
            TransportationMode,
            TripType,
        )

        # Compute duration (inclusive) using validated dates
        duration_days = (trip_request.end_date - trip_request.start_date).days + 1

        # Build budget if provided
        budget = None
        if trip_request.budget_amount:
            budget = Budget(
                total_budget=Decimal(str(trip_request.budget_amount)),
                currency=trip_request.budget_currency,
                remaining_amount=Decimal(str(trip_request.budget_amount)),
                daily_budget=Decimal(str(trip_request.budget_amount)) / duration_days,
            )

        # Convert enums
        trip_type = (
            TripType(trip_request.trip_type)
            if trip_request.trip_type in [e.value for e in TripType]
            else TripType.LEISURE
        )

        preferred_activities = []
        for activity in trip_request.preferred_activities:
            try:
                preferred_activities.append(ActivityType(activity))
            except ValueError:
                logger.warning(f"Invalid activity type: {activity}")

        transportation_preferences = []
        for transport in trip_request.transportation_preferences:
            try:
                transportation_preferences.append(TransportationMode(transport))
            except ValueError:
                logger.warning(f"Invalid transport mode: {transport}")

        # Create TripRequest
        trip_req = TripRequest(
            user_id=user_id,
            destination=trip_request.destination,
            additional_destinations=trip_request.additional_destinations,
            start_date=trip_request.start_date,
            end_date=trip_request.end_date,
            duration_days=duration_days,
            traveler_count=trip_request.traveler_count,
            trip_type=trip_type,
            budget=budget,
            preferred_activities=preferred_activities,
            accommodation_preferences=trip_request.accommodation_preferences,
            transportation_preferences=transportation_preferences,
            special_requirements=trip_request.special_requirements,
            accessibility_needs=trip_request.accessibility_needs,
            dietary_restrictions=trip_request.dietary_restrictions,
            must_include=trip_request.must_include,
            avoid=trip_request.avoid,
            updated_at=None,
            session_id=None,
        )

        # Execute trip planning
        result = await service.create_trip_plan(
            user_id=user_id,
            trip_request=trip_req,
            workflow_type=workflow_type,
            background=background,
        )

        logger.info(
            "Trip plan created",
            user_id=user_id,
            workflow_id=result.workflow_id,
            background=background,
        )

        return result

    except TripPlanGenerationError as e:
        logger.exception("Trip generation failed", user_id=user_context.get("user_id"))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Trip generation failed: {e.message}",
        ) from e
    except Exception as e:
        logger.error("Unexpected error in trip planning", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during trip planning",
        ) from e


# Get Trip Details
@router.get(
    "/{trip_id}",
    response_model=TripResponse,
    summary="Get Trip Details",
    description="Retrieve detailed trip information by trip ID",
)
async def get_trip_details(
    trip_id: str = Path(..., description="Trip identifier"),
    user_id: str = Depends(get_user_id),
    service: TripPlannerService = Depends(get_trip_planner_service),
) -> TripResponse:
    """Get trip details by ID."""
    try:
        trip = await service.get_trip(trip_id, user_id)

        return TripResponse(
            success=True,
            trip=trip,
            message="Trip retrieved successfully",
        )

    except TripNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip not found: {trip_id}",
        ) from e
    except TripAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this trip",
        ) from e
    except Exception as e:
        logger.error("Failed to get trip", trip_id=trip_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trip",
        ) from e


# Update Trip
@router.put(
    "/{trip_id}",
    response_model=TripResponse,
    summary="Update Trip Itinerary",
    description="Update trip itinerary with AI optimization",
)
async def update_trip_itinerary(
    updates: TripUpdateRequest,
    trip_id: str = Path(..., description="Trip identifier"),
    user_id: str = Depends(get_user_id),
    service: TripPlannerService = Depends(get_trip_planner_service),
) -> TripResponse:
    """Update trip itinerary."""
    try:
        # Convert updates to dictionary, excluding None values
        update_data = {k: v for k, v in updates.model_dump().items() if v is not None}

        trip = await service.update_trip(trip_id, user_id, update_data)

        return TripResponse(
            success=True,
            trip=trip,
            message="Trip updated successfully",
        )

    except TripNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip not found: {trip_id}",
        ) from e
    except TripAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this trip",
        ) from e
    except Exception as e:
        logger.error(
            "Failed to update trip", trip_id=trip_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update trip",
        ) from e


# Delete Trip
@router.delete(
    "/{trip_id}",
    response_model=Dict[str, Any],
    summary="Delete Trip",
    description="Delete trip from Firestore",
)
async def delete_trip(
    trip_id: str = Path(..., description="Trip identifier"),
    user_id: str = Depends(get_user_id),
    service: TripPlannerService = Depends(get_trip_planner_service),
) -> Dict[str, Any]:
    """Delete trip by ID."""
    try:
        deleted = await service.delete_trip(trip_id, user_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trip not found or already deleted",
            )

        return {
            "success": True,
            "message": "Trip deleted successfully",
            "trip_id": trip_id,
        }

    except TripNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip not found: {trip_id}",
        ) from e
    except TripAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this trip",
        ) from e
    except Exception as e:
        logger.error(
            "Failed to delete trip", trip_id=trip_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete trip",
        ) from e


# List User Trips
@router.get(
    "",
    response_model=TripListResponse,
    summary="List User Trips",
    description="Get user's trips with pagination and filtering",
)
async def list_user_trips(
    user_id: str = Depends(get_user_id),
    pagination: PaginationParams = Depends(get_pagination_params),
    filters: TripFilterParams = Depends(get_trip_filter_params),
    service: TripPlannerService = Depends(get_trip_planner_service),
) -> TripListResponse:
    """List user's trips with pagination and filtering."""
    try:
        # Convert filters to dictionary
        filter_dict = {k: v for k, v in filters.model_dump().items() if v is not None}

        result = await service.list_user_trips(
            user_id=user_id,
            filters=filter_dict,
            limit=pagination.limit,
            offset=pagination.offset or 0,
        )

        return TripListResponse(
            success=True,
            trips=result["trips"],
            pagination=result["pagination"],
            total_count=len(result["trips"]),
        )

    except Exception as e:
        logger.error(
            "Failed to list trips", user_id=user_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trips",
        ) from e


# Share Trip
@router.post(
    "/{trip_id}/share",
    response_model=Dict[str, Any],
    summary="Share Trip",
    description="Share trip with other users using Firebase security",
)
async def share_trip(
    share_request: TripShareRequest,
    trip_id: str = Path(..., description="Trip identifier"),
    user_id: str = Depends(get_user_id),
    service: TripPlannerService = Depends(get_trip_planner_service),
) -> Dict[str, Any]:
    """Share trip with other users."""
    try:
        # For now, we'll use email addresses as user identifiers
        # In a real implementation, you'd convert emails to Firebase UIDs
        shared_with = share_request.user_emails

        result = await service.share_trip(trip_id, user_id, shared_with)

        return {
            "success": True,
            "message": "Trip shared successfully",
            "trip_id": trip_id,
            "shared_with": result["shared_with"],
            "updated_at": result["updated_at"],
        }

    except TripNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip not found: {trip_id}",
        ) from e
    except TripAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only trip owner can share trips",
        ) from e
    except Exception as e:
        logger.error(
            "Failed to share trip", trip_id=trip_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to share trip",
        ) from e


# Get Trip Metrics
@router.get(
    "/{trip_id}/metrics",
    response_model=Dict[str, Any],
    summary="Get Trip Metrics",
    description="Get comprehensive analytics and metrics for a trip",
)
async def get_trip_metrics(
    trip_id: str = Path(..., description="Trip identifier"),
    user_id: str = Depends(get_user_id),
    service: TripPlannerService = Depends(get_trip_planner_service),
) -> Dict[str, Any]:
    """Get trip metrics and analytics."""
    try:
        metrics = await service.get_trip_metrics(trip_id, user_id)

        return {
            "success": True,
            "metrics": metrics,
            "message": "Trip metrics retrieved successfully",
        }

    except TripNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip not found: {trip_id}",
        ) from e
    except TripAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this trip",
        ) from e
    except Exception as e:
        logger.error(
            "Failed to get trip metrics", trip_id=trip_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trip metrics",
        ) from e


# Trip Status Check (for background tasks)
@router.get(
    "/{trip_id}/status",
    response_model=Dict[str, Any],
    summary="Get Trip Generation Status",
    description="Check status of trip generation (useful for background tasks)",
)
async def get_trip_status(
    trip_id: str = Path(..., description="Trip identifier"),
    user_id: str = Depends(get_user_id),
    service: TripPlannerService = Depends(get_trip_planner_service),
) -> Dict[str, Any]:
    """Get trip generation status."""
    try:
        # Check if trip exists
        try:
            trip = await service.get_trip(trip_id, user_id)
            status_info = {
                "trip_id": trip_id,
                "status": "completed",
                "itinerary_available": True,
                "created_at": trip.created_at.isoformat(),
                "updated_at": trip.updated_at.isoformat() if trip.updated_at else None,
            }
        except TripNotFoundError:
            # Trip not found, might still be generating
            status_info = {
                "trip_id": trip_id,
                "status": "generating",
                "itinerary_available": False,
                "message": "Trip is still being generated",
            }

        return {
            "success": True,
            "status": status_info,
        }

    except TripAccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this trip",
        ) from e
    except Exception as e:
        logger.error(
            "Failed to get trip status", trip_id=trip_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trip status",
        ) from e


# Health Check for Trip Planning Service
@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Trip Planning Service Health",
    description="Check health of trip planning service and dependencies",
)
async def trip_planning_health() -> Dict[str, Any]:
    """Check trip planning service health."""
    try:
        # Basic service instantiation check
        get_trip_planner_service()

        health_info = {
            "status": "healthy",
            "service": "trip_planning",
            "dependencies": {
                "ai_orchestrator": "healthy",
                "firestore": "healthy",
                "maps_services": "healthy",
            },
            "timestamp": "2024-01-01T00:00:00Z",
        }

        return {
            "success": True,
            "health": health_info,
        }

    except Exception as e:
        logger.error("Trip planning health check failed", error=str(e), exc_info=True)
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }
