"""Trip Planner Service Layer for AI-Powered Trip Planner Backend.

This module provides the main service layer for trip planning operations including
trip generation using multi-agent system, Firestore persistence, Maps API integration,
background task management, and trip sharing functionality.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.ai_services.agent_factory import (
    TripComplexity,
    TripRequirements,
    get_agent_factory,
)
from src.ai_services.agent_orchestrator import (
    TripPlannerOrchestrator,
    WorkflowDefinition,
    create_comprehensive_trip_planning_workflow,
    create_quick_trip_planning_workflow,
    get_trip_planner_orchestrator,
)
from src.ai_services.gemini_agents import AgentRole, BaseAgent
from src.ai_services.session_manager import get_session_manager
from src.core.logging import get_logger
from src.database.firestore_client import get_firestore_client
from src.maps_services.directions_service import get_directions_service
from src.maps_services.geocoding_service import get_geocoding_service
from src.maps_services.places_service import get_places_service
from src.maps_services.schemas import GeoLocation
from src.trip_planner.schemas import (
    Budget,
    DayPlan,
    TripItinerary,
    TripRequest,
    TripType,
    WorkflowResult,
    calculate_itinerary_metrics,
    validate_itinerary_completeness,
)

logger = get_logger(__name__)


class TripPlannerServiceError(Exception):
    """Base exception for trip planner service errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class TripNotFoundError(TripPlannerServiceError):
    """Raised when trip is not found."""


class TripAccessDeniedError(TripPlannerServiceError):
    """Raised when user doesn't have access to trip."""


class TripPlanGenerationError(TripPlannerServiceError):
    """Raised when trip plan generation fails."""


class TripPlannerService:
    """Main service for AI-powered trip planning operations."""

    def __init__(self) -> None:
        """Initialize trip planner service."""
        self._firestore_client = get_firestore_client()
        self._session_manager = get_session_manager()
        self._places_service = get_places_service()
        self._directions_service = get_directions_service()
        self._geocoding_service = get_geocoding_service()

        # Collections
        self._trips_collection = "trips"
        self._sessions_collection = "trip_sessions"
        self._users_collection = "users"

    async def create_trip_plan(
        self,
        user_id: str,
        trip_request: TripRequest,
        workflow_type: str = "comprehensive",
        background: bool = False,
    ) -> WorkflowResult:
        """Create a comprehensive trip plan using AI multi-agent system.

        Args:
            user_id: User identifier
            trip_request: Trip planning request
            workflow_type: Type of workflow (comprehensive/quick)
            background: Whether to run as background task

        Returns:
            WorkflowResult: Trip planning workflow result

        Raises:
            TripPlanGenerationError: If trip generation fails
        """
        try:
            logger.info(
                "Starting trip plan creation",
                user_id=user_id,
                destination=trip_request.destination,
                workflow_type=workflow_type,
                background=background,
            )

            # Update trip request with user context
            trip_request.user_id = user_id
            if not trip_request.request_id:
                trip_request.request_id = str(uuid4())

            # Get or create orchestrator
            session_id = f"user_{user_id}_trip_session"
            orchestrator = get_trip_planner_orchestrator(session_id)

            # Build agent requirements and provision agents
            trip_requirements = self._build_trip_requirements(trip_request)
            agent_factory = get_agent_factory()
            agent_team = agent_factory.create_agent_team_for_trip(
                trip_requirements, session_id
            )

            # Ensure every agent is initialised before registration
            for agent in agent_team.values():
                await agent.initialize()
                await orchestrator.register_agent(agent)

            # Select workflow
            if workflow_type == "quick":
                workflow_def = create_quick_trip_planning_workflow()
            else:
                workflow_def = create_comprehensive_trip_planning_workflow()

            # Prepare context for AI agents
            initial_context = await self._prepare_trip_context(trip_request)

            if background:
                # Submit as background task
                task = asyncio.create_task(
                    self._execute_trip_workflow(
                        orchestrator,
                        workflow_def,
                        initial_context,
                        trip_request,
                        agent_team,
                    )
                )
                # Store task reference for tracking
                await self._store_background_task(
                    user_id, trip_request.request_id, task
                )

                # Return immediate response for background task
                return WorkflowResult(
                    workflow_id=workflow_def.workflow_id,
                    execution_id=str(uuid4()),
                    request_id=trip_request.request_id,
                    success=True,
                    itinerary=None,
                    execution_time=0.0,
                    agents_used=[],
                    user_satisfaction_predicted=None,
                )
            else:
                # Execute synchronously
                return await self._execute_trip_workflow(
                    orchestrator,
                    workflow_def,
                    initial_context,
                    trip_request,
                    agent_team,
                )

        except Exception as e:
            logger.error(
                "Failed to create trip plan",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise TripPlanGenerationError(
                f"Failed to create trip plan: {e}",
                details={"user_id": user_id, "request_id": trip_request.request_id},
            ) from e

    async def _execute_trip_workflow(
        self,
        orchestrator: TripPlannerOrchestrator,
        workflow_def: WorkflowDefinition,
        initial_context: Dict[str, Any],
        trip_request: TripRequest,
        agent_team: Dict[AgentRole, BaseAgent],
    ) -> WorkflowResult:
        """Execute trip planning workflow."""
        start_time = datetime.now(timezone.utc)

        try:
            # Execute workflow
            execution = await orchestrator.execute_workflow(
                workflow_def, initial_context
            )

            # Process workflow results
            itinerary = None
            if execution.context.get("trip_plan_created"):
                itinerary = await self._build_itinerary_from_context(
                    execution.context, trip_request
                )

                # Save itinerary to Firestore
                if itinerary:
                    await self._save_itinerary(itinerary)
            else:
                logger.warning(
                    "Workflow completed without trip_plan_created flag",
                    workflow_id=workflow_def.workflow_id,
                    execution_id=execution.execution_id,
                )

            # Create workflow result
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            result = WorkflowResult(
                workflow_id=workflow_def.workflow_id,
                execution_id=execution.execution_id,
                request_id=trip_request.request_id,
                success=execution.state.value == "completed",
                itinerary=itinerary,
                agent_responses=[],  # Convert from execution results if needed
                execution_time=execution_time,
                agents_used=list(execution.agent_results.keys()),
                function_calls_total=sum(
                    len(r.function_calls) for r in execution.agent_results.values()
                ),
                tokens_consumed=execution.total_tokens,
                errors=execution.error_log,
                user_satisfaction_predicted=None,
            )

            logger.info(
                "Trip workflow completed",
                workflow_id=workflow_def.workflow_id,
                execution_id=execution.execution_id,
                success=result.success,
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            logger.error(
                "Trip workflow execution failed",
                workflow_id=workflow_def.workflow_id,
                error=str(e),
                exc_info=True,
            )

            return WorkflowResult(
                workflow_id=workflow_def.workflow_id,
                execution_id=str(uuid4()),
                request_id=trip_request.request_id,
                success=False,
                itinerary=None,
                execution_time=execution_time,
                agents_used=[],
                errors=[str(e)],
                user_satisfaction_predicted=None,
            )
        finally:
            await self._cleanup_agents(agent_team)

    def _build_trip_requirements(self, trip_request: TripRequest) -> TripRequirements:
        """Convert TripRequest model into agent factory requirements."""

        complexity = TripComplexity.SIMPLE
        duration_days = trip_request.duration_days

        if duration_days >= 5 or len(trip_request.additional_destinations) > 0:
            complexity = TripComplexity.MODERATE

        if (
            duration_days >= 8
            or trip_request.special_requirements
            or trip_request.avoid
        ):
            complexity = TripComplexity.COMPLEX

        if trip_request.trip_type in {TripType.BUSINESS, TripType.LUXURY}:
            complexity = TripComplexity.ENTERPRISE

        return TripRequirements(
            destination=trip_request.destination,
            duration_days=duration_days,
            traveler_count=trip_request.traveler_count,
            budget_range=(
                str(trip_request.budget.total_budget) if trip_request.budget else None
            ),
            trip_type=trip_request.trip_type.value,
            complexity=complexity,
            special_requirements=trip_request.special_requirements,
            preferred_activities=[
                activity.value for activity in trip_request.preferred_activities
            ],
            transportation_modes=[
                mode.value for mode in trip_request.transportation_preferences
            ],
            accommodation_preferences=trip_request.accommodation_preferences,
            dietary_restrictions=trip_request.dietary_restrictions,
            accessibility_needs=trip_request.accessibility_needs,
        )

    async def _cleanup_agents(self, agent_team: Dict[AgentRole, BaseAgent]) -> None:
        """Ensure agent resources are released after workflow execution."""
        cleanup_tasks = []
        for agent in agent_team.values():
            try:
                cleanup_tasks.append(agent.cleanup())
            except Exception:
                logger.warning(
                    "Failed to schedule agent cleanup",
                    agent_id=agent.agent_id,
                )

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def _prepare_trip_context(self, trip_request: TripRequest) -> Dict[str, Any]:
        """Prepare context for AI agents from trip request."""
        context = {
            "destination": trip_request.destination,
            "additional_destinations": trip_request.additional_destinations,
            "travel_dates": {
                "start_date": trip_request.start_date.isoformat(),
                "end_date": trip_request.end_date.isoformat(),
            },
            "duration_days": trip_request.duration_days,
            "traveler_count": trip_request.traveler_count,
            "trip_type": trip_request.trip_type.value,
            "preferences": {
                "activities": [act.value for act in trip_request.preferred_activities],
                "accommodation": trip_request.accommodation_preferences,
                "transportation": [
                    mode.value for mode in trip_request.transportation_preferences
                ],
                "dietary_restrictions": trip_request.dietary_restrictions,
                "accessibility_needs": trip_request.accessibility_needs,
            },
            "constraints": {
                "budget": (
                    trip_request.budget.model_dump() if trip_request.budget else None
                ),
                "must_include": trip_request.must_include,
                "avoid": trip_request.avoid,
                "special_requirements": trip_request.special_requirements,
            },
        }

        # Add location data from Maps API
        try:
            # Geocode main destination
            location_result = await self._geocoding_service.geocode_address(
                trip_request.destination
            )
            if location_result and location_result.results:
                # Handle the geocoding response properly
                first_result = location_result.results[0]

                # Extract coordinates based on the actual geocoding response structure
                dest_coords = None
                if isinstance(first_result, dict) and "geometry" in first_result:
                    geometry = first_result["geometry"]
                    if "location" in geometry:
                        location = geometry["location"]
                        dest_coords = GeoLocation(
                            lat=location.get("lat", 0.0), lng=location.get("lng", 0.0)
                        )

                if dest_coords:
                    context["destination_coordinates"] = {
                        "lat": dest_coords.latitude,
                        "lng": dest_coords.longitude,
                    }

                    # Get popular places for destination
                    popular_places = await self._places_service.find_popular_places(
                        location=dest_coords,
                        radius=10000,  # 10km radius
                        min_rating=4.0,
                    )
                    context["popular_places"] = [
                        {
                            "name": place.name,
                            "place_id": place.place_id,
                            "rating": place.rating,
                            "types": place.types,
                        }
                        for place in popular_places[:20]  # Limit to top 20
                    ]

        except Exception as e:
            logger.warning(
                "Failed to add location data to context",
                destination=trip_request.destination,
                error=str(e),
            )

        return context

    async def _build_itinerary_from_context(
        self, context: Dict[str, Any], trip_request: TripRequest
    ) -> Optional[TripItinerary]:
        """Build TripItinerary from workflow context."""
        try:
            duration_days = trip_request.duration_days
            # Extract itinerary data from context
            itinerary_data = context.get("itinerary", {})
            if not itinerary_data:
                logger.warning("No itinerary data found in context")
                return None

            # Build daily plans
            daily_plans = []
            for day_data in itinerary_data.get("daily_plans", []):
                day_plan = DayPlan(**day_data)
                daily_plans.append(day_plan)

            # Build budget
            budget_data = context.get("budget", {})
            if budget_data:
                breakdown = {
                    key: Decimal(str(value))
                    for key, value in budget_data.get(
                        "daily_cost_breakdown", {}
                    ).items()
                }
                daily_total = Decimal(str(budget_data.get("daily_total", "0")))
                recommended_total = Decimal(
                    str(budget_data.get("recommended_budget", "0"))
                )
                if recommended_total <= 0:
                    recommended_total = daily_total * Decimal(duration_days)

                budget = Budget(
                    total_budget=recommended_total,
                    currency=budget_data.get("currency", "USD"),
                    breakdown=breakdown,
                    contingency_percentage=0.15,
                    daily_budget=daily_total,
                    cost_optimization_tips=budget_data.get(
                        "cost_optimization_tips", []
                    ),
                )
            else:
                total_budget = Decimal("1000.0")
                budget = Budget(
                    total_budget=total_budget,
                    currency="INR",
                    remaining_amount=total_budget,
                    daily_budget=total_budget / trip_request.duration_days,
                )

            # Create itinerary
            itinerary = TripItinerary(
                request_id=trip_request.request_id,
                user_id=trip_request.user_id,
                title=f"AI-Generated Trip to {trip_request.destination}",
                description=f"Comprehensive {trip_request.duration_days}-day trip plan",
                destination=trip_request.destination,
                start_date=trip_request.start_date,
                end_date=trip_request.end_date,
                duration_days=trip_request.duration_days,
                traveler_count=trip_request.traveler_count,
                daily_plans=daily_plans,
                overall_budget=budget,
                created_by_agents=list(context.get("agents_used", [])),
                updated_at=None,
            )

            logger.info(
                "Itinerary built from context",
                itinerary_id=itinerary.itinerary_id,
                days=len(daily_plans),
            )

            context["trip_plan_created"] = True

            return itinerary

        except Exception as e:
            logger.error(
                "Failed to build itinerary from context",
                error=str(e),
                exc_info=True,
            )
            return None

    async def get_trip(self, trip_id: str, user_id: str) -> TripItinerary:
        """Get trip by ID with access validation.

        Args:
            trip_id: Trip identifier
            user_id: User identifier

        Returns:
            TripItinerary: Trip data

        Raises:
            TripNotFoundError: If trip not found
            TripAccessDeniedError: If user doesn't have access
        """
        try:
            # Get trip from Firestore
            trip_data = await self._firestore_client.get_document(
                self._trips_collection, trip_id
            )

            if not trip_data:
                raise TripNotFoundError(f"Trip not found: {trip_id}")

            # Validate access
            trip_user_id = trip_data.get("user_id")
            shared_with = trip_data.get("shared_with", [])

            if trip_user_id != user_id and user_id not in shared_with:
                raise TripAccessDeniedError(f"Access denied to trip: {trip_id}")

            # Convert to TripItinerary model
            itinerary = TripItinerary.model_validate(trip_data)

            logger.debug("Trip retrieved", trip_id=trip_id, user_id=user_id)

            return itinerary

        except (TripNotFoundError, TripAccessDeniedError):
            raise
        except Exception as e:
            logger.error(
                "Failed to get trip",
                trip_id=trip_id,
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise TripPlannerServiceError(f"Failed to get trip: {e}") from e

    async def update_trip(
        self, trip_id: str, user_id: str, updates: Dict[str, Any]
    ) -> TripItinerary:
        """Update trip with AI optimization.

        Args:
            trip_id: Trip identifier
            user_id: User identifier
            updates: Updates to apply

        Returns:
            TripItinerary: Updated trip

        Raises:
            TripNotFoundError: If trip not found
            TripAccessDeniedError: If user doesn't have access
        """
        try:
            # Get existing trip
            itinerary = await self.get_trip(trip_id, user_id)

            # Apply updates
            for key, value in updates.items():
                if hasattr(itinerary, key):
                    setattr(itinerary, key, value)

            # Update metadata
            itinerary.updated_at = datetime.now(timezone.utc)
            itinerary.version += 1

            # Save to Firestore
            await self._save_itinerary(itinerary)

            logger.info(
                "Trip updated",
                trip_id=trip_id,
                user_id=user_id,
                updates=list(updates.keys()),
            )

            return itinerary

        except (TripNotFoundError, TripAccessDeniedError):
            raise
        except Exception as e:
            logger.error(
                "Failed to update trip",
                trip_id=trip_id,
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise TripPlannerServiceError(f"Failed to update trip: {e}") from e

    async def delete_trip(self, trip_id: str, user_id: str) -> bool:
        """Delete trip from Firestore.

        Args:
            trip_id: Trip identifier
            user_id: User identifier

        Returns:
            bool: True if deleted successfully

        Raises:
            TripNotFoundError: If trip not found
            TripAccessDeniedError: If user doesn't have access
        """
        try:
            # Validate access first
            await self.get_trip(trip_id, user_id)

            # Delete from Firestore
            deleted = await self._firestore_client.delete_document(
                self._trips_collection, trip_id
            )

            if deleted:
                logger.info("Trip deleted", trip_id=trip_id, user_id=user_id)

            return deleted

        except (TripNotFoundError, TripAccessDeniedError):
            raise
        except Exception as e:
            logger.error(
                "Failed to delete trip",
                trip_id=trip_id,
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise TripPlannerServiceError(f"Failed to delete trip: {e}") from e

    async def list_user_trips(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List user's trips with pagination and filtering.

        Args:
            user_id: User identifier
            filters: Optional filters
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            limit: Number of trips to return
            offset: Pagination offset

        Returns:
            Dict containing trips and pagination info
        """
        try:
            # Build query filters
            query_filters = [("user_id", "==", user_id)]

            if filters:
                if filters.get("destination"):
                    query_filters.append(("destination", "==", filters["destination"]))
                if filters.get("status"):
                    query_filters.append(("status", "==", filters["status"]))

            # Query trips
            trip_docs = await self._firestore_client.query_documents(
                collection_name=self._trips_collection,
                filters=query_filters,
                order_by=[(sort_by, sort_order)],
                limit=limit + 1,  # Get one extra to check for next page
            )

            # Process results
            trips = []
            for doc_data in trip_docs[:limit]:  # Take only requested limit
                try:
                    trip = TripItinerary.model_validate(doc_data)
                    trips.append(trip)
                except Exception as e:
                    logger.warning(
                        "Failed to parse trip data",
                        trip_id=doc_data.get("id"),
                        error=str(e),
                    )

            # Pagination info
            has_next_page = len(trip_docs) > limit
            next_offset = offset + limit if has_next_page else None

            result = {
                "trips": trips,
                "pagination": {
                    "total": len(trips),
                    "offset": offset,
                    "limit": limit,
                    "has_next_page": has_next_page,
                    "next_offset": next_offset,
                },
            }

            logger.debug(
                "User trips listed",
                user_id=user_id,
                count=len(trips),
                has_next_page=has_next_page,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to list user trips",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise TripPlannerServiceError(f"Failed to list trips: {e}") from e

    async def share_trip(
        self, trip_id: str, user_id: str, shared_with: List[str]
    ) -> Dict[str, Any]:
        """Share trip with other users.

        Args:
            trip_id: Trip identifier
            user_id: Trip owner user ID
            shared_with: List of user IDs to share with

        Returns:
            Dict with sharing information
        """
        try:
            # Get and validate trip ownership
            itinerary = await self.get_trip(trip_id, user_id)

            if itinerary.user_id != user_id:
                raise TripAccessDeniedError("Only trip owner can share trips")

            # Get current sharing data from Firestore document
            trip_data = await self._firestore_client.get_document(
                self._trips_collection, trip_id
            )

            if not trip_data:
                raise TripNotFoundError(f"Trip not found: {trip_id}")

            # Update sharing
            current_shared = set(trip_data.get("shared_with", []))
            current_shared.update(shared_with)

            # Update in Firestore
            updates = {
                "shared_with": list(current_shared),
                "updated_at": datetime.now(timezone.utc),
            }

            await self._firestore_client.update_document(
                self._trips_collection, trip_id, updates
            )

            logger.info(
                "Trip shared",
                trip_id=trip_id,
                user_id=user_id,
                shared_with=shared_with,
            )

            return {
                "trip_id": trip_id,
                "shared_with": list(current_shared),
                "updated_at": updates["updated_at"],
            }

        except (TripNotFoundError, TripAccessDeniedError):
            raise
        except Exception as e:
            logger.error(
                "Failed to share trip",
                trip_id=trip_id,
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise TripPlannerServiceError(f"Failed to share trip: {e}") from e

    async def _save_itinerary(self, itinerary: TripItinerary) -> str:
        """Save itinerary to Firestore.

        Args:
            itinerary: Itinerary to save

        Returns:
            str: Document ID
        """
        try:
            # Convert to dictionary
            itinerary_data = itinerary.model_dump(mode="json")

            # Handle datetime fields
            itinerary_data["created_at"] = itinerary.created_at.isoformat()
            if itinerary.updated_at:
                itinerary_data["updated_at"] = itinerary.updated_at.isoformat()

            # Add sharing field (not part of the model but needed for database)
            itinerary_data["shared_with"] = []

            # Save to Firestore
            doc_id = await self._firestore_client.create_document(
                self._trips_collection,
                itinerary_data,
                document_id=itinerary.itinerary_id,
            )

            logger.debug("Itinerary saved", document_id=doc_id)
            return doc_id

        except Exception as e:
            logger.error(
                "Failed to save itinerary",
                itinerary_id=itinerary.itinerary_id,
                error=str(e),
                exc_info=True,
            )
            raise

    async def _store_background_task(
        self, user_id: str, request_id: str, task: asyncio.Task
    ) -> None:
        """Store background task reference for tracking."""
        # Note: In production, implement proper task storage using Redis or similar
        # For now, we just log the task creation
        logger.info(
            "Background task started",
            user_id=user_id,
            request_id=request_id,
            task_id=id(task),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    async def get_trip_metrics(self, trip_id: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a trip.

        Args:
            trip_id: Trip identifier
            user_id: User identifier

        Returns:
            Dict containing trip metrics
        """
        try:
            itinerary = await self.get_trip(trip_id, user_id)

            # Calculate metrics
            metrics = calculate_itinerary_metrics(itinerary)
            validation = validate_itinerary_completeness(itinerary)

            return {
                "trip_id": trip_id,
                "metrics": metrics,
                "validation": validation,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(
                "Failed to get trip metrics",
                trip_id=trip_id,
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise TripPlannerServiceError(f"Failed to get trip metrics: {e}") from e


# Global service instance
_trip_planner_service: Optional[TripPlannerService] = None


def get_trip_planner_service() -> TripPlannerService:
    """Get or create global trip planner service instance."""
    global _trip_planner_service

    if _trip_planner_service is None:
        _trip_planner_service = TripPlannerService()

    return _trip_planner_service
