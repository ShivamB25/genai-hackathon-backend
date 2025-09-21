"""Background Tasks for AI-Powered Trip Planner Backend.

This module provides background task implementations for async trip generation,
trip optimization, database cleanup, and user analytics with proper logging
and error handling.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.ai_services.agent_orchestrator import (
    create_comprehensive_trip_planning_workflow,
    get_trip_planner_orchestrator,
)
from src.ai_services.session_manager import SessionStatus
from src.core.logging import get_logger
from src.database.firestore_client import get_firestore_client
from src.trip_planner.schemas import TripRequest, WorkflowResult
from src.trip_planner.services import (
    get_trip_planner_service,
)

logger = get_logger(__name__)


class BackgroundTaskError(Exception):
    """Base exception for background task errors."""


class TripGenerationTaskError(BackgroundTaskError):
    """Error during trip generation background task."""


class CleanupTaskError(BackgroundTaskError):
    """Error during cleanup background task."""


class AnalyticsTaskError(BackgroundTaskError):
    """Error during analytics background task."""


async def generate_trip_async(
    user_id: str,
    trip_request: TripRequest,
    workflow_type: str = "comprehensive",
    notify_completion: bool = True,
) -> WorkflowResult:
    """Background task for async trip generation.

    Args:
        user_id: User identifier
        trip_request: Trip planning request
        workflow_type: Type of workflow to use
        notify_completion: Whether to notify user on completion

    Returns:
        WorkflowResult: Trip generation result

    Raises:
        TripGenerationTaskError: If generation fails
    """
    task_id = str(uuid4())
    start_time = datetime.now(timezone.utc)

    try:
        logger.info(
            "Starting async trip generation",
            task_id=task_id,
            user_id=user_id,
            destination=trip_request.destination,
            workflow_type=workflow_type,
        )

        # Get trip planner service
        service = get_trip_planner_service()

        # Execute trip planning workflow
        result = await service.create_trip_plan(
            user_id=user_id,
            trip_request=trip_request,
            workflow_type=workflow_type,
            background=False,  # We're already in background
        )

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Log task completion
        logger.info(
            "Async trip generation completed",
            task_id=task_id,
            user_id=user_id,
            success=result.success,
            execution_time=execution_time,
            itinerary_id=result.itinerary.itinerary_id if result.itinerary else None,
        )

        # Store task completion analytics
        await _store_task_analytics(
            task_type="trip_generation",
            task_id=task_id,
            user_id=user_id,
            execution_time=execution_time,
            success=result.success,
            metadata={
                "destination": trip_request.destination,
                "workflow_type": workflow_type,
                "duration_days": trip_request.duration_days,
            },
        )

        # Notify user of completion (placeholder - implement with notification service)
        if notify_completion:
            await _notify_trip_completion(user_id, result)

        return result

    except Exception as e:
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.error(
            "Async trip generation failed",
            task_id=task_id,
            user_id=user_id,
            error=str(e),
            execution_time=execution_time,
            exc_info=True,
        )

        # Store failure analytics
        await _store_task_analytics(
            task_type="trip_generation",
            task_id=task_id,
            user_id=user_id,
            execution_time=execution_time,
            success=False,
            error=str(e),
        )

        raise TripGenerationTaskError(f"Trip generation failed: {e}") from e


async def optimize_trip_async(
    trip_id: str,
    user_id: str,
    optimization_criteria: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Background task for trip optimization using AI agents.

    Args:
        trip_id: Trip identifier to optimize
        user_id: User identifier
        optimization_criteria: List of optimization criteria
        weights: Weights for different criteria

    Returns:
        Dict with optimization results

    Raises:
        TripGenerationTaskError: If optimization fails
    """
    task_id = str(uuid4())
    start_time = datetime.now(timezone.utc)

    try:
        logger.info(
            "Starting async trip optimization",
            task_id=task_id,
            trip_id=trip_id,
            user_id=user_id,
            criteria=optimization_criteria,
        )

        # Get trip planner service and orchestrator
        service = get_trip_planner_service()
        orchestrator = get_trip_planner_orchestrator(f"user_{user_id}_optimization")

        # Get existing trip
        trip = await service.get_trip(trip_id, user_id)

        # Prepare optimization context
        optimization_context = {
            "trip_id": trip_id,
            "current_itinerary": trip.model_dump(),
            "optimization_criteria": optimization_criteria,
            "weights": weights or {},
            "user_preferences": {},  # Add user preferences if available
        }

        # Create optimization workflow
        workflow_def = create_comprehensive_trip_planning_workflow()
        workflow_def.name = "Trip Optimization"
        workflow_def.description = "Optimize existing trip using AI agents"

        # Execute optimization workflow
        execution = await orchestrator.execute_workflow(
            workflow_def, optimization_context
        )

        # Process optimization results
        optimization_result = {
            "trip_id": trip_id,
            "optimization_id": str(uuid4()),
            "success": execution.state.value == "completed",
            "execution_time": execution.total_execution_time,
            "improvements": execution.context.get("optimizations", {}),
            "updated_itinerary": execution.context.get("optimized_itinerary"),
            "cost_savings": execution.context.get("cost_savings", 0.0),
            "time_savings": execution.context.get("time_savings", 0.0),
        }

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            "Async trip optimization completed",
            task_id=task_id,
            trip_id=trip_id,
            success=optimization_result["success"],
            execution_time=execution_time,
        )

        # Store analytics
        await _store_task_analytics(
            task_type="trip_optimization",
            task_id=task_id,
            user_id=user_id,
            execution_time=execution_time,
            success=optimization_result["success"],
            metadata={
                "trip_id": trip_id,
                "criteria": optimization_criteria,
                "cost_savings": optimization_result["cost_savings"],
            },
        )

        return optimization_result

    except Exception as e:
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.error(
            "Async trip optimization failed",
            task_id=task_id,
            trip_id=trip_id,
            error=str(e),
            execution_time=execution_time,
            exc_info=True,
        )

        await _store_task_analytics(
            task_type="trip_optimization",
            task_id=task_id,
            user_id=user_id,
            execution_time=execution_time,
            success=False,
            error=str(e),
        )

        raise TripGenerationTaskError(f"Trip optimization failed: {e}") from e


async def cleanup_expired_sessions() -> Dict[str, Any]:
    """Background task for Firestore cleanup of expired sessions and trips.

    Returns:
        Dict with cleanup statistics

    Raises:
        CleanupTaskError: If cleanup fails
    """
    task_id = str(uuid4())
    start_time = datetime.now(timezone.utc)

    try:
        logger.info("Starting Firestore cleanup task", task_id=task_id)

        db = get_firestore_client()

        cleanup_stats = {
            "sessions_cleaned": 0,
            "trips_cleaned": 0,
            "temp_data_cleaned": 0,
            "execution_time": 0.0,
        }

        # Clean up expired AI sessions
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        expired_sessions = await db.query_documents(
            collection_name="ai_sessions",
            filters=[
                ("status", "==", SessionStatus.EXPIRED.value),
                ("expires_at", "<=", cutoff_time.isoformat()),
            ],
            limit=100,  # Process in batches
        )

        for session_doc in expired_sessions:
            try:
                await db.delete_document("ai_sessions", session_doc["id"])
                cleanup_stats["sessions_cleaned"] += 1
                logger.debug("Cleaned expired session", session_id=session_doc["id"])
            except Exception as e:
                logger.warning(
                    "Failed to delete expired session",
                    session_id=session_doc["id"],
                    error=str(e),
                )

        # Clean up old temporary data
        temp_data_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
        temp_docs = await db.query_documents(
            collection_name="_temp_data",
            filters=[("created_at", "<=", temp_data_cutoff.isoformat())],
            limit=100,
        )

        for temp_doc in temp_docs:
            try:
                await db.delete_document("_temp_data", temp_doc["id"])
                cleanup_stats["temp_data_cleaned"] += 1
            except Exception as e:
                logger.warning(
                    "Failed to delete temp data",
                    doc_id=temp_doc["id"],
                    error=str(e),
                )

        # Clean up draft trips older than 7 days
        draft_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        old_drafts = await db.query_documents(
            collection_name="trips",
            filters=[
                ("status", "==", "draft"),
                ("created_at", "<=", draft_cutoff.isoformat()),
            ],
            limit=50,
        )

        for draft_doc in old_drafts:
            try:
                await db.delete_document("trips", draft_doc["id"])
                cleanup_stats["trips_cleaned"] += 1
                logger.debug("Cleaned old draft trip", trip_id=draft_doc["id"])
            except Exception as e:
                logger.warning(
                    "Failed to delete old draft",
                    trip_id=draft_doc["id"],
                    error=str(e),
                )

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        cleanup_stats["execution_time"] = execution_time

        logger.info(
            "Firestore cleanup completed",
            task_id=task_id,
            stats=cleanup_stats,
        )

        # Store cleanup analytics
        await _store_task_analytics(
            task_type="firestore_cleanup",
            task_id=task_id,
            user_id="system",
            execution_time=execution_time,
            success=True,
            metadata=cleanup_stats,
        )

        return cleanup_stats

    except Exception as e:
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.error(
            "Firestore cleanup failed",
            task_id=task_id,
            error=str(e),
            execution_time=execution_time,
            exc_info=True,
        )

        await _store_task_analytics(
            task_type="firestore_cleanup",
            task_id=task_id,
            user_id="system",
            execution_time=execution_time,
            success=False,
            error=str(e),
        )

        raise CleanupTaskError(f"Cleanup task failed: {e}") from e


async def track_user_analytics(
    user_id: str,
    event_type: str,
    event_data: Dict[str, Any],
) -> None:
    """Background task for user analytics and usage tracking.

    Args:
        user_id: User identifier
        event_type: Type of event to track
        event_data: Event metadata

    Raises:
        AnalyticsTaskError: If analytics tracking fails
    """
    task_id = str(uuid4())

    try:
        logger.debug(
            "Tracking user analytics",
            task_id=task_id,
            user_id=user_id,
            event_type=event_type,
        )

        db = get_firestore_client()

        # Create analytics event
        analytics_data = {
            "user_id": user_id,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now(timezone.utc),
            "session_id": event_data.get("session_id"),
            "ip_address": event_data.get("ip_address"),
            "user_agent": event_data.get("user_agent"),
        }

        # Store in analytics collection
        await db.create_document("user_analytics", analytics_data)

        # Update user activity timestamp
        await db.update_document(
            "users",
            user_id,
            {"last_activity": datetime.now(timezone.utc)},
        )

        logger.debug(
            "User analytics tracked",
            task_id=task_id,
            user_id=user_id,
            event_type=event_type,
        )

    except Exception as e:
        logger.error(
            "Failed to track user analytics",
            task_id=task_id,
            user_id=user_id,
            event_type=event_type,
            error=str(e),
            exc_info=True,
        )
        # Don't raise exception for analytics failures to avoid disrupting main flow


async def optimize_trip_recommendations(user_id: str) -> Dict[str, Any]:
    """Background task to generate personalized trip recommendations.

    Args:
        user_id: User identifier

    Returns:
        Dict with recommendation results
    """
    task_id = str(uuid4())
    start_time = datetime.now(timezone.utc)

    try:
        logger.info(
            "Starting trip recommendations optimization",
            task_id=task_id,
            user_id=user_id,
        )

        db = get_firestore_client()

        # Get user profile and preferences
        user_profile = await db.get_document("users", user_id)
        if not user_profile:
            logger.warning(
                "User profile not found for recommendations", user_id=user_id
            )
            return {"success": False, "error": "User profile not found"}

        # Get user's trip history
        user_trips = await db.query_documents(
            collection_name="trips",
            filters=[("user_id", "==", user_id)],
            order_by=[("created_at", "desc")],
            limit=10,
        )

        # Analyze patterns from trip history
        destinations = []
        preferred_activities = []
        budget_ranges = []

        for trip in user_trips:
            destinations.append(trip.get("destination", ""))

            # Extract activities from daily plans
            daily_plans = trip.get("daily_plans", [])
            for day_plan in daily_plans:
                activities = day_plan.get("activities", [])
                for activity in activities:
                    preferred_activities.append(activity.get("activity_type", ""))

            # Extract budget info
            budget = trip.get("overall_budget", {})
            if budget.get("total_budget"):
                budget_ranges.append(float(budget["total_budget"]))

        # Generate recommendations based on patterns
        recommendations = {
            "recommended_destinations": _generate_destination_recommendations(
                destinations
            ),
            "recommended_activities": _generate_activity_recommendations(
                preferred_activities
            ),
            "suggested_budget_range": _calculate_budget_suggestions(budget_ranges),
            "personalized_tips": _generate_personalized_tips(user_profile, user_trips),
        }

        # Store recommendations
        recommendation_data = {
            "user_id": user_id,
            "recommendations": recommendations,
            "generated_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(days=30),
        }

        await db.create_document(
            "user_recommendations",
            recommendation_data,
            document_id=f"{user_id}_recommendations",
        )

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            "Trip recommendations generated",
            task_id=task_id,
            user_id=user_id,
            execution_time=execution_time,
        )

        return {
            "success": True,
            "recommendations": recommendations,
            "execution_time": execution_time,
        }

    except Exception as e:
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.error(
            "Failed to generate trip recommendations",
            task_id=task_id,
            user_id=user_id,
            error=str(e),
            execution_time=execution_time,
            exc_info=True,
        )

        return {
            "success": False,
            "error": str(e),
            "execution_time": execution_time,
        }


async def periodic_system_maintenance() -> Dict[str, Any]:
    """Background task for periodic system maintenance.

    Returns:
        Dict with maintenance results
    """
    task_id = str(uuid4())
    start_time = datetime.now(timezone.utc)

    try:
        logger.info("Starting periodic system maintenance", task_id=task_id)

        maintenance_stats = {
            "sessions_cleaned": 0,
            "expired_trips_cleaned": 0,
            "analytics_aggregated": 0,
            "errors": [],
        }

        # Clean up expired sessions
        try:
            session_cleanup = await cleanup_expired_sessions()
            maintenance_stats["sessions_cleaned"] = session_cleanup.get(
                "sessions_cleaned", 0
            )
        except Exception as e:
            maintenance_stats["errors"].append(f"Session cleanup failed: {e}")

        # Clean up expired trips
        try:
            db = get_firestore_client()
            expired_cutoff = datetime.now(timezone.utc) - timedelta(
                days=365
            )  # 1 year old

            expired_trips = await db.query_documents(
                collection_name="trips",
                filters=[
                    ("status", "==", "expired"),
                    ("created_at", "<=", expired_cutoff.isoformat()),
                ],
                limit=100,
            )

            for trip_doc in expired_trips:
                try:
                    await db.delete_document("trips", trip_doc["id"])
                    maintenance_stats["expired_trips_cleaned"] += 1
                except Exception as e:
                    maintenance_stats["errors"].append(
                        f"Failed to delete expired trip {trip_doc['id']}: {e}"
                    )

        except Exception as e:
            maintenance_stats["errors"].append(f"Trip cleanup failed: {e}")

        # Aggregate analytics data (placeholder)
        try:
            # This would aggregate daily/weekly analytics
            maintenance_stats["analytics_aggregated"] = 1
        except Exception as e:
            maintenance_stats["errors"].append(f"Analytics aggregation failed: {e}")

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        maintenance_stats["execution_time"] = execution_time

        logger.info(
            "Periodic system maintenance completed",
            task_id=task_id,
            stats=maintenance_stats,
        )

        return maintenance_stats

    except Exception as e:
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.error(
            "Periodic system maintenance failed",
            task_id=task_id,
            error=str(e),
            execution_time=execution_time,
            exc_info=True,
        )

        return {
            "success": False,
            "error": str(e),
            "execution_time": execution_time,
        }


# Helper Functions


async def _store_task_analytics(
    task_type: str,
    task_id: str,
    user_id: str,
    execution_time: float,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Store task execution analytics."""
    try:
        db = get_firestore_client()

        analytics_data = {
            "task_type": task_type,
            "task_id": task_id,
            "user_id": user_id,
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata or {},
        }

        if error:
            analytics_data["error"] = error

        await db.create_document("task_analytics", analytics_data)

    except Exception as e:
        logger.warning(
            "Failed to store task analytics",
            task_id=task_id,
            error=str(e),
        )


async def _notify_trip_completion(user_id: str, result: WorkflowResult) -> None:
    """Notify user of trip completion (placeholder implementation)."""
    try:
        # This would integrate with notification service (email, push, etc.)
        logger.info(
            "Trip completion notification sent",
            user_id=user_id,
            trip_id=result.itinerary.itinerary_id if result.itinerary else None,
            success=result.success,
        )
    except Exception as e:
        logger.warning(
            "Failed to send trip completion notification",
            user_id=user_id,
            error=str(e),
        )


def _generate_destination_recommendations(destinations: List[str]) -> List[str]:
    """Generate destination recommendations based on history."""
    # Simple implementation - in practice, use ML/AI for better recommendations
    unique_destinations = list(set(dest for dest in destinations if dest))

    # For now, return some popular destinations based on patterns
    india_destinations = [
        "Goa",
        "Kerala",
        "Rajasthan",
        "Himachal Pradesh",
        "Tamil Nadu",
    ]
    international_destinations = [
        "Thailand",
        "Singapore",
        "Dubai",
        "Nepal",
        "Sri Lanka",
    ]

    recommendations = []

    # Add recommendations based on user's destination patterns
    if any("india" in dest.lower() for dest in unique_destinations):
        recommendations.extend(india_destinations[:3])
    else:
        recommendations.extend(international_destinations[:3])

    return recommendations[:5]


def _generate_activity_recommendations(activities: List[str]) -> List[str]:
    """Generate activity recommendations based on history."""
    activity_counts = {}
    for activity in activities:
        if activity:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

    # Sort by frequency and return top activities plus similar ones
    top_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)

    recommendations = [activity for activity, _ in top_activities[:3]]

    # Add complementary activities
    activity_mapping = {
        "sightseeing": ["cultural", "historical"],
        "cultural": ["sightseeing", "educational"],
        "adventure": ["nature", "sports"],
        "relaxation": ["wellness", "nature"],
    }

    for activity in recommendations:
        if activity in activity_mapping:
            recommendations.extend(activity_mapping[activity])

    return list(set(recommendations))[:5]


def _calculate_budget_suggestions(budget_ranges: List[float]) -> Dict[str, Any]:
    """Calculate budget suggestions based on history."""
    if not budget_ranges:
        return {
            "suggested_range": "moderate",
            "min_budget": 5000.0,
            "max_budget": 15000.0,
            "currency": "INR",
        }

    avg_budget = sum(budget_ranges) / len(budget_ranges)
    min_budget = min(budget_ranges)
    max_budget = max(budget_ranges)

    # Categorize budget range
    if avg_budget < 10000:
        budget_category = "budget"
    elif avg_budget < 50000:
        budget_category = "moderate"
    else:
        budget_category = "luxury"

    return {
        "suggested_range": budget_category,
        "min_budget": min_budget,
        "max_budget": max_budget,
        "average_budget": avg_budget,
        "currency": "INR",
    }


def _generate_personalized_tips(
    user_profile: Dict[str, Any], trip_history: List[Dict[str, Any]]
) -> List[str]:
    """Generate personalized travel tips."""
    tips = []

    # Based on travel frequency
    if len(trip_history) > 5:
        tips.append(
            "As an experienced traveler, consider exploring off-the-beaten-path destinations"
        )
    else:
        tips.append("Start with popular destinations to build your travel confidence")

    # Based on preferences
    preferences = user_profile.get("preferences", {})
    travel_style = preferences.get("travel_style")

    if travel_style == "budget":
        tips.append("Consider traveling during off-peak seasons for better deals")
        tips.append("Look for local transportation options to save money")
    elif travel_style == "luxury":
        tips.append("Book accommodations well in advance for the best luxury options")
        tips.append("Consider private tours for exclusive experiences")

    # Based on activity interests
    interests = preferences.get("activity_interests", [])
    if "cultural" in interests:
        tips.append(
            "Research local festivals and cultural events during your travel dates"
        )
    if "outdoor" in interests:
        tips.append(
            "Check weather conditions and pack appropriate gear for outdoor activities"
        )

    return tips[:5]


# Task scheduling functions (would integrate with Celery, RQ, or similar in production)


# Global task references
_cleanup_task: Optional[asyncio.Task] = None
_analytics_task: Optional[asyncio.Task] = None


async def schedule_cleanup_task() -> None:
    """Schedule periodic cleanup task."""
    global _cleanup_task
    try:
        # Create background task for cleanup
        _cleanup_task = asyncio.create_task(cleanup_expired_sessions())
        logger.info("Cleanup task scheduled")
    except Exception:
        logger.exception("Failed to schedule cleanup task")


async def schedule_user_analytics_aggregation() -> None:
    """Schedule user analytics aggregation task."""
    global _analytics_task
    try:
        # This would run daily to aggregate user analytics
        _analytics_task = asyncio.create_task(periodic_system_maintenance())
        logger.info("Analytics aggregation task scheduled")
    except Exception:
        logger.exception("Failed to schedule analytics task")
