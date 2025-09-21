from unittest.mock import AsyncMock, patch

import pytest

from src.trip_planner.schemas import TripRequest
from src.trip_planner.services import TripPlannerService


@pytest.fixture
def mock_orchestrator():
    """Fixture for a mocked AgentOrchestrator."""
    return AsyncMock()


@pytest.fixture
def mock_db_client():
    """Fixture for a mocked database client."""
    return AsyncMock()


@pytest.fixture
def trip_planner_service(mock_orchestrator, mock_db_client):
    """Fixture for the TripPlannerService with mocked dependencies."""
    with (
        patch(
            "src.trip_planner.services.get_trip_planner_orchestrator"
        ) as mock_get_orchestrator,
        patch("src.trip_planner.services.get_firestore_client") as mock_get_db,
    ):
        mock_get_orchestrator.return_value = mock_orchestrator
        mock_get_db.return_value = mock_db_client
        service = TripPlannerService()
        yield service


@pytest.mark.asyncio
async def test_create_trip(trip_planner_service, mock_orchestrator, mock_db_client):
    """Test creating a new trip."""
    trip_request_data = {
        "user_id": "user1",
        "destination": "Hawaii",
        "start_date": "2026-01-01",
        "end_date": "2026-01-07",
        "duration_days": 7,
    }
    trip_request = TripRequest(**trip_request_data)

    mock_orchestrator.execute_workflow.return_value.context = {
        "trip_plan_created": True
    }
    mock_db_client.create_document.return_value = "new_trip_id"

    result = await trip_planner_service.create_trip_plan(
        user_id="user1", trip_request=trip_request
    )

    assert result.success is True
    mock_orchestrator.execute_workflow.assert_called_once()


@pytest.mark.asyncio
async def test_get_trip(trip_planner_service, mock_db_client):
    """Test retrieving an existing trip."""
    mock_db_client.get_document.return_value = {
        "itinerary_id": "trip1",
        "user_id": "user1",
        "title": "Trip to Hawaii",
        "destination": "Hawaii",
        "start_date": "2026-01-01",
        "end_date": "2026-01-07",
        "duration_days": 7,
        "traveler_count": 1,
        "daily_plans": [],
        "overall_budget": {"total_budget": 2000, "currency": "USD"},
        "request_id": "req1",
    }

    trip = await trip_planner_service.get_trip("trip1", "user1")

    assert trip is not None
    assert trip.destination == "Hawaii"
    mock_db_client.get_document.assert_called_once_with("trips", "trip1")


@pytest.mark.asyncio
async def test_get_trip_not_found(trip_planner_service, mock_db_client):
    """Test retrieving a nonexistent trip."""
    mock_db_client.get_document.return_value = None

    with pytest.raises(Exception):
        await trip_planner_service.get_trip("nonexistent", "user1")
