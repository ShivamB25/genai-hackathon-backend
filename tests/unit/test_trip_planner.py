from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.ai_services.gemini_agents import AgentRole
from src.maps_services.schemas import GeoLocation
from src.trip_planner.schemas import ActivityType, TripRequest
from src.trip_planner.services import TripNotFoundError, TripPlannerService


def _build_stub_agent(role: AgentRole) -> AsyncMock:
    agent = AsyncMock()
    agent.agent_id = f"{role.value}_agent"
    agent.capabilities = SimpleNamespace(role=role)
    agent.initialize = AsyncMock()
    agent.cleanup = AsyncMock()
    agent.process_message = AsyncMock(return_value={})
    return agent


@pytest.fixture
def mock_orchestrator():
    orchestrator = AsyncMock()
    orchestrator.register_agent = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_db_client():
    return AsyncMock()


@pytest.fixture
def trip_planner_service(mock_orchestrator, mock_db_client):
    with (
        patch(
            "src.trip_planner.services.get_trip_planner_orchestrator"
        ) as mock_get_orchestrator,
        patch("src.trip_planner.services.get_firestore_client") as mock_get_db,
        patch("src.trip_planner.services.get_agent_factory") as mock_get_factory,
        patch(
            "src.trip_planner.services.get_session_manager"
        ) as mock_get_session_manager,
    ):
        mock_get_orchestrator.return_value = mock_orchestrator
        mock_get_db.return_value = mock_db_client

        class StubAgentFactory:
            def create_agent_team_for_trip(self, *_args, **_kwargs):
                return {
                    role: _build_stub_agent(role)
                    for role in [
                        AgentRole.DESTINATION_EXPERT,
                        AgentRole.BUDGET_ADVISOR,
                        AgentRole.TRIP_PLANNER,
                    ]
                }

        mock_get_factory.return_value = StubAgentFactory()

        session_stub = AsyncMock()
        session_stub.update_context = lambda **_kwargs: None
        mock_session_manager = AsyncMock()
        mock_session_manager.get_session = AsyncMock(return_value=session_stub)
        mock_session_manager.update_session = AsyncMock()
        mock_session_manager.create_session = AsyncMock(return_value=session_stub)
        mock_get_session_manager.return_value = mock_session_manager

        service = TripPlannerService()

        geocode_result = SimpleNamespace(
            geometry=SimpleNamespace(location=GeoLocation(lat=21.3, lng=-157.8))
        )
        service._geocoding_service = AsyncMock()
        service._geocoding_service.geocode_address = AsyncMock(
            return_value=SimpleNamespace(results=[geocode_result])
        )

        popular_place = SimpleNamespace(
            name="Sample Attraction",
            place_id="poi_1",
            rating=4.7,
            types=["tourist_attraction"],
        )

        service._places_service = AsyncMock()
        service._places_service.find_popular_places = AsyncMock(
            return_value=[popular_place]
        )
        service._places_service.search_nearby = AsyncMock(
            return_value=SimpleNamespace(results=[])
        )
        service._places_service.search_text = AsyncMock(
            return_value=SimpleNamespace(results=[])
        )

        service._directions_service = AsyncMock()

        yield service


@pytest.mark.asyncio
async def test_create_trip(trip_planner_service, mock_orchestrator, mock_db_client):
    trip_request = TripRequest(
        user_id="user1",
        destination="Hawaii",
        start_date="2026-01-01",
        end_date="2026-01-07",
        duration_days=7,
    )

    daily_plan_template = {
        "theme": "Explore Hawaii",
        "activities": [
            {
                "name": "Beach Walk",
                "description": "Relax on the beach",
                "activity_type": ActivityType.RELAXATION.value,
                "location": {
                    "place_id": "place1",
                    "name": "Waikiki Beach",
                    "address": {
                        "formatted_address": "Waikiki Beach, Honolulu",
                        "city": "Honolulu",
                        "country": "USA",
                        "location": {
                            "latitude": 21.2767,
                            "longitude": -157.8275,
                        },
                    },
                    "place_types": ["tourist_attraction"],
                },
                "duration": 120,
                "currency": "USD",
            }
        ],
        "transportation": [],
        "meals": [],
        "accommodation": None,
    }

    daily_plans = []
    for offset in range(7):
        plan_date = (trip_request.start_date + timedelta(days=offset)).isoformat()
        daily_plans.append(
            {"day_number": offset + 1, "plan_date": plan_date, **daily_plan_template}
        )

    mock_orchestrator.execute_workflow.return_value = SimpleNamespace(
        workflow_id="wf_123",
        execution_id="exec_456",
        state=SimpleNamespace(value="completed"),
        context={
            "trip_plan_created": True,
            "itinerary": {"daily_plans": daily_plans},
            "budget": {
                "currency": "USD",
                "daily_total": 250,
                "recommended_budget": 2000,
                "daily_cost_breakdown": {
                    "accommodation": 150,
                    "food": 60,
                    "activities": 30,
                    "transport": 10,
                },
                "cost_optimization_tips": [],
            },
        },
        agent_results={},
        total_tokens=0,
        error_log=[],
    )
    mock_db_client.create_document.return_value = "new_trip_id"

    result = await trip_planner_service.create_trip_plan(
        user_id="user1", trip_request=trip_request
    )

    assert result.success is True
    assert result.itinerary is not None
    mock_orchestrator.execute_workflow.assert_called_once()


@pytest.mark.asyncio
async def test_get_trip(trip_planner_service, mock_db_client):
    mock_db_client.get_document.return_value = {
        "itinerary_id": "trip1",
        "user_id": "user1",
        "title": "Trip to Hawaii",
        "description": "Test trip",
        "destination": "Hawaii",
        "start_date": "2026-01-01",
        "end_date": "2026-01-01",
        "duration_days": 1,
        "traveler_count": 1,
        "daily_plans": [
            {
                "day_number": 1,
                "plan_date": "2026-01-01",
                "theme": "Arrival",
                "activities": [],
                "transportation": [],
                "meals": [],
                "accommodation": None,
            }
        ],
        "overall_budget": {"total_budget": 2000, "currency": "USD"},
        "request_id": "req1",
    }

    trip = await trip_planner_service.get_trip("trip1", "user1")

    assert trip is not None
    assert trip.destination == "Hawaii"
    mock_db_client.get_document.assert_called_once_with("trips", "trip1")


@pytest.mark.asyncio
async def test_get_trip_not_found(trip_planner_service, mock_db_client):
    mock_db_client.get_document.return_value = None

    with pytest.raises(TripNotFoundError, match="Trip not found: nonexistent"):
        await trip_planner_service.get_trip("nonexistent", "user1")
