from unittest.mock import AsyncMock, MagicMock

import pytest

from src.maps_services.places_service import PlacesService
from src.maps_services.schemas import GeoLocation


@pytest.fixture
def mock_maps_client():
    """Fixture for a mocked MapsAPIClient."""
    return AsyncMock()


@pytest.fixture
def places_service(mock_maps_client):
    """Fixture for the PlacesService with a mocked maps client."""
    service = PlacesService()
    service._client = mock_maps_client
    return service


@pytest.mark.asyncio
async def test_find_places_by_text(places_service, mock_maps_client):
    """Test finding places by a text query."""
    mock_maps_client.places_text_search.return_value = {
        "results": [{"name": "Eiffel Tower", "place_id": "place1"}],
        "status": "OK",
    }

    places = await places_service.find_places_by_text("Eiffel Tower")

    assert len(places) == 1
    assert places[0].name == "Eiffel Tower"
    mock_maps_client.places_text_search.assert_called_once_with(
        query="Eiffel Tower", language="en"
    )


@pytest.mark.asyncio
async def test_find_nearby_places(places_service, mock_maps_client):
    """Test finding nearby places."""
    mock_maps_client.places_nearby.return_value = {
        "results": [{"name": "Louvre Museum", "place_id": "place2"}],
        "status": "OK",
    }

    location = GeoLocation(lat=48.8584, lng=2.2945)
    places = await places_service.find_nearby_places(
        location, radius=1000, keyword="museum"
    )

    assert len(places) == 1
    assert places[0].name == "Louvre Museum"
    mock_maps_client.places_nearby.assert_called_once()


@pytest.mark.asyncio
async def test_get_place_details(places_service, mock_maps_client):
    """Test getting details for a specific place."""
    mock_maps_client.place_details.return_value = {
        "result": {"name": "Eiffel Tower", "formatted_address": "Paris, France"},
        "status": "OK",
    }

    details = await places_service.get_place_details("place1")

    assert details["name"] == "Eiffel Tower"
    assert "Paris, France" in details["formatted_address"]
    mock_maps_client.place_details.assert_called_once_with(
        place_id="place1", language="en"
    )
