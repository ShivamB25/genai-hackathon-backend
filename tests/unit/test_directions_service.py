from unittest.mock import AsyncMock

import pytest

from src.maps_services.directions_service import DirectionsService
from src.maps_services.schemas import GeoLocation, TravelMode


@pytest.fixture
def mock_maps_client():
    """Fixture for a mocked MapsAPIClient."""
    return AsyncMock()


@pytest.fixture
def directions_service(mock_maps_client):
    """Fixture for the DirectionsService with a mocked maps client."""
    service = DirectionsService()
    service._client = mock_maps_client
    return service


@pytest.mark.asyncio
async def test_get_directions(directions_service, mock_maps_client):
    """Test getting directions between two points."""
    mock_maps_client.directions.return_value = {
        "routes": [
            {
                "summary": "Main St",
                "legs": [
                    {"distance": {"text": "10 km"}, "duration": {"text": "15 mins"}}
                ],
            }
        ],
        "status": "OK",
    }

    origin = GeoLocation(lat=48.8584, lng=2.2945)
    destination = GeoLocation(lat=48.8606, lng=2.3376)

    route = await directions_service.get_directions(origin, destination)

    assert route is not None
    assert route["summary"] == "Main St"
    mock_maps_client.directions.assert_called_once()


@pytest.mark.asyncio
async def test_get_directions_with_transit(directions_service, mock_maps_client):
    """Test getting directions using a specific travel mode."""
    mock_maps_client.directions.return_value = {
        "routes": [{"summary": "Metro Line 1"}],
        "status": "OK",
    }

    origin = "Eiffel Tower"
    destination = "Louvre Museum"

    await directions_service.get_directions(
        origin, destination, mode=TravelMode.TRANSIT
    )

    mock_maps_client.directions.assert_called_once_with(
        origin=origin, destination=destination, mode="transit", language="en"
    )


@pytest.mark.asyncio
async def test_get_directions_no_route_found(directions_service, mock_maps_client):
    """Test handling of no route found."""
    mock_maps_client.directions.return_value = {"routes": [], "status": "ZERO_RESULTS"}

    origin = "Paris"
    destination = "Tokyo"  # Assuming no driving route

    route = await directions_service.get_directions(origin, destination)

    assert route is None
