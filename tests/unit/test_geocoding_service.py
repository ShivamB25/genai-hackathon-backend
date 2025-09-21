from unittest.mock import AsyncMock

import pytest

from src.maps_services.geocoding_service import GeocodingService
from src.maps_services.schemas import GeoLocation


@pytest.fixture
def mock_maps_client():
    """Fixture for a mocked MapsAPIClient."""
    return AsyncMock()


@pytest.fixture
def geocoding_service(mock_maps_client):
    """Fixture for the GeocodingService with a mocked maps client."""
    service = GeocodingService()
    service._client = mock_maps_client
    return service


@pytest.mark.asyncio
async def test_geocode_address(geocoding_service, mock_maps_client):
    """Test geocoding a street address."""
    mock_maps_client.geocode.return_value = {
        "results": [
            {
                "formatted_address": "1600 Amphitheatre Parkway, Mountain View, CA",
                "geometry": {"location": {"lat": 37.422, "lng": -122.084}},
            }
        ],
        "status": "OK",
    }

    result = await geocoding_service.geocode(
        "1600 Amphitheatre Parkway, Mountain View, CA"
    )

    assert result is not None
    assert result["formatted_address"] == "1600 Amphitheatre Parkway, Mountain View, CA"
    assert result["geometry"]["location"]["lat"] == 37.422
    mock_maps_client.geocode.assert_called_once_with(
        address="1600 Amphitheatre Parkway, Mountain View, CA"
    )


@pytest.mark.asyncio
async def test_reverse_geocode(geocoding_service, mock_maps_client):
    """Test reverse geocoding coordinates."""
    mock_maps_client.geocode.return_value = {
        "results": [{"formatted_address": "Eiffel Tower, Paris, France"}],
        "status": "OK",
    }

    location = GeoLocation(lat=48.8584, lng=2.2945)
    result = await geocoding_service.reverse_geocode(location)

    assert result is not None
    assert "Eiffel Tower" in result["formatted_address"]
    mock_maps_client.geocode.assert_called_once_with(location=location)


@pytest.mark.asyncio
async def test_geocode_no_results(geocoding_service, mock_maps_client):
    """Test geocoding with no results found."""
    mock_maps_client.geocode.return_value = {"results": [], "status": "ZERO_RESULTS"}

    result = await geocoding_service.geocode("nonexistent address")

    assert result is None
