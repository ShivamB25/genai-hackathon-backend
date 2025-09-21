from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from src.database.firestore_client import FirestoreClient

pytestmark = pytest.mark.asyncio


async def test_create_trip_and_get_status(
    async_client: AsyncClient, mock_firestore_client: MagicMock
):
    """
    Tests that a trip can be created and its status can be retrieved.
    """
    # Mock Firestore behavior
    mock_firestore_client.get_document.return_value = None
    mock_firestore_client.set_document.return_value = None

    # Create a trip
    response = await async_client.post(
        "/trips",
        json={
            "destination": "Paris",
            "start_date": "2025-10-10",
            "end_date": "2025-10-15",
            "user_preferences": ["museums", "food"],
        },
    )
    assert response.status_code == 202
    assert "trip_id" in response.json()
    trip_id = response.json()["trip_id"]

    # Check the trip status
    response = await async_client.get(f"/trips/{trip_id}/status")
    assert response.status_code == 200
    assert "status" in response.json()
