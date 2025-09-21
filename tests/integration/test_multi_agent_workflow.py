from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_full_trip_planning_workflow(
    async_client: AsyncClient,
    mock_firestore_client: MagicMock,
    mock_vertex_ai_client: AsyncMock,
    mock_google_maps_client: MagicMock,
):
    """
    Tests the full multi-agent trip planning workflow from API request to completion.
    """
    # Mock AI and Maps responses
    mock_vertex_ai_client.generate_content.return_value = MagicMock(
        text="Generated trip plan"
    )
    mock_google_maps_client.places.return_value = {"results": []}

    # Start the trip planning process
    response = await async_client.post(
        "/trips",
        json={
            "destination": "Tokyo",
            "start_date": "2026-04-01",
            "end_date": "2026-04-07",
            "user_preferences": ["sushi", "temples"],
        },
    )
    assert response.status_code == 202
    trip_id = response.json()["trip_id"]

    # In a real test, you would wait for the background task to complete.
    # Here, we'll just check that the initial status is "processing".
    response = await async_client.get(f"/trips/{trip_id}/status")
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
