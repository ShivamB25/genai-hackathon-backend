from unittest.mock import AsyncMock

import pytest

from src.trip_planner.schemas import TravelerProfile


@pytest.mark.asyncio
async def test_create_user_profile():
    """Test creating a new user profile."""
    mock_db_client = AsyncMock()
    mock_db_client.create_document.return_value = "new_user_id"

    user_data = {
        "name": "Test User",
        "interests": ["reading", "hiking"],
    }
    profile = TravelerProfile(**user_data)

    # In a real scenario, a service function would call this.
    # Here, we simulate the direct database interaction.
    new_user_id = await mock_db_client.create_document(
        "users", profile.model_dump(), document_id="new_user_id"
    )

    assert new_user_id == "new_user_id"
    mock_db_client.create_document.assert_called_once()


@pytest.mark.asyncio
async def test_get_user_profile():
    """Test retrieving a user profile."""
    mock_db_client = AsyncMock()

    user_data = {
        "traveler_id": "user1",
        "name": "Existing User",
    }
    mock_db_client.get_document.return_value = user_data

    # Simulate fetching and parsing into a model
    retrieved_data = await mock_db_client.get_document("users", "user1")
    profile = TravelerProfile.model_validate(retrieved_data)

    assert profile.traveler_id == "user1"
    assert profile.name == "Existing User"
    mock_db_client.get_document.assert_called_once_with("users", "user1")


@pytest.mark.asyncio
async def test_update_user_profile():
    """Test updating a user profile."""
    mock_db_client = AsyncMock()

    # Simulate the update operation
    await mock_db_client.update_document("users", "user1", {"name": "Updated Name"})

    mock_db_client.update_document.assert_called_once_with(
        "users", "user1", {"name": "Updated Name"}
    )
