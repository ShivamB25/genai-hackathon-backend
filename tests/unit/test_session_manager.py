from unittest.mock import AsyncMock, patch

import pytest

from src.ai_services.session_manager import SessionNotFoundError, get_session_manager


@pytest.fixture
def session_manager():
    """Fixture for a SessionManager with a mocked Firestore client."""
    with patch(
        "src.ai_services.session_manager.get_firestore_client"
    ) as mock_get_firestore:
        mock_firestore = mock_get_firestore.return_value
        manager = get_session_manager()
        manager._firestore_client = mock_firestore
        # Use an in-memory dict for the test store
        manager._active_sessions = {}
        yield manager


@pytest.mark.asyncio
async def test_create_session(session_manager):
    """Test creating a new session."""
    session_id = "session1"
    user_id = "user1"

    session_manager._firestore_client.update_document = create_async_mock()

    session = await session_manager.create_session(
        user_id=user_id, initial_context={"theme": "travel"}
    )

    assert session.user_id == user_id
    assert session.context.session_metadata["theme"] == "travel"


@pytest.mark.asyncio
async def test_get_session_found(session_manager):
    """Test retrieving an existing session."""
    session = await session_manager.create_session(user_id="user1")

    found_session = await session_manager.get_session(session.session_id)

    assert found_session is not None
    assert found_session.user_id == "user1"


@pytest.mark.asyncio
async def test_get_session_not_found(session_manager):
    """Test that retrieving a nonexistent session raises an error."""
    session_manager._firestore_client.get_document.return_value = None
    with pytest.raises(SessionNotFoundError):
        await session_manager.get_session("nonexistent")


@pytest.mark.asyncio
async def test_update_session(session_manager):
    """Test updating a session's context."""
    session = await session_manager.create_session(user_id="user1")

    session.update_context(new_data="value")
    await session_manager.update_session(session)

    updated_session = await session_manager.get_session(session.session_id)
    assert updated_session.context.session_metadata["new_data"] == "value"


# Helper for mocking async functions
def create_async_mock(*args, **kwargs):
    m = AsyncMock(*args, **kwargs)

    async def mock_coro(*args, **kwargs):
        return m(*args, **kwargs)

    return mock_coro
