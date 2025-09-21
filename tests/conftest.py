import asyncio
import os
from unittest.mock import MagicMock, patch

import firebase_admin
import pytest
from firebase_admin import credentials
from google.cloud import firestore
from httpx import AsyncClient

# Set environment variables for testing
os.environ["ENV"] = "testing"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/mock/credentials.json"

from src.ai_services.model_config import get_async_client
from src.auth.dependencies import get_current_user
from src.core.config import settings
from src.database.firestore_client import get_firestore_client
from src.main import app
from src.maps_services.maps_client import get_maps_client


# Mock Firebase credentials
class MockCredentials(credentials.ApplicationDefault):
    def get_access_token(self):
        return MagicMock()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def initialize_firebase_app():
    """Initialize a mock Firebase app for the test session."""
    if not firebase_admin._apps:
        cred = MockCredentials()
        firebase_admin.initialize_app(cred, name="test_app")
    # No teardown needed for the mock app


@pytest.fixture
def mock_settings():
    """Fixture to mock application settings."""
    settings.firebase_project_id = "test-project"
    settings.google_maps_api_key = "test_maps_key"
    return settings


@pytest.fixture
def mock_firebase_user():
    """Fixture for a mock Firebase user."""
    return {"uid": "test_user_123", "email": "test@example.com"}


def override_get_current_user():
    """Override dependency to return a mock user."""
    return {"uid": "test_user_123", "email": "test@example.com"}


@pytest.fixture
def test_firestore_client():
    """Fixture for a test Firestore client connected to the emulator."""
    # Ensure Firestore emulator is running
    os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
    client = firestore.AsyncClient(
        project=settings.firebase_project_id,
        credentials=MockCredentials(),
    )
    return client


@pytest.fixture
def mock_google_maps_client():
    """Fixture for a mock Google Maps client."""
    mock_client = MagicMock()
    mock_client.places.return_value = {"results": [], "status": "OK"}
    mock_client.directions.return_value = []
    mock_client.geocode.return_value = []
    return mock_client


@pytest.fixture
def mock_vertex_ai_client():
    """Fixture for a mock Vertex AI client."""
    with patch("src.ai_services.model_config.GenerativeModel") as mock_generative_model:
        mock_model_instance = MagicMock()
        mock_generative_model.return_value = mock_model_instance
        yield mock_model_instance


@pytest.fixture
async def test_app(
    mock_settings, test_firestore_client, mock_google_maps_client, mock_vertex_ai_client
):
    """Fixture to create a test application instance with mocked dependencies."""
    app.dependency_overrides[get_firestore_client] = lambda: test_firestore_client
    app.dependency_overrides[get_maps_client] = lambda: mock_google_maps_client
    app.dependency_overrides[get_async_client] = lambda: mock_vertex_ai_client
    app.dependency_overrides[get_current_user] = override_get_current_user

    yield app

    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(test_app):
    """Fixture for an async test client."""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client
