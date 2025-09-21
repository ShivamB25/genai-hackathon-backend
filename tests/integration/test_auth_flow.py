import pytest
from httpx import AsyncClient

from tests.utils import generate_mock_jwt  # Import the mock JWT generator

pytestmark = pytest.mark.asyncio


async def test_access_protected_route_with_valid_token(async_client: AsyncClient):
    """
    Tests that a protected route can be accessed with a valid token.
    """
    token = generate_mock_jwt(user_id="test-user-1", email="user@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    response = await async_client.get("/users/me", headers=headers)
    assert response.status_code == 200
    assert "user_id" in response.json()


async def test_access_protected_route_without_token(async_client: AsyncClient):
    """
    Tests that a protected route cannot be accessed without a token.
    """
    response = await async_client.get("/users/me")
    assert response.status_code == 401
